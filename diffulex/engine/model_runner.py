import torch
import pickle

import torch.distributed as dist

from typing import Callable
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from diffulex.config import Config
from diffulex.sampler import AutoSampler
from diffulex.engine.sequence import SequenceBase
from diffulex.model import AutoModelForDiffusionLM
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry
from diffulex.utils.quantization.factory import QuantizationStrategyFactory
from diffulex.utils.quantization.context import get_kv_cache_strategy
from diffulex.utils.quantization.strategies import NoQuantizationStrategy


class ModelRunnerBase(ABC):
    """Base class for model runners supporting different model types."""
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Initialize model, sampler, and kv cache
        init_method = f"tcp://{config.master_addr}:{config.master_port}"
        dist.init_process_group("nccl", init_method, world_size=self.world_size, rank=rank)
        device_id = (getattr(config, "device_start", 0) or 0) + rank
        assert 0 <= device_id < torch.cuda.device_count(), f"Invalid device_id {device_id}."
        torch.cuda.set_device(device_id)
        default_dtype = torch.get_default_dtype()
        default_dtype = (hf_config.torch_dtype if hasattr(hf_config, "torch_dtype") 
                         and hf_config.torch_dtype else torch.bfloat16)
        torch.set_default_dtype(default_dtype)
        torch.set_default_device(f"cuda:{device_id}")
        self.model = self.load_model(config)
        self.sampler = self.load_sampler(config)
        # Initialize quantization context
        QuantizationStrategyFactory.create_from_config(config)
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()

        # Allocate shared memory for inter-process communication
        # NOCHANGE
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        if self.world_size > 1:
            if rank == 0:
                try:
                    shm = SharedMemory(name=config.shm_name)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass
                shm_size = 2**25
                self.shm = SharedMemory(name=config.shm_name, create=True, size=shm_size)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=config.shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        
        if n + 4 > len(self.shm.buf):
            raise ValueError(f"Serialized data size ({n} bytes) exceeds shared memory buffer size ({len(self.shm.buf)} bytes). "
                           f"Consider increasing shared memory size or reducing batch size.")
        
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def load_model(self, config: Config):
        """Instantiate the underlying model; override to customize."""
        return AutoModelForDiffusionLM.from_config(config)

    def load_sampler(self, config: Config):
        """Instantiate the sampler implementation; override to customize."""
        return AutoSampler.from_config(config)

    @abstractmethod
    def warmup_model(self):
        """Model-specific warmup logic."""
        pass

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = getattr(
            hf_config,
            "num_key_value_heads",
            getattr(hf_config, "n_kv_heads", None),
        ) // self.world_size

        if hasattr(hf_config, "head_dim"):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")

        # Get storage dtype and itemsize from quantization strategy
        strategy = get_kv_cache_strategy()
        if strategy is None:
            strategy = NoQuantizationStrategy()
        storage_dtype, itemsize = strategy.get_storage_dtype()
        
        block_bytes = (
            2
            * hf_config.num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * itemsize
        )
        get_num_kvcache_blocks = (
            lambda gpu_memory_utilization: int(total * gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        try:
            num_kvcache_blocks = get_num_kvcache_blocks(config.gpu_memory_utilization)
            assert num_kvcache_blocks > 0
        except Exception:
            gpu_memory_utilization = config.gpu_memory_utilization
            while num_kvcache_blocks <= 200:
                print(
                    "Warning: GPU memory utilization "
                    f"{gpu_memory_utilization} is too low to allocate kv cache. "
                    "Automatically adding 0.05."
                )
                gpu_memory_utilization += 0.05
                num_kvcache_blocks = get_num_kvcache_blocks(gpu_memory_utilization)
            print(
                f"Set gpu_memory_utilization to {gpu_memory_utilization:.2f} "
                "to allocate kv cache."
            )
            config.gpu_memory_utilization = gpu_memory_utilization

        config.num_kvcache_blocks = num_kvcache_blocks
        print(
            "Allocated {num_blocks} blocks of size {block_size} for kv cache on rank {rank}.".format(
                num_blocks=config.num_kvcache_blocks,
                block_size=self.block_size,
                rank=self.rank,
            )
        )

        if config.kv_cache_layout == "distinct":
            x = config.k_cache_hdim_split_factor_x
            self.k_cache = torch.zeros(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                num_kv_heads,
                head_dim // x,
                self.block_size,
                x,
                dtype=storage_dtype,
            )
            self.v_cache = torch.zeros(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                num_kv_heads,
                head_dim,
                self.block_size,
                dtype=storage_dtype,
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.k_cache[layer_id]
                    module.v_cache = self.v_cache[layer_id]
                    layer_id += 1
        elif config.kv_cache_layout == "unified":
            self.kv_cache = torch.zeros(
                2,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                num_kv_heads,
                head_dim,
                dtype=storage_dtype,
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
        else:
            raise ValueError(
                "Unsupported kv_cache_layout: {layout}. Supported values are 'distinct' and 'unified'.".format(
                    layout=config.kv_cache_layout
                )
            )
        
        # Allocate scale tensors if quantization strategy requires them
        # Get device from cache (already allocated above)
        if config.kv_cache_layout == "distinct":
            device = self.k_cache.device
        else:  # unified
            device = self.kv_cache.device
        k_scale_init, v_scale_init = strategy.init_scales(num_kv_heads, device)
        if k_scale_init is not None and v_scale_init is not None:
            # Allocate scale tensors: [num_layers, num_kv_heads]
            self.k_scale = torch.zeros(
                hf_config.num_hidden_layers, num_kv_heads,
                dtype=torch.float32, device=device
            )
            self.v_scale = torch.zeros(
                hf_config.num_hidden_layers, num_kv_heads,
                dtype=torch.float32, device=device
            )
            # Initialize with strategy's initial scale values
            self.k_scale[:] = k_scale_init[None, :]
            self.v_scale[:] = v_scale_init[None, :]
            
            # Bind scales to Attention modules
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_scale = self.k_scale[layer_id]
                    module.v_scale = self.v_scale[layer_id]
                    layer_id += 1

    def prepare_block_tables(self, seqs: list[SequenceBase]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    @abstractmethod
    def prepare_prefill(self, seqs: list[SequenceBase]):
        """Model-specific prefill preparation."""
        pass

    @abstractmethod
    def prepare_decode(self, seqs: list[SequenceBase]):
        """Model-specific decode preparation."""
        pass

    def prepare_sample(self, seqs: list[SequenceBase]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @abstractmethod
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Model-specific forward pass."""
        pass

    @abstractmethod
    def run(self, seqs: list[SequenceBase], is_prefill: bool) -> list[int]:
        """Main inference pipeline."""
        pass

    @abstractmethod
    @torch.inference_mode()
    def capture_cudagraph(self):
        """Model-specific CUDA graph capture."""
        pass
    

RunnerFactory = Callable[[Config, int, Event | list[Event]], "ModelRunnerBase"]


class AutoModelRunner(DiffulexStrategyRegistry):
    """Registry and factory that selects a ModelRunner implementation based on the configured decoding strategy.

        Example:
            >>> @AutoModelRunner.register("my_strategy")
            ... class MyRunner(ModelRunnerBase):
            ...     ...

        This allows `LLMEngine` to instantiate the appropriate runner using `Config.decoding_strategy`.
    """

    @classmethod
    def from_config(cls, config: Config, rank: int, event: Event | list[Event]):
        # Ensure project root is in sys.path for spawn mode subprocesses
        import sys
        import os
        if not any('diffulex_kernel' in p for p in sys.path):
            # Try to find project root by locating diffulex package
            diffulex_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if os.path.basename(diffulex_path) == 'diffulex':
                project_root = os.path.dirname(diffulex_path)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
        cls._MODULE_MAPPING: dict[str, RunnerFactory]
        candidates: list[str] = []
        for attr in ("decoding_strategy",):
            value = getattr(config, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        candidates.append(cls._DEFAULT_KEY)

        for key in candidates:
            factory = cls._MODULE_MAPPING.get(key)
            if factory is not None:
                return factory(config, rank, event)

        available = ", ".join(cls.available_modules()) or "<none>"
        raise ValueError(
            "No model runner registered for decoding_strategy="
            f"'{getattr(config, 'decoding_strategy', None)}'. Available runners: {available}."
        )
