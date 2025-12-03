import torch
import pickle

import torch.distributed as dist

from typing import Callable
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from diffulex.config import Config
from diffulex.layer.sampler import AutoSampler
from diffulex.engine.sequence import SequenceBase
from diffulex.model import AutoModelForDiffusionLM
from diffulex.engine.strategy_registry import DiffulexStrategyRegistry


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
        self.warmup_model()
        self.allocate_kv_cache()  # NOCHANGE
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

    @abstractmethod
    def allocate_kv_cache(self):
        pass

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
