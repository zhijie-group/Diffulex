import time
import torch
import pickle

import torch.distributed as dist

from typing import List
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from diffulex_legacy.config import Config
from diffulex_legacy.engine.sequence import SequenceForCausalLM, SequenceForDiffusionLM, SequenceBase
from diffulex_legacy.models.auto_model import AutoModelLM
from diffulex_legacy.layers.sampler import AutoSampler
from diffulex_legacy.utils.checker import CHECK_SLOT_MAPPING
from diffulex_legacy.utils.context import (
    set_context_causal_lm, 
    get_context_causal_lm, 
    reset_context_causal_lm,
    set_context_diffusion_lm,
    get_context_diffusion_lm,
    reset_context_diffusion_lm
)
from diffulex.utils.kv_cache_dtype import parse_kv_cache_dtype


def _get_kv_cache_storage_info(kv_cache_dtype: str) -> tuple[torch.dtype, int]:
    """
    Returns (storage_dtype, itemsize) for KV cache allocation.
    - For FP8: returns (torch.uint8, 1) because FP8 uses uint8 storage
    - For other dtypes: returns (torch.bfloat16/fp16/fp32, itemsize)
    """
    spec = parse_kv_cache_dtype(kv_cache_dtype)
    if spec.is_fp8:
        return torch.uint8, 1
    elif spec.enum.value == 0:  # BF16
        return torch.bfloat16, 2
    elif spec.enum.value == 1:  # FP16
        return torch.float16, 2
    elif spec.enum.value == 2:  # FP32
        return torch.float32, 4
    else:
        raise ValueError(f"Unsupported kv_cache_dtype: {kv_cache_dtype}")


class ModelRunnerBase(ABC):
    """Base class for model runners supporting different model types."""
    def __init__(self, config: Config, rank: int, event: Event | List[Event]):
        self.config = config
        self.model_type = config.model_type
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
        self.model = AutoModelLM.from_config(config)
        self.sampler = AutoSampler.from_config(config)
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
                shm_size = 2**25 if self.model_type == "diffusion_lm" else 2**20
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

    @abstractmethod
    def warmup_model(self):
        """Model-specific warmup logic."""
        pass

    @abstractmethod
    def allocate_kv_cache(self):
        pass

    def prepare_block_tables(self, seqs: List[SequenceBase]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    @abstractmethod
    def prepare_prefill(self, seqs: List[SequenceBase]):
        """Model-specific prefill preparation."""
        pass

    @abstractmethod
    def prepare_decode(self, seqs: List[SequenceBase]):
        """Model-specific decode preparation."""
        pass

    def prepare_sample(self, seqs: List[SequenceBase]):
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
    def run(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        """Main inference pipeline."""
        pass

    @abstractmethod
    @torch.inference_mode()
    def capture_cudagraph(self):
        """Model-specific CUDA graph capture."""
        pass


class ModelRunnerForCausalLM(ModelRunnerBase):
    """Model runner for Causal Language Models."""
    def warmup_model(self):
        # return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [SequenceForCausalLM(test_input_ids) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()
    
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        storage_dtype, itemsize = _get_kv_cache_storage_info(config.kv_cache_dtype)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        
        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")
        
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * 
                       head_dim * itemsize)
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - 
                                        used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # [kv_separated, layer_id, block_id, block_size(segmented seq_len), head, head_dim]
        self.kv_cache = torch.zeros(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
            self.block_size, num_kv_heads, head_dim, dtype=storage_dtype)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: List[SequenceForCausalLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context_causal_lm(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, kv_cache_dtype=self.config.kv_cache_dtype)
        return input_ids, positions

    def prepare_decode(self, seqs: List[SequenceForCausalLM]):
        input_ids = []
        positions = []
        cu_seqlens_k = [0]
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            seqlen_k = len(seq)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context_causal_lm(False, cu_seqlens_k=cu_seqlens_k, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, kv_cache_dtype=self.config.kv_cache_dtype)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context_causal_lm()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run_verbose(self, seqs: List[SequenceForCausalLM], is_prefill: bool) -> List[int]:
        print("= =" * 20)
        print(f"Running {'prefill' if is_prefill else 'decode'} for {len(seqs)} sequences on rank {self.rank}")
        s = time.time()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        print(f"Prepared input in {time.time() - s:.2f} seconds")
        s = time.time()
        logits = self.run_model(input_ids, positions, is_prefill)
        print(f"Ran model in {time.time() - s:.2f} seconds")
        s = time.time()
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        print(f"Sampled tokens in {time.time() - s:.2f} seconds")
        reset_context_causal_lm()
        return token_ids

    def run(self, seqs: List[SequenceForCausalLM], is_prefill: bool) -> List[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context_causal_lm()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context_causal_lm(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], kv_cache_dtype=self.config.kv_cache_dtype)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context_causal_lm()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )


class ModelRunnerForDiffusionLM(ModelRunnerBase):
    """Model runner for Diffusion Language Models. TODO: Implement DLM-specific logic."""
    def __init__(self, config: Config, rank: int, event: Event | List[Event]):
        super().__init__(config, rank, event)
        self.diffusion_block_size = config.diffusion_block_size
        self.mask_token_id = config.mask_token_id
        self.decoding_strategy = config.decoding_strategy
            
    def warmup_model(self):
        # return
        print("Warming up model...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        test_input_ids = [0] * max_model_len
        seqs = [SequenceForDiffusionLM(test_input_ids, config=self.config) for _ in range(num_seqs)]
        self.run(seqs, True)
        for seq in seqs:
            seq.post_process()
        torch.cuda.empty_cache()
        
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        storage_dtype, itemsize = _get_kv_cache_storage_info(config.kv_cache_dtype)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = getattr(hf_config, "num_key_value_heads", getattr(hf_config, "n_kv_heads", None)) // self.world_size

        if hasattr(hf_config, 'head_dim'):
            head_dim = hf_config.head_dim
        elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise AttributeError(f"Cannot determine head_dim from config: {type(hf_config)}")
        
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * itemsize)
        get_num_kvcache_blocks = lambda gpu_memory_utilization: int(total * gpu_memory_utilization -  # noqa: E731
                                                                    used - peak + current) // block_bytes
        try:
            num_kvcache_blocks = get_num_kvcache_blocks(config.gpu_memory_utilization)
            assert num_kvcache_blocks > 0
        except:  # noqa: E722
            gpu_memory_utilization = config.gpu_memory_utilization
            while num_kvcache_blocks <= 200: 
                print(f"Warning: GPU memory utilization {gpu_memory_utilization} is too low to allocate kv cache. "
                    f"Automatically adding 0.05, which is {gpu_memory_utilization + 0.05:.2f} now.")
                gpu_memory_utilization += 0.05
                num_kvcache_blocks = get_num_kvcache_blocks(gpu_memory_utilization)
            print(f"Set gpu_memory_utilization to {gpu_memory_utilization:.2f} to allocate kv cache.")
            config.gpu_memory_utilization = gpu_memory_utilization
            
        config.num_kvcache_blocks = num_kvcache_blocks
        print(f"Allocated {config.num_kvcache_blocks} blocks of size {self.block_size} for kv cache on rank {self.rank}.")

        if config.kv_cache_layout == "distinct":
            # k_cache: [layer_id, block_id, head, head_dim // x, block_size(segmented seq_len), x]
            # v_cache: [layer_id, block_id, head, head_dim, block_size(segmented seq_len)]
            x = config.k_cache_hdim_split_factor_x
            
            self.k_cache = torch.zeros(
                hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                num_kv_heads, head_dim // x, self.block_size, x, dtype=storage_dtype
            )
            self.v_cache = torch.zeros(
                hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                num_kv_heads, head_dim, self.block_size, dtype=storage_dtype
            )
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.k_cache[layer_id]
                    module.v_cache = self.v_cache[layer_id]
                    layer_id += 1
        elif config.kv_cache_layout == "unified":
            # [kv_separated, layer_id, block_id, block_size(segmented seq_len), head, head_dim]
            self.kv_cache = torch.zeros(
                2, hf_config.num_hidden_layers, config.num_kvcache_blocks, 
                self.block_size, num_kv_heads, head_dim, dtype=storage_dtype)
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
        else:
            raise ValueError(f"Unsupported kv_cache_layout: {config.kv_cache_layout}. "
                             f"Supported values are 'distinct' and 'unified'.")

    def prepare_prefill(self, seqs: List[SequenceForDiffusionLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        context_lens = []
        seq_lens = []

        for seq in seqs:
            seq.next_diffusion_step(is_prefill=True)

            total_seqlen = len(seq)
            # tokens and positions to run in this prefill step
            input_ids.extend(seq[seq.cached_num_tokens:])
            positions.extend(list(range(seq.cached_num_tokens, total_seqlen)))
            seq_lens.append(total_seqlen)
            context_lens.append(0)
            assert len(input_ids) == len(positions), (
                f"prepare_prefill(diffusion): len(input_ids) {len(input_ids)} != len(positions) {len(positions)}"
            )
            
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                continue
            # build slot mapping for prefix cache prompt blocks
            for i in range(0, seq.num_prompt_blocks):
                if seq.block_cache_missed[i]:
                    start = seq.block_table[i] * self.block_size
                    if i != seq.num_prompt_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_prompt_num_tokens
                    slot_mapping.extend(list(range(start, end)))
                else:
                    slot_mapping.extend([-1] * self.block_size)
            # pad to a full diffusion block
            slot_mapping.extend([-1] * seq.diffusion_block_size)

        # For diffusion prefill we always need block tables for prefix cache bookkeeping
        block_tables = self.prepare_block_tables(seqs)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # More checks to avoid downstream rotary errors
        assert cu_seqlens_q[-1].item() == input_ids.numel(), (
            f"prepare_prefill(diffusion): cu_seqlens_q[-1]={cu_seqlens_q[-1].item()} != num_tokens={input_ids.numel()}"
        )
        assert cu_seqlens_k[-1].item() == sum(seq_lens), (
            f"prepare_prefill(diffusion): cu_seqlens_k[-1]={cu_seqlens_k[-1].item()} != sum(seq_lens)={sum(seq_lens)}"
        )

        set_context_diffusion_lm(
            True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            seqs=seqs,
            kv_cache_layout=self.config.kv_cache_layout,
            seq_lens=seq_lens,
            seq_lens_ts=seq_lens_ts,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: List[SequenceForDiffusionLM]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        slot_mapping = []
        context_lens = []
        seq_lens = []
        seq_id_to_queue_id = {}
        need_kv_cache_store = False
        max_seqlen_q = 0
        max_seqlen_k = 0
        # if sum((sum(seq.active_blocks) + sum(seq.to_cache_blocks)) * seq.diffusion_block_size for seq in seqs) == 1536:
        #     pass
        for seq_idx_in_queue, seq in enumerate(seqs): 
            seq_id = seq.seq_id
            seq_id_to_queue_id[seq_id] = seq_idx_in_queue
            seq.next_diffusion_step()
            cur_input_ids, cur_positions, cur_context_len = seq.diffusion_decoding_inputs()
            
            seq_lens.append(len(cur_input_ids))
            input_ids.extend(cur_input_ids)
            positions.extend(cur_positions)
            context_lens.append(cur_context_len)
            
            total_seqlen = len(seq)
            seqlen_q = total_seqlen - seq.cached_num_tokens
            seqlen_k = total_seqlen
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)

            mem_block_to_diffusion_blocks_map = seq.mem_block_to_diffusion_blocks_map
            context_len = context_lens[seq_id_to_queue_id[seq_id]]
            for mem_block_idx in range(0, seq.num_blocks):
                start_idx = mem_block_idx * seq.block_size
                end_idx = start_idx + seq.block_size
                cur_map = mem_block_to_diffusion_blocks_map[mem_block_idx]
                is_last_block = False
                meet_active_block = False
                while start_idx < end_idx and not is_last_block and not meet_active_block:
                    local_start_idx = lambda: start_idx % seq.block_size
                    diffusion_block = seq.diffusion_blocks[cur_map[local_start_idx()]]
                    if diffusion_block.block_id == 0 and diffusion_block.cursor != start_idx:
                        diffusion_block.cursor = start_idx
                    if cur_map[local_start_idx()] == seq.num_diffusion_blocks - 1:
                        is_last_block = True
                    get_step = lambda diff_blk, start_idx: (
                        diff_blk.remaining_length(start_idx)
                        if diff_blk.remaining_length(start_idx) + local_start_idx() <= seq.block_size
                        else seq.block_size - local_start_idx()
                    )
                    if diffusion_block.is_in_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        start_idx += step
                    elif diffusion_block.is_to_cache:
                        step = get_step(diffusion_block, start_idx)
                        diffusion_block.cursor += step
                        cur_diffusion_block_start = 0
                        cur_diffusion_block_end = step
                        start_idx += step
                        mem_block_start = seq.block_table[mem_block_idx] * self.block_size + context_len % seq.block_size
                        context_len += step
                        slot_mapping.extend(list(range(mem_block_start + cur_diffusion_block_start,
                                                       mem_block_start + cur_diffusion_block_end)))
                        need_kv_cache_store = True
                    elif diffusion_block.is_active:
                        meet_active_block = True
                        
                if meet_active_block:
                    # Covering all the after-active blocks
                    active = seq.active_blocks
                    first_active_idx = next((i for i, v in enumerate(active) if v), None)
                    if first_active_idx is not None:
                        num_blocks_to_pad = len(active) - first_active_idx
                        padding_slots = [-1] * (num_blocks_to_pad * seq.diffusion_block_size)
                        slot_mapping.extend(padding_slots)
                    break
            assert len(input_ids) == len(positions), f"Input IDs length {len(input_ids)} does not match positions length {len(positions)}"
            assert len(input_ids) == len(slot_mapping), f"Input IDs length {len(input_ids)} does not match slot mapping length {len(slot_mapping)}"

        # CHECK_SLOT_MAPPING(seqs, slot_mapping)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        seq_lens_ts = torch.tensor(seq_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context_diffusion_lm(False, slot_mapping=slot_mapping, context_lens=context_lens, 
                                 cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                                 max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
                                 block_tables=block_tables, seqs=seqs, 
                                 seq_lens=seq_lens, seq_lens_ts=seq_lens_ts, 
                                 kv_cache_layout=self.config.kv_cache_layout, need_kv_cache_store=need_kv_cache_store,
                                 d2f_pp=True)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context_diffusion_lm()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run_verbose(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        print("= =" * 20)
        print(f"Running {'prefill' if is_prefill else 'decode'} for {len(seqs)} sequences on rank {self.rank}")
        s = time.time()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        print(f"Prepared input in {time.time() - s:.2f} seconds")
        s = time.time()
        logits = self.run_model(input_ids, positions, is_prefill)
        print(f"Ran model in {time.time() - s:.2f} seconds")
        s = time.time()
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        print(f"Sampled tokens in {time.time() - s:.2f} seconds")
        reset_context_diffusion_lm()
        return sample_output

    def run(self, seqs: List[SequenceBase], is_prefill: bool) -> List[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        sample_output = self.sampler(logits, temperatures) if self.rank == 0 else None
        reset_context_diffusion_lm()
        return sample_output

    @torch.inference_mode()
    def capture_cudagraph(self):
        '''
            TODO: Varlen decoding does not support CUDA graph capture yet.
            Can be implemented, but requires drastically high overhead.
        '''
        raise NotImplementedError("CUDA graph capture for DiffusionLM is not implemented yet.")
    

class AutoModelRunner:
    @classmethod
    def from_config(cls, config: Config, rank: int, event: Event | List[Event]):
        """Factory method to create a model runner based on the model type."""
        if config.model_type == "causal_lm":
            return ModelRunnerForCausalLM(config, rank, event)
        elif config.model_type == "diffusion_lm":
            return ModelRunnerForDiffusionLM(config, rank, event)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")