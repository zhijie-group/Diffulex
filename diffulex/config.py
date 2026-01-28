import os

from dataclasses import dataclass, field
from transformers import AutoConfig
from diffulex.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Config:
    model: str
    lora_path: str = ""
    model_name: str = "dream"
    decoding_strategy: str = "d2f" # "d2f", "fast-dllm-v2", "block-diffusion"
    
    mask_token_id: int = 151666
    diffusion_block_size: int = 32
    
    accept_threshold: float = 0.9
    complete_threshold: float = 0.95
    add_new_block_threshold: float = 0.1
    
    use_lora: bool = False
    max_num_batched_tokens: int = 4096
    max_num_seqs: int = 128
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.9
    
    data_parallel_size: int = 1
    tensor_parallel_size: int = 2
    # Distributed comm (per tensor-parallel group). When using multiple DP
    # replicas on one host, assign unique master_port per replica.
    master_addr: str = "localhost"
    master_port: int = 2333
    # Shared memory segment name for intra-TP RPC; must be unique per DP group.
    shm_name: str = "diffulex_shm"
    # Start device index for this TP group (set by DP launcher).
    device_start: int = 0
    device_ids: list[int] = field(default_factory=lambda: [])
    
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 32
    num_kvcache_blocks: int = -1
    k_cache_hdim_split_factor_x: int = 8
    kv_cache_layout: str = "unified"  # "unified" or "distinct"
    kv_cache_dtype: str = "bf16"  # "bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2"
    decode_mode: str | None = None  # "static" or "varlen", None means auto-select based on kv_cache_dtype
    # Attention-Q dtype (activation quantization). "bf16" default; "fp8" is a placeholder
    # for future kernels (enabling it will currently raise NotImplementedError at runtime).
    attn_q_dtype: str = "bf16"
    # Linear quantization (weights + activations). All are placeholders for future kernels.
    # Use "bf16" to disable quantization.
    # Supported aliases (normalized in registry): bf16/int8/int4/fp8/fp8_e4m3/fp8_e5m2/gptq/awq.
    linear_attn_weight_dtype: str = "bf16"
    linear_mlp_weight_dtype: str = "bf16"
    linear_attn_act_dtype: str = "bf16"
    linear_mlp_act_dtype: str = "bf16"

    # Kernel tuning knobs (avoid environment-variable based tuning in library code).
    # Currently used by some W8A16 linear strategies.
    linear_w8a16_quant_block_n: int = 256
    linear_w8a16_allspark_cublas_m_threshold: int = 256

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 16 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert 1 <= self.data_parallel_size <= 1024
        assert isinstance(self.master_port, int) and 0 < self.master_port < 65536
        assert isinstance(self.device_start, int) and self.device_start >= 0

        # LoRA validation
        if self.use_lora:
            if not self.lora_path:
                raise ValueError("lora_path must be provided when use_lora is True")
            if not os.path.exists(self.lora_path):
                logger.warning(f"LoRA path {self.lora_path} does not exist")

        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        cfg_max_model_len = self.hf_config.max_position_embeddings if hasattr(self.hf_config, "max_position_embeddings") else self.hf_config.max_sequence_length
        self.max_model_len = min(self.max_model_len, cfg_max_model_len)
        assert self.max_num_batched_tokens >= self.max_model_len
        
        if not self.device_ids:
            import torch
            # When CUDA_VISIBLE_DEVICES is set, PyTorch maps physical devices to logical device 0, 1, ...
            # So we should use logical device indices (0, 1, ...) instead of physical device IDs
            self.device_ids = list(range(torch.cuda.device_count()))
            logger.info(f"Using CUDA devices: {self.device_ids}")