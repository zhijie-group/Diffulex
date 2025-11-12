import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    lora_path: str = ""
    model_name: str = "dream"
    model_type: str = "diffusion_lm" # "causal_lm" or "diffusion_lm"
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
    shm_name: str = "d2f_vllm"
    # Start device index for this TP group (set by DP launcher).
    device_start: int = 0
    
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    k_cache_hdim_split_factor_x: int = 8
    kv_cache_layout: str = "unified"  # "unified" or "distinct"

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
                print(f"Warning: LoRA path {self.lora_path} does not exist")

        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        cfg_max_model_len = self.hf_config.max_position_embeddings if hasattr(self.hf_config, "max_position_embeddings") else self.hf_config.max_sequence_length
        self.max_model_len = min(self.max_model_len, cfg_max_model_len)
        assert self.max_num_batched_tokens >= self.max_model_len