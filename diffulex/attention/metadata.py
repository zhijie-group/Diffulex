import torch

from typing import Callable
from dataclasses import dataclass


@dataclass
class AttnMetaDataBase:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    page_block_size: int = 32
    attn_type: str = "block_attention"
    diffusion_block_size: int = 32
    decode_mode: str = "static"
    k_scale: torch.Tensor | None = None  # Quantization scale for K cache, shape [num_kv_heads]
    v_scale: torch.Tensor | None = None  # Quantization scale for V cache, shape [num_kv_heads]
    q_scale: torch.Tensor | None = None  # Quantization scale for Q, strategy-defined shape (e.g. [num_heads] or [1])
    
    @property
    def num_seqs(self) -> int:
        return len(self.cu_seqlens_q) - 1

FN_TYPE_AttnMetaDataFetch = Callable[[], AttnMetaDataBase]

fetch_attn_metadata: FN_TYPE_AttnMetaDataFetch = ...

def set_fetch_fn_for_attn_metadata(fn: FN_TYPE_AttnMetaDataFetch) -> None:
    global fetch_attn_metadata
    fetch_attn_metadata = fn
    
WARMING_UP = False

def set_warming_up(is_warming_up: bool) -> None:
    global WARMING_UP
    WARMING_UP = is_warming_up

def is_warming_up() -> bool:
    return WARMING_UP

def reset_warming_up() -> None:
    global WARMING_UP
    WARMING_UP = False