import torch

from typing import List
from dataclasses import dataclass

from diffulex.attention.metadata import AttnMetaDataBase
from diffulex.strategy.d2f.engine.sequence import D2FSequence


@dataclass
class D2FAttnMetaData(AttnMetaDataBase):
    seq_lens: list[int] = None
    seq_lens_ts: torch.Tensor | None = None
    d2f_pp: bool = False
    block_mask: torch.Tensor | None = None
    seqs: List[D2FSequence] = None
    kv_cache_layout: str = "unified"
    need_kv_cache_store: bool = True
    
    def __post_init__(self):
        if self.seq_lens_ts is not None and self.context_lens is not None:
            self.total_lens = self.seq_lens_ts + self.context_lens
        if not self.is_prefill and self.d2f_pp:
            return
        if self.seqs is not None and len(self.seqs) > 0:
            if self.is_prefill:
                masks = [seq.current_block_mask for seq in self.seqs]
                total_len = sum(mask.size(-1) for mask in masks)
                self.block_mask = torch.zeros(total_len, total_len, dtype=torch.bool)
                
                start_idx = 0
                for mask in masks:
                    seq_len = mask.size(-1)
                    end_idx = start_idx + seq_len
                    self.block_mask[start_idx:end_idx, start_idx:end_idx] = mask.clone()
                    start_idx = end_idx
                self.block_mask = self.block_mask.to(mask.device)
            else:
                masks = [seq.current_block_mask for seq in self.seqs]
                total_height = sum(mask.size(-2) for mask in masks)
                total_width = sum(mask.size(-1) for mask in masks)
                self.block_mask = torch.zeros(total_height, total_width, dtype=torch.bool)
                start_row = 0
                start_col = 0
                for mask in masks:
                    height, width = mask.size(-2), mask.size(-1)
                    end_row = start_row + height
                    end_col = start_col + width
                    self.block_mask[start_row:end_row, start_col:end_col] = mask.clone()
                    start_row, start_col = end_row, end_col
                self.block_mask = self.block_mask.to(mask.device)
    
    @property
    def total_num_seqs(self) -> int:
        return len(self.seqs) if self.seqs is not None else 0


D2F_ATTN_METADATA = D2FAttnMetaData()

def fetch_d2f_attn_metadata() -> D2FAttnMetaData:
    return D2F_ATTN_METADATA

def set_d2f_attn_metadata(
    is_prefill: bool = False,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    block_tables: torch.Tensor | None = None,
    seqs: List[D2FSequence] | None = None,
    seq_lens: list[int] | None = None,
    seq_lens_ts: torch.Tensor | None = None,
    kv_cache_layout: str = "unified",
    need_kv_cache_store: bool = True,
    d2f_pp: bool = False,
    block_mask: torch.Tensor | None = None,
) -> None:
    global D2F_ATTN_METADATA
    D2F_ATTN_METADATA = D2FAttnMetaData(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        seq_lens=seq_lens,
        seq_lens_ts=seq_lens_ts,
        d2f_pp=d2f_pp,
        block_mask=block_mask,
        seqs=seqs,
        kv_cache_layout=kv_cache_layout,
        need_kv_cache_store=need_kv_cache_store,
    )

def reset_d2f_attn_metadata() -> None:
    global D2F_ATTN_METADATA
    D2F_ATTN_METADATA = D2FAttnMetaData()