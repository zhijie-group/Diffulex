import os
import torch

import torch.nn as nn

from functools import lru_cache, partial
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask 
from flash_attn import flash_attn_varlen_func
from transformers.integrations.flex_attention import compile_friendly_flex_attention as flex_attention

from diffulex.attention.ops import (
    causal_lm_flash_decoding, diffusion_lm_flash_decoding, diffusion_lm_parallel_flash_decoding,
    store_kvcache_unified_layout, store_kvcache_distinct_layout, load_kvcache,
    CHECK_STORING, CHECK_LOADING, CHECK_ATTENTION
)
from diffulex.attention.metadata import AttnMetaDataBase, fetch_attn_metadata


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        is_rtx_xx90 = lambda x: "4090" in x or "3090" in x
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        } if is_rtx_xx90(torch.cuda.get_device_name(0)) else None
        self.attention = torch.compile(
            partial(flex_attention, kernel_options=kernel_options, enable_gqa=True, 
                    return_lse=False, training=False), dynamic=True)
        self._block_mask_cache = {}

    @lru_cache(maxsize=32)
    def dllm_block_mask(self, block_mask: torch.Tensor, 
                        B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        cache_key = (B, H, Q_LEN, KV_LEN, device)
        def _mask_mod(batch, head, token_q, token_kv):
            return block_mask[token_q, token_kv]
        if cache_key not in self._block_mask_cache:
            self._block_mask_cache[cache_key] = create_block_mask(
                _mask_mod, B, H, Q_LEN, KV_LEN, device=device
            )
        return self._block_mask_cache[cache_key]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: list[torch.Tensor] | None = None) -> torch.Tensor:
        # Reshape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        attn_metadata: AttnMetaDataBase = fetch_attn_metadata()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = attn_metadata.kv_cache_layout == "unified"

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if not (not attn_metadata.need_kv_cache_store):
                store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
                store_kvcache(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)
                # CHECK_STORING(k_cache, v_cache, k, v, context)

        transpose_fn = lambda x: rearrange(x, 's h d -> 1 h s d').contiguous()
        # Prefill / Decode logic TODO: Replace the Flex Attention Prefilling
        if attn_metadata.is_prefill:
            # Block PK
            if attn_metadata.block_tables is not None:
                # TODO: Implement Prefix Caching
                pass

            # Attention computation
            q_t, k_t, v_t = [transpose_fn(t) for t in (q, k, v)]

            B, H, S, _ = q_t.shape
            block_mask = self.dllm_block_mask(attn_metadata.block_mask, B, H, S, S, str(q.device))
            o = self.attention(q_t, k_t, v_t, block_mask=block_mask)
        else:
            config = attn_metadata.seqs[0].config
            diffusion_block_size = config.diffusion_block_size
            if is_unified_layout:
                k_comb, v_comb = load_kvcache(self.k_cache, self.v_cache, attn_metadata, k, v)
                o = flash_attn_varlen_func(q, k_comb, v_comb, 
                                            attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
                                            attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
                                            softmax_scale=self.scale, block_table=None)
            else:
                # FIXME: Kernel not ok...
                o = torch.empty_like(q).to(q.device).to(q.dtype)
                q, k, o, k_cache, v_cache = map(lambda x: x.to(torch.float32), (q, k, o, k_cache, v_cache))
                diffusion_lm_parallel_flash_decoding(
                    q, k, v, o, str(k_cache.dtype), k_cache, v_cache, 
                    attn_metadata.block_tables, attn_metadata.cu_seqlens_q, attn_metadata.total_lens,
                    max(attn_metadata.total_lens), max(attn_metadata.seq_lens), 1.0, 1.0,
                    diffusion_block_size, attn_metadata.block_mask
                )
                CHECK_ATTENTION(o, q, k, v, k_cache, v_cache, attn_metadata)
            
        # Final reshape
        if not attn_metadata.is_prefill:
            o = o.view(-1, self.num_heads * self.head_dim).contiguous()
        elif attn_metadata.is_prefill:
            o = rearrange(o, '1 h s d -> s (h d)').contiguous()

        return o