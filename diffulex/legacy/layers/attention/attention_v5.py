import os
import torch

import torch.nn as nn

from typing import List
from functools import lru_cache, partial
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask 
from flash_attn import flash_attn_varlen_func
from transformers.integrations.flex_attention import compile_friendly_flex_attention as flex_attention

from diffulex.legacy.layers.attention.ops import (
    causal_lm_flash_decoding, diffusion_lm_flash_decoding, diffusion_lm_parallel_flash_decoding,
    store_kvcache_unified_layout, store_kvcache_distinct_layout, load_kvcache,
    CHECK_STORING, CHECK_LOADING, CHECK_ATTENTION
)
from diffulex.legacy.utils.context import ContextForDiffusionLM, get_context_causal_lm, get_context_diffusion_lm


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        model_type='causal_lm'
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.causal = model_type == 'causal_lm'
        self.model_type = model_type
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
    
    @lru_cache(maxsize=32)
    def causal_lm_block_mask(self, cum_seq_lens: torch.Tensor, B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        cache_key = (B, H, Q_LEN, KV_LEN, device)
        document_ids = torch.zeros((cum_seq_lens[-1],), dtype=torch.int32, device=device)
        start_idx = 0
        for doc_idx, seq_len in enumerate(cum_seq_lens[1:]):
            end_idx = seq_len
            document_ids[start_idx:end_idx] = doc_idx
            start_idx = end_idx
        
        def _mask_mod(batch, head, token_q, token_kv):
            causal_mask = token_q >= token_kv
            document_mask = document_ids[token_q] == document_ids[token_kv]
            return causal_mask & document_mask
        
        if cache_key not in self._block_mask_cache:
            self._block_mask_cache[cache_key] = create_block_mask(
                _mask_mod, B, H, Q_LEN, KV_LEN, device=device
            )
        return self._block_mask_cache[cache_key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: List[torch.Tensor] | None = None) -> torch.Tensor:
        # Reshape
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context: ContextForDiffusionLM = get_context_causal_lm() if self.model_type == 'causal_lm' else get_context_diffusion_lm()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = context.kv_cache_layout == "unified"

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if not (self.model_type == 'diffusion_lm' and not context.need_kv_cache_store):
                store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.model_type, context)
                # CHECK_STORING(k_cache, v_cache, k, v, context)

        transpose_fn = lambda x: rearrange(x, 's h d -> 1 h s d').contiguous()
        # Prefill / Decode logic TODO: Replace the Flex Attention Prefilling
        if context.is_prefill:
            # Block PK
            if context.block_tables is not None and self.model_type == 'causal_lm':
                k, v = k_cache, v_cache
            elif context.block_tables is not None and self.model_type == 'diffusion_lm':
                # TODO: Implement Prefix Caching
                pass

            # Attention computation
            q_t, k_t, v_t = [transpose_fn(t) for t in (q, k, v)]

            B, H, S, _ = q_t.shape
            block_mask_fn = self.causal_lm_block_mask if self.model_type == 'causal_lm' else self.dllm_block_mask
            input_obj = context.cu_seqlens_q if self.model_type == 'causal_lm' else context.block_mask
            block_mask = block_mask_fn(input_obj, B, H, S, S, str(q.device))
            o = self.attention(q_t, k_t, v_t, block_mask=block_mask)
        else:
            if self.model_type == 'causal_lm':
                o = causal_lm_flash_decoding(
                    q, k_cache, v_cache,
                    cache_seqlens=context.context_lens, block_tables=context.block_tables, 
                    softmax_scale=self.scale, page_size=256
                )
            else: 
                config = context.seqs[0].config
                diffusion_block_size = config.diffusion_block_size
                if is_unified_layout:
                    k_comb, v_comb = load_kvcache(self.k_cache, self.v_cache, context, k, v)
                    o = flash_attn_varlen_func(q, k_comb, v_comb, 
                                               context.cu_seqlens_q, context.cu_seqlens_k,
                                               context.max_seqlen_q, context.max_seqlen_k,
                                               softmax_scale=self.scale, block_table=None)
                else:
                    # FIXME: Kernel not ok...
                    o = torch.empty_like(q).to(q.device).to(q.dtype)
                    q, k, o, k_cache, v_cache = map(lambda x: x.to(torch.float32), (q, k, o, k_cache, v_cache))
                    diffusion_lm_parallel_flash_decoding(
                        q, k, v, o, str(k_cache.dtype), k_cache, v_cache, 
                        context.block_tables, context.cu_seqlens_q, context.total_lens,
                        max(context.total_lens), max(context.seq_lens), 1.0, 1.0,
                        diffusion_block_size, context.block_mask
                    )
                    CHECK_ATTENTION(o, q, k, v, k_cache, v_cache, context)
            
        # Final reshape
        if not context.is_prefill:
            o = o.view(-1, self.num_heads * self.head_dim).contiguous()
        elif context.is_prefill:
            o = rearrange(o, '1 h s d -> s (h d)').contiguous()

        return o