import torch
import torch.nn as nn
from einops import rearrange

from diffulex_kernel import (
    store_kvcache_distinct_layout, 
    store_kvcache_unified_layout, 
    dllm_flash_attn_decode, 
    dllm_flash_attn_prefill
)
from diffulex.attention.metadata import AttnMetaDataBase


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
        # Quantization scales (will be bound by ModelRunner if strategy requires them)
        self.k_scale = None
        self.v_scale = None
        
        self.q_shape = {
            'nh': self.num_heads,
            'hd': self.head_dim,
        }
        self.kv_shape = {
            'nkvh': self.num_kv_heads,
            'hd': self.head_dim,
        }
        # Import the specified fetch function
        from diffulex.attention import fetch_attn_metadata
        self.fetch_attn_metadata = fetch_attn_metadata
        
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: list[torch.Tensor] | None = None) -> torch.Tensor:
        # Reshape
        q = rearrange(q, 's (nh hd) -> s nh hd', **self.q_shape)
        k = rearrange(k, 's (nkvh hd) -> s nkvh hd', **self.kv_shape)
        v = rearrange(v, 's (nkvh hd) -> s nkvh hd', **self.kv_shape)

        attn_metadata: AttnMetaDataBase = self.fetch_attn_metadata()
        k_cache, v_cache = self.k_cache, self.v_cache
        is_unified_layout = attn_metadata.kv_cache_layout == "unified"

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            if attn_metadata.need_kv_cache_store:
                # Update scales if quantization strategy requires them
                if self.k_scale is not None and self.v_scale is not None:
                    from diffulex.utils.quantization.context import get_kv_cache_strategy
                    strategy = get_kv_cache_strategy()
                    if strategy is not None:
                        self.k_scale, self.v_scale = strategy.update_scales(
                            k, v, self.k_scale, self.v_scale,
                            self.num_kv_heads, k.device
                        )
                    # Pass scale to metadata if required by strategy
                    if strategy is not None:
                        strategy.maybe_set_attn_metadata_scales(
                            attn_metadata, k_scale=self.k_scale, v_scale=self.v_scale
                        )
                
                store_kvcache = store_kvcache_unified_layout if is_unified_layout else store_kvcache_distinct_layout
                store_kvcache(k, v, k_cache, v_cache, attn_metadata.slot_mapping, attn_metadata)

        # Prefill / Decode logic
        if attn_metadata.is_prefill:
            if attn_metadata.block_tables is not None:
                # TODO: Implement Prefix Caching
                pass
            o = dllm_flash_attn_prefill(q, k, v, self.scale, attn_metadata)
        else:
            if is_unified_layout:
                from diffulex.utils.quantization.context import get_kv_cache_strategy
                strategy = get_kv_cache_strategy()
                if strategy is not None:
                    # e.g. FP8: pass scales to metadata for kernel / load_kvcache to handle
                    strategy.maybe_set_attn_metadata_scales(
                        attn_metadata, k_scale=self.k_scale, v_scale=self.v_scale
                    )
                
                o = dllm_flash_attn_decode(q, k, v, k_cache, v_cache, self.scale, attn_metadata)
            else:
                # Distinct layout: use varlen mode with load_kvcache
                from diffulex_kernel import load_kvcache
                from diffulex.utils.quantization.context import get_kv_cache_strategy
                strategy = get_kv_cache_strategy()
                if strategy is not None:
                    # e.g. FP8: pass scales to metadata for load_kvcache to handle
                    strategy.maybe_set_attn_metadata_scales(
                        attn_metadata, k_scale=self.k_scale, v_scale=self.v_scale
                    )
                
                # Distinct layout uses varlen mode
                k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
                from flash_attn import flash_attn_varlen_func
                o = flash_attn_varlen_func(
                    q, k_comb, v_comb,
                    attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
                    attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
                    softmax_scale=self.scale, block_table=None
                )
            
        # Final reshape
        return rearrange(o, 's nh hd -> s (nh hd)').contiguous()