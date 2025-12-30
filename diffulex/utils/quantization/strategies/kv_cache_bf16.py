"""
BF16 KV Cache quantization strategy (no actual quantization, just storage format).
"""

import torch
from typing import Optional
from diffulex.utils.quantization.strategy import KVCacheQuantizationStrategy
from diffulex.utils.quantization.registry import register_kv_cache_strategy


class KVCacheBF16Strategy(KVCacheQuantizationStrategy):
    """BF16 KV Cache strategy (no quantization, just storage format)."""
    
    @property
    def name(self) -> str:
        return "kv_cache_bf16"
    
    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        """Returns BF16 storage dtype."""
        return torch.bfloat16, 2
    
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor,
                      num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """No scales needed for BF16 (no quantization)."""
        # Return dummy scales (will not be used)
        k_scale = torch.ones((num_kv_heads,), device=device, dtype=torch.float32)
        v_scale = torch.ones((num_kv_heads,), device=device, dtype=torch.float32)
        return k_scale, v_scale
    
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                     k_scale: Optional[torch.Tensor], v_scale: Optional[torch.Tensor],
                     num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """No scales needed for BF16 (no quantization)."""
        if k_scale is None:
            k_scale = torch.ones((num_kv_heads,), device=device, dtype=torch.float32)
        if v_scale is None:
            v_scale = torch.ones((num_kv_heads,), device=device, dtype=torch.float32)
        return k_scale, v_scale
    
    def quantize(self, tensor: torch.Tensor, scale: Optional[torch.Tensor] = None, **kwargs) -> tuple[torch.Tensor, None]:
        """No quantization, just convert to BF16 if needed."""
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        return tensor, None
    
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: None, **kwargs) -> torch.Tensor:
        """No dequantization needed."""
        return quantized
    
    def get_scale_shape(self, original_shape: tuple[int, ...], num_kv_heads: int, **kwargs) -> tuple[int, ...]:
        """No scale needed for BF16."""
        return (0,)  # Empty shape


# NOTE: fp16/fp32 are currently routed to the BF16 kernels in Diffulex.
# Keeping them registered avoids breaking older configs while we add
# true fp16/fp32 KV-cache kernels in the future.
@register_kv_cache_strategy("bf16", "bfloat16", "fp16", "float16", "fp32", "float32")
def _build_kv_cache_bf16() -> KVCacheBF16Strategy:
    return KVCacheBF16Strategy()

