"""
FP8 KV Cache quantization strategy using running max for scale management.
"""

import torch
from typing import Optional

from diffulex.utils.quantization.strategy import KVCacheQuantizationStrategy
from diffulex.utils.quantization.kv_cache_dtype import parse_kv_cache_dtype, view_fp8_cache
from diffulex.utils.quantization.registry import register_kv_cache_strategy


class KVCacheFP8RunningMaxStrategy(KVCacheQuantizationStrategy):
    """FP8 KV Cache quantization strategy using running max for scale management."""
    
    def __init__(self, dtype: str = "fp8_e4m3"):
        """
        Initialize FP8 KV Cache strategy.
        
        Args:
            dtype: FP8 dtype string ("fp8_e4m3" or "fp8_e5m2")
        """
        self.dtype_str = dtype
        self.spec = parse_kv_cache_dtype(dtype)
        if not self.spec.is_fp8:
            raise ValueError(f"Expected FP8 dtype, got {dtype}")
    
    @property
    def name(self) -> str:
        return f"kv_cache_fp8_running_max_{self.dtype_str}"

    @property
    def kv_cache_format(self) -> str:
        return "fp8"

    @property
    def requires_runtime_scales(self) -> bool:
        return True
    
    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        """Returns uint8 as storage dtype for FP8 (FP8 values are stored as uint8)."""
        return torch.uint8, 1

    def view_kv_cache_for_kernels(self, cache: torch.Tensor) -> torch.Tensor:
        # For kernels expecting float8 dtype (e.g. TileLang FP8 decode), keep uint8
        # storage but return a float8 view tensor.
        return view_fp8_cache(cache, self.dtype_str)

    def quantize_kv_for_store(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized quantization for KV cache store.

        Args:
            k/v: [N, H, D] BF16/FP16/FP32
            k_scale/v_scale: [H] float32
        Returns:
            k_q/v_q: [N, H, D] uint8
        """
        if k_scale is None or v_scale is None:
            raise ValueError("FP8 quantization requires k_scale and v_scale")
        k_q, _ = self.quantize(k, scale=k_scale)
        v_q, _ = self.quantize(v, scale=v_scale)
        return k_q, v_q
    
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor,
                      num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute initial quantization scales for K and V.
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (k_scale, v_scale): Scales with shape [num_kv_heads]
        """
        eps = 1e-8
        fp8_max = float(self.spec.fp8_max)
        
        # Compute per-head absmax: [num_kv_heads]
        k_absmax = k.to(torch.float32).abs().amax(dim=(0, 2))
        v_absmax = v.to(torch.float32).abs().amax(dim=(0, 2))
        
        # Compute scales
        k_scale = (k_absmax / fp8_max).clamp_min(eps)
        v_scale = (v_absmax / fp8_max).clamp_min(eps)
        
        return k_scale.to(device, dtype=torch.float32), v_scale.to(device, dtype=torch.float32)
    
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                     k_scale: Optional[torch.Tensor], v_scale: Optional[torch.Tensor],
                     num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update quantization scales using running max strategy.
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            k_scale: Current K scale (None if first time)
            v_scale: Current V scale (None if first time)
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (updated_k_scale, updated_v_scale): Updated scales with shape [num_kv_heads]
        """
        eps = 1e-8
        fp8_max = float(self.spec.fp8_max)
        
        # Compute current per-head absmax: [num_kv_heads]
        k_absmax = k.to(torch.float32).abs().amax(dim=(0, 2))
        v_absmax = v.to(torch.float32).abs().amax(dim=(0, 2))
        
        # Update running max
        if k_scale is None:
            k_scale = k_absmax.clone().detach()
        else:
            k_scale = torch.maximum(k_scale, k_absmax)
        
        if v_scale is None:
            v_scale = v_absmax.clone().detach()
        else:
            v_scale = torch.maximum(v_scale, v_absmax)
        
        # Compute scales from running max
        k_scale = (k_scale / fp8_max).clamp_min(eps)
        v_scale = (v_scale / fp8_max).clamp_min(eps)
        
        return k_scale.to(device, dtype=torch.float32), v_scale.to(device, dtype=torch.float32)
    
    def quantize(self, tensor: torch.Tensor, scale: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor using FP8.
        
        Args:
            tensor: Tensor to quantize [seq_len, head_dim] or [seq_len, num_heads, head_dim]
            scale: Quantization scale [1] or [num_heads]
            **kwargs: Additional arguments
        
        Returns:
            (quantized_tensor, scale): Tuple of quantized tensor (uint8) and scale
        """
        assert scale is not None, "FP8 quantization requires scale"
        assert self.spec.fp8_view_dtype is not None
        assert self.spec.fp8_min is not None and self.spec.fp8_max is not None
        
        # Handle both [seq_len, head_dim] and [seq_len, num_heads, head_dim] input shapes
        if tensor.dim() == 2:
            # [seq_len, head_dim] case: scale should be [1]
            descale = (1.0 / scale).view(-1, 1)  # [1, 1] for broadcasting to [seq_len, head_dim]
        elif tensor.dim() == 3:
            # [seq_len, num_heads, head_dim] case: scale should be [num_heads]
            descale = (1.0 / scale).view(1, -1, 1)  # [1, num_heads, 1] for broadcasting
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {tensor.dim()}D tensor with shape {tensor.shape}")
        
        # Quantize: value / scale, then clamp to FP8 range
        quantized = (tensor.float() * descale).clamp(
            min=float(self.spec.fp8_min),
            max=float(self.spec.fp8_max)
        )
        
        # Convert to FP8 view dtype, then view as uint8 for storage
        quantized_fp8 = quantized.to(self.spec.fp8_view_dtype)
        quantized_uint8 = quantized_fp8.view(torch.uint8)
        
        return quantized_uint8, scale
    
    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Dequantize a tensor from FP8.
        
        Args:
            quantized: Quantized tensor (uint8 storage) [seq_len, num_heads, head_dim]
            scale: Quantization scale [num_heads]
            **kwargs: Additional arguments
        
        Returns:
            Dequantized tensor
        """
        assert scale is not None, "FP8 dequantization requires scale"
        assert self.spec.fp8_view_dtype is not None
        
        # View uint8 as FP8 dtype
        fp8_tensor = quantized.view(self.spec.fp8_view_dtype).float()
        
        # Reshape scale to broadcast: [num_heads] -> [1, num_heads, 1]
        scale_view = scale.view(1, -1, 1)
        
        # Dequantize: value * scale
        return fp8_tensor * scale_view
    
    def get_scale_shape(self, original_shape: tuple[int, ...], num_kv_heads: int, **kwargs) -> tuple[int, ...]:
        """
        Returns the shape of scale tensor.
        
        Args:
            original_shape: Original tensor shape (not used for KV cache)
            num_kv_heads: Number of KV heads
        
        Returns:
            Scale shape: [num_kv_heads]
        """
        return (num_kv_heads,)
    
    def init_scales(self, num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize quantization scales for K and V.
        
        Args:
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (k_scale, v_scale): Initial scales with shape [num_kv_heads], initialized to 1.0
        """
        # Initialize scales to 1.0 (will be updated on first update_scales call)
        k_scale = torch.ones((num_kv_heads,), device=device, dtype=torch.float32)
        v_scale = torch.ones((num_kv_heads,), device=device, dtype=torch.float32)
        return k_scale, v_scale


@register_kv_cache_strategy("fp8", "fp8_e4m3", "e4m3")
def _build_kv_cache_fp8_e4m3() -> KVCacheFP8RunningMaxStrategy:
    return KVCacheFP8RunningMaxStrategy("fp8_e4m3")


@register_kv_cache_strategy("fp8_e5m2", "e5m2")
def _build_kv_cache_fp8_e5m2() -> KVCacheFP8RunningMaxStrategy:
    return KVCacheFP8RunningMaxStrategy("fp8_e5m2")

