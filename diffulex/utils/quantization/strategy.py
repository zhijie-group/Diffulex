"""
Quantization strategy interfaces.

This module defines abstract base classes for different types of quantization strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F


class _AttnMetaDataLike(Protocol):
    """A minimal protocol for attention metadata used by Diffulex runtime.

    We avoid importing `diffulex.attention.metadata` here to reduce the chance
    of circular imports.
    """

    k_scale: Optional[torch.Tensor]
    v_scale: Optional[torch.Tensor]


class QuantizationStrategy(ABC):
    """Quantization strategy abstract base class."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass
    
    @abstractmethod
    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        """
        Returns storage dtype and itemsize.
        
        Returns:
            (storage_dtype, itemsize): Tuple of storage dtype and item size in bytes.
        """
        pass
    
    @abstractmethod
    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """
        Quantize a tensor.
        
        Args:
            tensor: Input tensor to quantize.
            **kwargs: Additional arguments for quantization.
        
        Returns:
            (quantized_tensor, scale_or_metadata): Tuple of quantized tensor and scale/metadata.
        """
        pass
    
    @abstractmethod
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """
        Dequantize a tensor.
        
        Args:
            quantized: Quantized tensor to dequantize.
            scale_or_metadata: Scale or metadata needed for dequantization.
            **kwargs: Additional arguments for dequantization.
        
        Returns:
            Dequantized tensor.
        """
        pass
    
    @abstractmethod
    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """
        Returns the shape of scale tensor.
        
        Args:
            original_shape: Original tensor shape.
            **kwargs: Additional arguments (e.g., num_kv_heads for KV cache).
        
        Returns:
            Scale tensor shape.
        """
        pass

    def configure(self, *, diffulex_config: Any | None = None) -> None:
        """Optional hook to configure a strategy from Diffulex `Config`.

        We intentionally keep this a no-op by default to avoid forcing configuration
        plumbing through every call site. Strategy-specific tuning knobs should be
        surfaced via explicit fields on `diffulex.config.Config`, not environment variables.
        """
        _ = diffulex_config
        return

    # ---- Optional capability flags / helpers (non-abstract) ----
    # These helpers are used to avoid hard-coding isinstance(...) checks in the runtime.
    @property
    def requires_runtime_scales(self) -> bool:
        """Whether this strategy requires runtime scale tensors to be allocated/updated."""
        return False


class KVCacheQuantizationStrategy(QuantizationStrategy):
    """KV Cache quantization strategy interface (extended interface)."""

    # NOTE: We use a small string tag for dispatch instead of importing enums everywhere.
    # Known values:
    # - "bf16": no quantization, cache stored as bf16
    # - "fp8": FP8 cache stored as uint8 (storage) with float8 view for kernels
    @property
    def kv_cache_format(self) -> str:
        return "bf16"
    
    @abstractmethod
    def compute_scales(self, k: torch.Tensor, v: torch.Tensor, 
                      num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantization scales for K and V.
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (k_scale, v_scale): Tuple of K and V scales, shape [num_kv_heads]
        """
        pass
    
    @abstractmethod
    def update_scales(self, k: torch.Tensor, v: torch.Tensor,
                     k_scale: Optional[torch.Tensor], v_scale: Optional[torch.Tensor],
                     num_kv_heads: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update quantization scales (e.g., using running max strategy).
        
        Args:
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            v: Value tensor [seq_len, num_kv_heads, head_dim]
            k_scale: Current K scale (None if first time)
            v_scale: Current V scale (None if first time)
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (updated_k_scale, updated_v_scale): Updated scales, shape [num_kv_heads]
        """
        pass
    
    def init_scales(self, num_kv_heads: int, device: torch.device) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Initialize quantization scales for K and V.
        
        This method should be called once per layer to initialize scale tensors.
        Strategies that don't require scales (e.g., BF16) should return (None, None).
        
        Args:
            num_kv_heads: Number of KV heads
            device: Target device
        
        Returns:
            (k_scale, v_scale): Initial scales, shape [num_kv_heads], or (None, None) if not needed
        """
        # Default implementation: return None (no scales needed)
        return None, None

    # ---- Diffulex integration helpers (non-abstract) ----
    @property
    def requires_kv_cache_scales(self) -> bool:
        """Whether KV cache kernels / decode require per-head scales."""
        return self.requires_runtime_scales

    def maybe_set_attn_metadata_scales(
        self,
        attn_metadata: _AttnMetaDataLike,
        *,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> None:
        """Populate `attn_metadata.k_scale/v_scale` when needed."""
        if not self.requires_kv_cache_scales:
            return
        if k_scale is None or v_scale is None:
            raise ValueError(
                f"{self.name} requires k_scale/v_scale but got "
                f"k_scale={k_scale is not None}, v_scale={v_scale is not None}"
            )
        attn_metadata.k_scale = k_scale
        attn_metadata.v_scale = v_scale

    def view_kv_cache_for_kernels(self, cache: torch.Tensor) -> torch.Tensor:
        """Return a view of cache suitable for kernel consumption.

        - BF16 strategies: return as-is
        - FP8 strategies: subclasses may return a float8 view while keeping uint8 storage
        """
        return cache

    def quantize_kv_for_store(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V for KV cache store (optional helper).

        Returns:
            (k_quantized, v_quantized): Usually uint8 tensors for FP8 strategies.
        """
        raise NotImplementedError(f"{self.name} does not implement quantize_kv_for_store")


class WeightQuantizationStrategy(QuantizationStrategy):
    """Weight quantization strategy interface (for future extension)."""
    
    @abstractmethod
    def quantize_weight(self, weight: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """
        Quantize model weights.
        
        Args:
            weight: Weight tensor to quantize.
            **kwargs: Additional arguments for quantization.
        
        Returns:
            (quantized_weight, scale_or_metadata): Tuple of quantized weight and scale/metadata.
        """
        pass
    
    @abstractmethod
    def dequantize_weight(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """
        Dequantize model weights.
        
        Args:
            quantized: Quantized weight tensor.
            scale_or_metadata: Scale or metadata needed for dequantization.
            **kwargs: Additional arguments for dequantization.
        
        Returns:
            Dequantized weight tensor.
        """
        pass


class LinearQuantizationStrategy(QuantizationStrategy):
    """Linear layer quantization strategy interface (weights + activations).

    This is an architecture hook: kernels/packed weights can be implemented later.
    The runtime (Linear layers) should dispatch by `quant_kind` ("attn"/"mlp"/"other")
    and use this strategy to compute the Linear output.
    """

    @property
    def linear_weight_format(self) -> str:
        """Small tag used for kernel dispatch for weights.

        Known values (initial set):
        - "bf16": no weight quantization
        - "int8"/"int4"/"fp8_e4m3"/"fp8_e5m2"/"gptq"/"awq": placeholders
        """
        return "bf16"

    @property
    def linear_act_format(self) -> str:
        """Small tag used for kernel dispatch for activations."""
        return "bf16"

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Optionally quantize/pack weight for kernel consumption.

        Default behavior: no-op, returns (weight, None).
        """
        if device is not None:
            weight = weight.to(device=device)
        return weight, None

    def quantize_act_for_kernel(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Optionally quantize activations for kernel consumption.

        Default behavior: no-op, returns (x, None).
        """
        if device is not None:
            x = x.to(device=device)
        return x, None

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Linear output for a given kind.

        Default behavior: `F.linear(x, weight, bias)` (no quantization).
        Quantized strategies may override this to call custom kernels.
        """
        _ = quant_kind, kwargs
        return F.linear(x, weight, bias)

