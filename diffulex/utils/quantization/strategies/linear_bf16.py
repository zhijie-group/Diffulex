from __future__ import annotations

import torch

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


@register_linear_strategy(weight_dtype="bf16", act_dtype="bf16")
def _build_linear_bf16() -> LinearQuantizationStrategy:
    return LinearBF16Strategy()


class LinearBF16Strategy(LinearQuantizationStrategy):
    """Default Linear strategy: no quantization (bf16/bf16)."""

    @property
    def name(self) -> str:
        return "linear_bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # No special storage; keep as-is.
        return torch.bfloat16, 2

    def quantize(self, tensor: torch.Tensor, **kwargs):
        _ = kwargs
        return tensor, None

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata, **kwargs) -> torch.Tensor:
        _ = scale_or_metadata, kwargs
        return quantized

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        _ = original_shape, kwargs
        return tuple()



