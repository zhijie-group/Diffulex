from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


@register_linear_strategy(weight_dtype="__stub__", act_dtype="__stub__")
def _build_linear_stub() -> LinearQuantizationStrategy:
    # Default fallback stub. Actual requested dtypes will be attached by the caller
    # via attributes after creation if needed.
    return LinearStubStrategy(weight_dtype="__stub__", act_dtype="__stub__")


@dataclass
class LinearStubStrategy(LinearQuantizationStrategy):
    """Generic stub for any non-bf16 Linear quantization combination."""

    weight_dtype: str
    act_dtype: str

    @property
    def name(self) -> str:
        return f"linear_stub(w={self.weight_dtype},a={self.act_dtype})"

    @property
    def linear_weight_format(self) -> str:
        return self.weight_dtype

    @property
    def linear_act_format(self) -> str:
        return self.act_dtype

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # Placeholder; real implementations may store packed weights in int4/int8 etc.
        return torch.uint8, 1

    def quantize(self, tensor: torch.Tensor, **kwargs):
        raise NotImplementedError(f"{self.name}: quantize is not implemented (stub). kwargs={list(kwargs.keys())}")

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        raise NotImplementedError(f"{self.name}: dequantize is not implemented (stub). kwargs={list(kwargs.keys())}")

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        _ = original_shape, kwargs
        return tuple()

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = x, weight, bias, kwargs
        raise NotImplementedError(
            "Linear quantization kernel is not implemented yet. "
            f"kind={quant_kind!r}, weight_dtype={self.weight_dtype!r}, act_dtype={self.act_dtype!r}"
        )



