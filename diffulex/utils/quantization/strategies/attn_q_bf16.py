"""
BF16 Attention-Q strategy (no quantization).
"""

import torch

from diffulex.utils.quantization.registry import register_attn_q_strategy
from diffulex.utils.quantization.strategy import AttnQQuantizationStrategy


class AttnQBF16Strategy(AttnQQuantizationStrategy):
    @property
    def name(self) -> str:
        return "attn_q_bf16"

    @property
    def attn_q_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # Q is not stored long-term; this is only to satisfy base interface.
        return torch.bfloat16, 2

    def quantize(self, tensor: torch.Tensor, **kwargs):
        return tensor, None

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata, **kwargs) -> torch.Tensor:
        return quantized

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        return (0,)


@register_attn_q_strategy("bf16", "bfloat16", "none")
def _build_attn_q_bf16() -> AttnQBF16Strategy:
    return AttnQBF16Strategy()


