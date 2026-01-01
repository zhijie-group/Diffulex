"""
FP8 Attention-Q strategy (placeholder).

This strategy is intended to be used once a matching attention kernel supports
FP8 Q inputs. For now, it is only used to exercise the dynamic dispatch path
and will lead to NotImplementedError in kernel wrappers.
"""

import torch

from diffulex.utils.quantization.registry import register_attn_q_strategy
from diffulex.utils.quantization.strategy import AttnQQuantizationStrategy


class AttnQFP8StubStrategy(AttnQQuantizationStrategy):
    @property
    def name(self) -> str:
        return "attn_q_fp8_stub"

    @property
    def attn_q_format(self) -> str:
        return "fp8"

    @property
    def requires_runtime_scales(self) -> bool:
        return True

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # Placeholder: if we store, we'd likely use uint8 or float8.
        return torch.uint8, 1

    def maybe_compute_q_scale(self, q: torch.Tensor, *, device: torch.device):
        # Placeholder: for a real kernel you'd likely compute per-head or per-tensor scale.
        # Here we just return a scalar tensor to show the plumbing works.
        return torch.ones((1,), device=device, dtype=torch.float32)

    def quantize_q_for_kernel(self, q: torch.Tensor, *, q_scale):
        # Placeholder: do NOT actually change dtype to avoid silently breaking existing kernels.
        # Real implementation should return FP8 tensor + store scales in metadata.
        return q

    # Base QuantizationStrategy methods (not used by the stub right now)
    def quantize(self, tensor: torch.Tensor, **kwargs):
        return tensor, None

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata, **kwargs) -> torch.Tensor:
        return quantized

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        return (1,)


@register_attn_q_strategy("fp8")
def _build_attn_q_fp8_stub() -> AttnQFP8StubStrategy:
    return AttnQFP8StubStrategy()






