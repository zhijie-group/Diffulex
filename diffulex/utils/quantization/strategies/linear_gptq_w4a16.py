"""
GPTQ W4A16 Linear quantization strategy (vLLM standard format).

- Weight format: vLLM GPTQ (packed int32 qweight/qzeros + fp16 scales)
- Activation: bf16 (no activation quantization)
- Forward: vLLM custom op `gptq_gemm`

Design notes:
- Diffulex follows vLLM's fast path: run `gptq_shuffle` once (handled by
  `LinearBase._maybe_prepare_offline_gptq`) and then call `gptq_gemm` with
  `use_exllama=True`.
- No TileLang dependency.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

try:
    from vllm import _custom_ops as ops  # type: ignore
except Exception:  # pragma: no cover
    ops = None  # type: ignore


@register_linear_strategy(weight_dtype="gptq", act_dtype="bf16")
def _build_linear_gptq_w4a16() -> LinearQuantizationStrategy:
    return LinearGPTQW4A16Strategy()


class LinearGPTQW4A16Strategy(LinearQuantizationStrategy):
    def __init__(self) -> None:
        super().__init__()
        self._ops_available: bool = bool(ops is not None and hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "gptq_gemm"))

    @property
    def name(self) -> str:
        return "linear_gptq_w4a16"

    @property
    def linear_weight_format(self) -> str:
        return "gptq"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # vLLM GPTQ stores packed weights in int32.
        return torch.int32, 4

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        # vLLM GPTQ scales: [K/group, N], where Linear weight is (N, K).
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight shape, got {original_shape}")
        out_features, in_features = original_shape
        group_size = int(kwargs.get("group_size", 128))
        group_size = in_features if group_size == -1 else group_size
        if group_size <= 0 or in_features % group_size != 0:
            raise ValueError(f"Invalid group_size={group_size} for in_features={in_features}")
        num_groups = in_features // group_size
        return (num_groups, out_features)

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        # Offline GPTQ is handled by `diffulex.utils.quantization.quantize_model`.
        return tensor, {}

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        if quantized.is_floating_point():
            return quantized
        raise NotImplementedError(
            "GPTQ dequantize is not implemented in Diffulex. "
            "Use vLLM kernels via linear_forward."
        )

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        gptq_qweight: Optional[torch.Tensor] = None,
        gptq_qzeros: Optional[torch.Tensor] = None,
        gptq_scales: Optional[torch.Tensor] = None,
        gptq_g_idx: Optional[torch.Tensor] = None,
        weight_bits: int = 0,
        use_v2_format: bool = False,
        out_features: Optional[int] = None,
        in_features: Optional[int] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        _ = quant_kind, weight, in_features, group_size
        if not self._ops_available:
            raise RuntimeError(
                "vLLM is required for GPTQ W4A16 (missing `vllm._custom_ops`). "
                "Please install/build vLLM with CUDA ops."
            )
        qweight = gptq_qweight
        qzeros = gptq_qzeros
        scales = gptq_scales
        g_idx = gptq_g_idx

        if qweight is None or qzeros is None or scales is None:
            # correctness fallback (should not happen for offline GPTQ weights)
            if weight is None:
                raise RuntimeError("GPTQ offline weights missing packed tensors and bf16 weight is not present.")
            return F.linear(x, weight, bias)

        if weight_bits <= 0:
            raise RuntimeError("GPTQ requires explicit weight_bits (>0) for the CUDA kernel path.")

        # vLLM GPTQ kernels expect FP16 activations.
        x_in = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)
        x2 = x_in.reshape(-1, x_in.shape[-1]) if x_in.dim() != 2 else x_in
        if not x2.is_contiguous():
            x2 = x2.contiguous()

        if g_idx is None or g_idx.numel() == 0:
            g_idx_t = torch.empty((0,), device=x.device, dtype=torch.int)
        else:
            g_idx_t = g_idx if (g_idx.device == x.device and g_idx.dtype == torch.int) else g_idx.to(device=x.device, dtype=torch.int)

        output = torch.ops._C.gptq_gemm(
            x2,
            qweight,
            qzeros,
            scales,
            g_idx_t,
            True,  # use_exllama
            bool(use_v2_format),
            int(weight_bits),
        )
        if bias is not None:
            output.add_(bias.to(dtype=output.dtype))

        out_shape = x.shape[:-1] + (int(out_features) if out_features is not None else int(qweight.shape[-1]),)
        output = output.reshape(out_shape)
        return output.to(dtype=x.dtype) if output.dtype != x.dtype else output

