"""
W8A8 Linear quantization strategy (int8 weight + int8 activation), TileLang-free.

Implementation (vLLM-aligned):
- Activation quantization: `vllm._custom_ops.scaled_int8_quant` (dynamic per-token).
- GEMM+dequant: `vllm._custom_ops.cutlass_scaled_mm` (CUTLASS, with internal
  triton fallback depending on shape/platform) — no `F.linear` slow path.

Notes:
- Weight is stored as int8 in **K×N** layout (transposed), matching vLLM CUTLASS
  kernels.
- Weight scale is stored as **[1, N]** float32 for broadcasting.
"""

from __future__ import annotations

from typing import Any, Optional

import torch  # type: ignore

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


try:
    from vllm import _custom_ops as _vllm_ops  # type: ignore
except Exception:  # pragma: no cover
    _vllm_ops = None  # type: ignore


@register_linear_strategy(weight_dtype="int8", act_dtype="int8")
def _build_linear_int8_w8a8() -> LinearQuantizationStrategy:
    return LinearInt8W8A8Strategy()


class LinearInt8W8A8Strategy(LinearQuantizationStrategy):
    def __init__(self) -> None:
        super().__init__()
        # Cache: id(weight) -> (qweight_int8 [N,K], w_scales_fp32 [N])
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def name(self) -> str:
        return "linear_int8_w8a8"

    @property
    def linear_weight_format(self) -> str:
        return "int8"

    @property
    def linear_act_format(self) -> str:
        return "int8"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        return torch.int8, 1

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs: Any) -> tuple[int, ...]:
        _ = kwargs
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight [N,K], got {original_shape}")
        return (original_shape[0],)

    def quantize(self, tensor: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(tensor.shape)}")
        # per-output-channel symmetric int8, store K×N for cutlass_scaled_mm
        w = tensor.to(torch.float32)
        abs_max = w.abs().amax(dim=-1, keepdim=False)  # [N]
        scales = (abs_max.clamp(min=1e-8) / 127.0).to(torch.float32)  # [N]
        q_nk = torch.round(w / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)  # [N,K]
        # NOTE: vLLM CUTLASS scaled_mm expects b.stride(0) == 1, which is true
        # for a transpose-view (non-contiguous) but not for a contiguous K×N tensor.
        q_kn = q_nk.t()  # [K,N], stride(0)==1
        scale_b = scales.unsqueeze(0).contiguous()  # [1,N]
        return q_kn, {"scales": scale_b}

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        # Return int8 K×N weights + fp32 [1,N] scales for vLLM CUTLASS path.
        q_kn, meta = self.quantize(weight)
        if device is not None:
            q_kn = q_kn.to(device=device)
            meta["scales"] = meta["scales"].to(device=device)
        return q_kn, meta["scales"]

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        _ = kwargs
        scales = scale_or_metadata.get("scales") if isinstance(scale_or_metadata, dict) else scale_or_metadata
        if scales is None:
            raise ValueError("scales required for dequantization")
        raise RuntimeError(
            "W8A8 不提供 dequantize 路径（避免走慢的 bf16 GEMM）。"
        )

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        quant_scales: Optional[torch.Tensor] = None,
        out_features: Optional[int] = None,
    ) -> torch.Tensor:
        _ = quant_kind
        if _vllm_ops is None:
            raise RuntimeError("vLLM custom ops are required for W8A8 (scaled_int8_quant / cutlass_scaled_mm).")

        # Weight/scales: prefer load-time quantized buffers.
        if weight is not None and weight.dtype == torch.int8 and quant_scales is not None:
            qweight = weight
            w_scales = quant_scales
        else:
            wid = id(weight)
            cached = self._weight_cache.get(wid)
            if cached is None or cached[0].device != x.device:
                qweight, meta = self.quantize(weight)
                qweight = qweight.to(device=x.device)
                w_scales = meta["scales"].to(device=x.device, dtype=torch.float32)
                self._weight_cache[wid] = (qweight, w_scales)
            else:
                qweight, w_scales = cached

        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
        if x2.dtype not in (torch.bfloat16, torch.float16):
            x2 = x2.to(torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()

        # dynamic per-token int8 quant + fused GEMM+dequant
        x_q, x_s, _ = _vllm_ops.scaled_int8_quant(x2, scale=None, azp=None, symmetric=True)
        y = _vllm_ops.cutlass_scaled_mm(
            x_q,
            qweight,
            scale_a=x_s,
            scale_b=w_scales,
            out_dtype=x2.dtype,
            bias=bias.to(dtype=x2.dtype) if bias is not None else None,
        )

        if orig_shape == x2.shape:
            return y
        if x.dim() == 1:
            return y.squeeze(0)
        return y.reshape(*orig_shape[:-1], y.shape[-1])

