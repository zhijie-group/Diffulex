"""
FP8 W8A16 Linear quantization strategy (FP8 weight + bf16 activation), TileLang-free.

vLLM-aligned implementation:
- Weight quantization: `vllm._custom_ops.scaled_fp8_quant` (FP8 weight + per-tensor scale).
- Forward: use vLLM's `Fp8LinearOp` (CUTLASS scaled_mm when available).

Note:
- vLLM 的 FP8 linear 核心路径以 e4m3 为主（由 vLLM 当前平台决定的 fp8 dtype）。
- 为了避免“静默走慢路径”，这里不再使用 `F.linear` 的反量化 GEMM。
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy


@register_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
def _build_linear_fp8_e4m3_w8a16() -> LinearQuantizationStrategy:
    return LinearFP8W8A16Strategy("fp8_e4m3")


@register_linear_strategy(weight_dtype="fp8_e5m2", act_dtype="bf16")
def _build_linear_fp8_e5m2_w8a16() -> LinearQuantizationStrategy:
    return LinearFP8W8A16Strategy("fp8_e5m2")


class LinearFP8W8A16Strategy(LinearQuantizationStrategy):
    def __init__(self, weight_dtype: str = "fp8_e4m3") -> None:
        super().__init__()
        self.weight_dtype_str = weight_dtype
        # Cache: id(weight) -> (q_fp8_KN [K,N], scale_fp32 [1])
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        try:
            from vllm.model_executor.layers.quantization.utils.w8a8_utils import (  # type: ignore
                Fp8LinearOp,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError("FP8 需要 vLLM（Fp8LinearOp / _custom_ops）。") from e

        # dynamic activation quantization to FP8 inside vLLM
        self._fp8_linear = Fp8LinearOp(act_quant_static=False)

    @property
    def name(self) -> str:
        return f"linear_fp8_{self.weight_dtype_str}_w8a16"

    @property
    def linear_weight_format(self) -> str:
        return self.weight_dtype_str

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # vLLM stores fp8 weights as float8 dtype tensor
        return torch.uint8, 1

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs: Any) -> tuple[int, ...]:
        _ = kwargs
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight [N,K], got {original_shape}")
        # per-tensor scale
        return (1,)

    def quantize(self, tensor: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(tensor.shape)}")
        from vllm import _custom_ops as ops  # type: ignore
        from vllm.platforms import current_platform  # type: ignore

        # vLLM: per-tensor scale, output dtype = current_platform.fp8_dtype()
        q_fp8, scale = ops.scaled_fp8_quant(tensor.to(torch.float32).contiguous(), scale=None)
        # Keep transpose-view for CUTLASS expectation (b.stride(0) == 1).
        q_kn_fp8 = q_fp8.t()  # [K,N] fp8 dtype, non-contiguous
        scale = scale.to(torch.float32).reshape(1).contiguous()
        return q_kn_fp8, {"scales": scale, "fp8_dtype": current_platform.fp8_dtype()}

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, Any]:
        q_fp8, meta = self.quantize(weight)
        if device is not None:
            q_fp8 = q_fp8.to(device=device)
            meta["scales"] = meta["scales"].to(device=device)
        return q_fp8, meta["scales"]

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        _ = kwargs
        raise RuntimeError("FP8 不提供 dequantize 路径（避免走慢的反量化 + F.linear）。")

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
        _ = quant_kind, out_features
        if weight is not None and quant_scales is not None:
            # Expected: weight is fp8 K×N tensor (transpose-view is fine).
            q_kn = weight.to(device=x.device)
            scales = quant_scales.to(device=x.device, dtype=torch.float32).reshape(1)
        else:
            wid = id(weight)
            cached = self._weight_cache.get(wid)
            if cached is None or cached[0].device != x.device:
                q_fp8, meta = self.quantize(weight)
                q_fp8 = q_fp8.to(device=x.device)
                scales = meta["scales"].to(device=x.device, dtype=torch.float32).reshape(1)
                q_kn = q_fp8
                self._weight_cache[wid] = (q_fp8, scales)
            else:
                q_kn, scales = cached

        # vLLM Fp8LinearOp expects weight as [K,N] fp8 tensor and per-tensor scale.
        return self._fp8_linear.apply(
            input=x,
            weight=q_kn,
            weight_scale=scales,
            out_dtype=x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16,
            input_scale=None,
            bias=bias,
        )

