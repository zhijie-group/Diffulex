"""
Marlin-style (vLLM AllSpark) W8A16 Linear quantization strategy.

Goal:
- Replace Diffulex current W8A16 path (TileLang kernel that casts int8->bf16 inside)
  with a vLLM-like fused path for decode small-M:
  - per-out-channel int8 quantization (stored as uint8 with +128 bias)
  - one-time N32K16 reorder (AllSpark repack)
  - fused dequant + GEMM kernel (AllSpark w8a16 gemm)

Notes:
- Despite the filename mentioning "marlin", the actual fused kernel we vendor is
  vLLM's AllSpark Ampere W8A16 fused GEMM, which is the effective INT8 W8A16
  fast path in vLLM for this use-case.
- Fallback behavior is critical: if the extension is unavailable, or shapes are
  unsupported (e.g., K%16!=0), we fall back to existing TileLang W8A16 or BF16.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

# Optional: existing TileLang fallback (already used by linear_int8_w8a16.py)
try:
    from diffulex_kernel.python.linear_kernels import w8a16_gemm as _tilelang_w8a16_gemm
    _TILELANG_AVAILABLE = True
except Exception:
    _tilelang_w8a16_gemm = None
    _TILELANG_AVAILABLE = False

# Vendored vLLM-style fused W8A16 (AllSpark) ops.
try:
    from diffulex_kernel.python.marlin_ops import (  # noqa: F401
        allspark_w8a16_gemm as _allspark_w8a16_gemm,
        rearrange_kn_weight_as_n32k16_order as _allspark_repack,
        is_available as _allspark_is_available,
    )
except Exception:
    _allspark_w8a16_gemm = None
    _allspark_repack = None

    def _allspark_is_available() -> bool:
        return False


@register_linear_strategy(weight_dtype="marlin_int8", act_dtype="bf16")
def _build_linear_marlin_int8_w8a16() -> LinearQuantizationStrategy:
    return LinearMarlinInt8W8A16Strategy()


class LinearMarlinInt8W8A16Strategy(LinearQuantizationStrategy):
    """W8A16 strategy using vendored vLLM AllSpark fused GEMM + repack."""

    def __init__(self) -> None:
        super().__init__()
        # Cache for bf16 Parameters only (load-time quantized path bypasses this).
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def name(self) -> str:
        return "linear_marlin_int8_w8a16"

    @property
    def linear_weight_format(self) -> str:
        # Important: keep "int8" so LinearBase load-time quantization path triggers
        # and drops bf16 weights to save memory.
        return "int8"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # We store qweight as uint8 (bias128 representation).
        return torch.uint8, 1

    # ---- Required abstract methods (for registry/factory instantiation) ----
    def quantize(self, tensor: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        """Reference per-output-channel symmetric int8 quantization.

        Returns:
          quantized_int8: [N,K] int8
          scales: [N] bf16
        """
        _ = kwargs
        if tensor.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(tensor.shape)}")
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(dtype=torch.bfloat16)
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [N,1]
        scales = (abs_max.clamp(min=1e-8) / 127.0).to(dtype=torch.bfloat16)  # [N,1]
        q = torch.round(tensor.to(torch.float32) / scales.to(torch.float32)).clamp(-128, 127).to(torch.int8)
        return q, scales.squeeze(-1)

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs: Any) -> torch.Tensor:
        """Reference dequantization back to bf16."""
        _ = kwargs
        scales = scale_or_metadata.get("scales") if isinstance(scale_or_metadata, dict) else scale_or_metadata
        if scales is None:
            raise ValueError("scales required for dequantization")
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)
        return (quantized.to(torch.float32) * scales.to(torch.float32)).to(torch.bfloat16)

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs: Any) -> tuple[int, ...]:
        _ = kwargs
        if len(original_shape) < 2:
            raise ValueError(f"Expected weight shape with at least 2 dims, got {original_shape}")
        return (original_shape[0],)

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Quantize+repack bf16 weight for AllSpark fused kernel.

        Input:
          weight: [N, K] bf16/fp16
        Output:
          qweight_reorder: [N_32align, K] uint8 in N32K16 reorder layout
          scales_reorder:  [N_32align] bf16 scales (reordered/padded)
        """
        _ = kwargs
        if device is not None:
            weight = weight.to(device=device)

        if weight.dim() != 2:
            raise ValueError(f"Expected 2D weight [N,K], got shape={tuple(weight.shape)}")

        # Ensure bf16 for stable scales.
        if weight.dtype != torch.bfloat16:
            weight = weight.to(dtype=torch.bfloat16)

        n, k = weight.shape
        n_32 = ((n + 31) // 32) * 32

        # Per-output-channel symmetric scale.
        abs_max = torch.abs(weight).max(dim=-1)[0]  # [N]
        scales = (abs_max.clamp(min=1e-8) / 127.0).to(dtype=torch.bfloat16)  # [N]

        # Quantize to signed int8, then store as uint8 with +128 bias.
        w_fp32 = weight.to(torch.float32)
        s_fp32 = scales.to(torch.float32).unsqueeze(-1)  # [N,1]
        q_i8 = torch.round(w_fp32 / s_fp32).clamp(-128, 127).to(torch.int16)  # [N,K]
        q_u8 = (q_i8 + 128).to(torch.uint8)  # [N,K] in [0,255]

        if not _allspark_is_available() or _allspark_repack is None:
            # Fallback storage (no reorder). Keep [N,K] and [N].
            # Note: forward will detect unavailable allspark and fallback further.
            if n_32 != n:
                q_pad = torch.full((n_32, k), 128, device=q_u8.device, dtype=torch.uint8)
                q_pad[:n, :] = q_u8
                s_pad = torch.zeros((n_32,), device=scales.device, dtype=torch.bfloat16)
                s_pad[:n] = scales
                return q_pad.contiguous(), s_pad.contiguous()
            return q_u8.contiguous(), scales.contiguous()

        # AllSpark repack expects B in (K,N) contiguous layout.
        b_kn = q_u8.transpose(0, 1).contiguous()  # [K,N]

        q_reorder = torch.empty((n_32, k), device=b_kn.device, dtype=torch.uint8)
        s_reorder = torch.empty((n_32,), device=scales.device, dtype=torch.bfloat16)

        # No zero-point path for symmetric signed int8 (bias128 already handled).
        _allspark_repack(
            b_kn,
            scales.contiguous(),
            None,
            False,  # has_zp
            q_reorder,
            s_reorder,
            None,
            int(k),
            int(n),
            int(n_32),
        )

        return q_reorder.contiguous(), s_reorder.contiguous()

    def quantize_act_for_kernel(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        if device is not None:
            x = x.to(device=device)
        # No activation quantization for W8A16.
        return x, None

    def _get_sm_info(self, device: torch.device) -> tuple[int, int]:
        try:
            props = torch.cuda.get_device_properties(device)
            sm_count = int(getattr(props, "multi_processor_count", 0))
            sm_version = int(props.major) * 10 + int(props.minor)
            return sm_count, sm_version
        except Exception:
            return 0, 0

    def _cublas_m_threshold(self) -> int:
        # For decode, M is typically small, so AllSpark custom kernel is preferred.
        # For large-M prefill, AllSpark falls back to a dequant+cuBLAS path if M > threshold.
        try:
            return int(os.getenv("DIFFULEX_ALLSPARK_CUBLAS_M_THRESHOLD", "256"))
        except Exception:
            return 256

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        _ = quant_kind

        # Handle >2D like torch.nn.functional.linear: flatten then reshape back.
        orig_shape = x.shape
        if x.dim() == 1:
            x2 = x.unsqueeze(0)
        elif x.dim() == 2:
            x2 = x
        else:
            x2 = x.reshape(-1, x.shape[-1])

        # Load-time quantized module path: weight is uint8/int8 buffer and scales provided.
        quant_scales = kwargs.pop("quant_scales", None)
        if weight is not None and weight.dtype in (torch.uint8, torch.int8):
            if quant_scales is None:
                raise ValueError("quant_scales is required when weight is quantized")
            qweight = weight
            scales = quant_scales
        else:
            # Lazy cache for bf16 weights (not expected in steady-state, but keep for safety).
            weight_id = id(weight)
            cached = self._weight_cache.get(weight_id)
            if cached is None or cached[0].device != x2.device:
                qweight, scales = self.quantize_weight_for_kernel(weight, device=x2.device)
                self._weight_cache[weight_id] = (qweight, scales)
            else:
                qweight, scales = cached

        # If fused kernel isn't available, fall back to TileLang or BF16.
        if _allspark_w8a16_gemm is None or not _allspark_is_available():
            return self._fallback(x, weight, qweight, scales, bias)

        # AllSpark kernel requires CUDA and contiguous inputs.
        if x2.device.type != "cuda":
            return self._fallback(x, weight, qweight, scales, bias)

        if x2.dtype != torch.bfloat16:
            x2 = x2.to(dtype=torch.bfloat16)

        # Shape checks: x2 [M,K], qweight [N_32align,K]
        m, k = x2.shape
        n_32, k_w = qweight.shape
        if k_w != k:
            return self._fallback(x, weight, qweight, scales, bias)
        if k % 16 != 0:
            return self._fallback(x, weight, qweight, scales, bias)

        # Recover real N from module bias/metadata if available; default to n_32.
        # In Diffulex, LinearBase stores output_size; but strategy doesn't receive module.
        # So we infer N from bias if present else from scales length (can be N_32align).
        n = int(bias.numel()) if bias is not None else int(min(scales.numel(), n_32))
        if n <= 0 or n > n_32:
            n = n_32

        sm_count, sm_version = self._get_sm_info(x2.device)
        cublas_thr = self._cublas_m_threshold()

        y2 = _allspark_w8a16_gemm(
            x2.contiguous(),
            qweight.contiguous(),
            scales.contiguous(),
            None,  # b_qzeros
            n,
            -1,  # group_size (only supports -1)
            sm_count,
            sm_version,
            cublas_thr,
            False,  # has_zp
            True,  # n32k16_reorder
        )
        if bias is not None:
            y2 = y2 + bias

        # Reshape back
        if x.dim() == 1:
            y = y2.squeeze(0)
        elif x.dim() == 2:
            y = y2
        else:
            y = y2.reshape(*orig_shape[:-1], y2.shape[-1])
        return y

    def _fallback(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Prefer existing TileLang W8A16 if available and inputs are CUDA.
        if _TILELANG_AVAILABLE and _tilelang_w8a16_gemm is not None and x.device.type == "cuda":
            try:
                x2 = x if x.dim() == 2 else x.reshape(-1, x.shape[-1])
                # TileLang expects int8 weight. If our qweight is uint8 bias128, convert to int8 on the fly.
                if qweight.dtype == torch.uint8:
                    q_i8 = (qweight.to(torch.int16) - 128).to(torch.int8)
                else:
                    q_i8 = qweight
                y2 = _tilelang_w8a16_gemm(x2, q_i8, scales, False)
                if bias is not None:
                    y2 = y2 + bias
                if x.dim() == 2:
                    return y2
                if x.dim() == 1:
                    return y2.squeeze(0)
                return y2.reshape(*x.shape[:-1], y2.shape[-1])
            except Exception:
                pass

        # Last resort: BF16 F.linear using dequantized weight if bf16 is available.
        if weight is not None and getattr(weight, "dtype", None) in (torch.float16, torch.bfloat16):
            return F.linear(x, weight, bias)

        # Dequantize from qweight + scales and use cuBLAS via F.linear.
        # qweight may be [N_32,K] or reordered; we cannot reliably undo reorder here.
        # So only attempt this if qweight looks like plain [N,K] (no padding).
        if qweight.dim() == 2 and scales.dim() == 1 and qweight.shape[0] == scales.shape[0]:
            if qweight.dtype == torch.uint8:
                q = (qweight.to(torch.int16) - 128).to(torch.int8)
            else:
                q = qweight
            s = scales.unsqueeze(-1).to(torch.float32)
            w_deq = (q.to(torch.float32) * s).to(torch.bfloat16)
            return F.linear(x, w_deq, bias)

        raise RuntimeError("AllSpark/TileLang unavailable and safe fallback path not found for marlin_int8 W8A16.")

