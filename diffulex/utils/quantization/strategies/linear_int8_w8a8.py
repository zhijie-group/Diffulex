"""
W8A8 Linear quantization strategy (int8 weight + int8 activation).

Implementation notes:
- We keep per-output-channel weight scales (same as W8A16).
- We quantize activations per-row (per token) to int8 and keep per-row scales.
- GEMM uses `torch._int_mm` (int8 x int8 -> int32) when available.
  This op has a small-M constraint on some builds (e.g. M must be > 16), so we pad M minimally.
- If int8 GEMM is not available, we fall back to dequantized BF16 + cuBLAS (F.linear).
"""

from __future__ import annotations

from typing import Any, Optional

import os
import warnings

import torch
import torch.nn.functional as F

from diffulex.attention.metadata import is_warming_up
from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

try:
    from diffulex_kernel.python.linear_kernels import (
        w8a8_gemm,
        w8a8_scaled_gemm,
        w8a8_act_quant,
        w8a8_fused_act_gemm,
    )
    _TILELANG_AVAILABLE = True
except ImportError:
    _TILELANG_AVAILABLE = False
    w8a8_gemm = None
    w8a8_scaled_gemm = None
    w8a8_act_quant = None
    w8a8_fused_act_gemm = None

try:
    # Optional: only needed for TileLang autotune warmup.
    from tilelang.autotuner import set_autotune_inputs  # type: ignore
except Exception:
    set_autotune_inputs = None


_DEFAULT_TL_LINEAR_CFG: dict[str, Any] = {
    "block_M": 64,
    "block_N": 64,
    "block_K": 128,
    "num_stages": 2,
    "threads": 128,
}


def _quantize_per_row_int8_torch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 quantization.

    Returns:
        x_q: int8 [M, K]
        x_scales: float32 [M] where dequant is x_q.float() * x_scales[:, None]
    """
    # x: [M, K]
    abs_max = x.abs().amax(dim=-1, keepdim=False)  # [M]
    scales = (abs_max.clamp(min=1e-8) / 127.0).to(torch.float32)  # [M]
    x_q = torch.round(x.to(torch.float32) / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return x_q, scales


def _quantize_per_row_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 quantization with optional TileLang fused kernel.

    Default: use TileLang fused kernel if available, otherwise fall back to torch ops.

    Env:
        - DIFFULEX_W8A8_USE_TL_ACT_QUANT=0 to force torch fallback.
    """
    use_tl = os.getenv("DIFFULEX_W8A8_USE_TL_ACT_QUANT", "1") == "1"
    if (
        use_tl
        and _TILELANG_AVAILABLE
        and (w8a8_act_quant is not None)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.dim() == 2
    ):
        m, k = x.shape
        # Choose a small set of block_M values to reduce wasted work on decode small-M.
        if m <= 16:
            block_m = 16
        elif m <= 32:
            block_m = 32
        else:
            block_m = 64
        try:
            kernel = w8a8_act_quant(
                m,
                k,
                block_M=block_m,
                block_K=256,
                threads=128,
            )
            x_q, scales = kernel(x)
            return x_q, scales
        except Exception:
            # Fall back silently to torch path for robustness (e.g., unsupported arch/toolchain).
            pass
    return _quantize_per_row_int8_torch(x)


def _int8_mm(a_int8: torch.Tensor, b_int8: torch.Tensor) -> torch.Tensor:
    """int8 GEMM -> int32.

    We prefer `torch._int_mm` when present.
    """
    if hasattr(torch, "_int_mm"):
        return torch._int_mm(a_int8, b_int8)
    if hasattr(torch.ops.aten, "_int_mm"):
        return torch.ops.aten._int_mm(a_int8, b_int8)
    raise RuntimeError("No int8 GEMM backend found (torch._int_mm / aten._int_mm missing)")


@register_linear_strategy(weight_dtype="int8", act_dtype="int8")
def _build_linear_int8_w8a8() -> LinearQuantizationStrategy:
    return LinearInt8W8A8Strategy()


class LinearInt8W8A8Strategy(LinearQuantizationStrategy):
    """W8A8 Linear strategy: int8 weight + int8 activation, output bf16."""

    def __init__(self):
        super().__init__()
        # weight_id -> (qweight_int8[N,K], scales_bf16[N])
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # weight_id -> qweight_t_int8[K,N] (for torch._int_mm)
        self._weight_t_cache: dict[int, torch.Tensor] = {}
        # speed-first option (uses extra memory)
        self._dequant_weight_cache: dict[int, torch.Tensor] = {}
        # (device_index, M_bucket, N, K) -> TileLang config dict for fused kernel
        self._tl_fused_cfg_cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}

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

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """Return shape of scales tensor for per-channel quantization.
        
        For [out_features, in_features] weight, scales shape is [out_features].
        """
        _ = kwargs
        if len(original_shape) < 2:
            raise ValueError(f"Expected weight shape with at least 2 dims, got {original_shape}")
        # Per-output-channel: scales shape is [out_features]
        return (original_shape[0],)

    def clear_cache(self) -> None:
        self._weight_cache.clear()
        self._weight_t_cache.clear()
        self._dequant_weight_cache.clear()
        self._tl_fused_cfg_cache.clear()

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        # Per-output-channel symmetric quantization: scales shape [N]
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [N, 1]
        # Keep scales in fp16 to reduce scale quantization error (A8 paths are sensitive).
        scales = (abs_max.clamp(min=1e-8) / 127.0).to(torch.float16)  # [N, 1]
        q = torch.round(tensor / scales).clamp(-128, 127).to(torch.int8)
        return q, scales.squeeze(-1)

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        _ = kwargs
        scales = scale_or_metadata.get("scales") if isinstance(scale_or_metadata, dict) else scale_or_metadata
        if scales is None:
            raise ValueError("scales required for dequantization")
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)  # [N, 1]
        return (quantized.to(torch.float32) * scales.to(torch.float32)).to(torch.bfloat16)

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        if device is not None:
            weight = weight.to(device=device)
        return self.quantize(weight)

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

        quant_scales = kwargs.pop("quant_scales", None)

        # Resolve / cache quantized weight + scales
        if weight.dtype == torch.int8:
            if quant_scales is None:
                raise ValueError("weight is int8 but quant_scales is None; expected per-channel scales tensor")
            qweight = weight if weight.device == x.device else weight.to(device=x.device)
            w_scales = quant_scales
            # Prefer fp16 scales for quality (and fused kernel expects fp16 scales).
            if w_scales.dtype != torch.float16:
                w_scales = w_scales.to(dtype=torch.float16)
            if w_scales.device != x.device:
                w_scales = w_scales.to(device=x.device)
            weight_id = id(weight)
        else:
            weight_id = id(weight)
            cached = self._weight_cache.get(weight_id)
            if cached is None:
                qweight, w_scales = self.quantize_weight_for_kernel(weight, device=x.device)
                self._weight_cache[weight_id] = (qweight, w_scales)
            else:
                qweight, w_scales = cached
                if qweight.device != x.device:
                    qweight = qweight.to(device=x.device)
                    w_scales = w_scales.to(device=x.device)
                    self._weight_cache[weight_id] = (qweight, w_scales)

        # Optional: use cuBLAS BF16 (dequant once)
        if os.getenv("DIFFULEX_W8A8_PREFER_CUBLAS", "0") == "1":
            deq_key = weight_id
            deq_w = self._dequant_weight_cache.get(deq_key)
            if deq_w is None or deq_w.device != x.device:
                s = w_scales
                if s.dim() == 1:
                    s = s.unsqueeze(-1)
                deq_w = (qweight.to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
                self._dequant_weight_cache[deq_key] = deq_w
            return F.linear(x, deq_w, bias)

        # Quantize activation per-row
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            x = x.to(torch.bfloat16)
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Try TileLang fused quant + GEMM first (bf16 activation input).
        use_fused = os.getenv("DIFFULEX_W8A8_USE_TL_FUSED_GEMM", "1") == "1"
        if (
            use_fused
            and _TILELANG_AVAILABLE
            and (w8a8_fused_act_gemm is not None)
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.is_contiguous()
        ):
            try:
                M, K = x.shape
                N, K_w = qweight.shape
                assert K == K_w, f"K dimension mismatch: {K} != {K_w}"

                # Reduce TileLang JIT compilation churn using M-bucketing (similar to W8A16)
                M_bucket = M
                if M > 1:
                    if M <= 64:
                        M_bucket = 1 << (M - 1).bit_length()
                    else:
                        M_bucket = ((M + 63) // 64) * 64

                x_for_kernel = x
                if M_bucket != M:
                    x_pad = torch.zeros((M_bucket, K), device=x.device, dtype=torch.bfloat16)
                    x_pad[:M, :] = x
                    x_for_kernel = x_pad

                dev_idx = x.device.index or 0
                cfg_key = (dev_idx, M_bucket, N, K)
                cfg = self._tl_fused_cfg_cache.get(cfg_key)
                kernel = None

                # Only run autotune during warmup when autotuner inputs are available.
                if cfg is None and is_warming_up() and set_autotune_inputs is not None:
                    try:
                        with set_autotune_inputs([x_for_kernel, qweight, w_scales]):
                            kernel = w8a8_fused_act_gemm(M_bucket, N, K)
                        # Only cache config if autotune succeeded (kernel has valid config)
                        if hasattr(kernel, 'config') and kernel.config is not None:
                            cfg = kernel.config
                            self._tl_fused_cfg_cache[cfg_key] = cfg
                    except Exception as autotune_err:
                        # Autotune failed (e.g., all configs failed to compile), use default
                        autotune_msg = str(autotune_err)
                        if len(autotune_msg) > 150:
                            autotune_msg = autotune_msg[:150] + "..."
                        warnings.warn(
                            f"W8A8 fused autotune failed ({autotune_msg}), using default config",
                            UserWarning,
                        )
                        kernel = None

                # Non-warmup path: keep deterministic behavior with a default config.
                if cfg is None:
                    cfg = _DEFAULT_TL_LINEAR_CFG

                if kernel is None:
                    kernel = w8a8_fused_act_gemm(M_bucket, N, K, **cfg)
                out_full = kernel(x_for_kernel, qweight, w_scales)
                out = out_full[:M, :] if M_bucket != M else out_full
                if bias is not None:
                    out = out + bias
                return out
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                warnings.warn(
                    f"W8A8 fused quant GEMM failed, falling back to quantize+GEMM: {error_msg}",
                    UserWarning,
                )

        # Step-local cache for activation quantization (reuse within one step for QKV/gate-up, etc.)
        use_cache = os.getenv("DIFFULEX_W8A8_ACT_QUANT_CACHE", "1") == "1"
        cached = None
        if use_cache:
            try:
                from diffulex.utils.quantization.context import get_cached_act_quant, set_cached_act_quant
                cached = get_cached_act_quant(x)
            except Exception:
                cached = None
        if cached is not None:
            x_q, x_scales = cached
        else:
            x_q, x_scales = _quantize_per_row_int8(x)
            if use_cache:
                try:
                    set_cached_act_quant(x, x_q, x_scales)
                except Exception:
                    pass
        if x_q.device != x.device:
            x_q = x_q.to(device=x.device)
            x_scales = x_scales.to(device=x.device)

        # Get shapes
        M, K = x_q.shape
        N, K_w = qweight.shape
        assert K == K_w, f"K dimension mismatch: {K} != {K_w}"

        # Try TileLang kernel first if available
        if _TILELANG_AVAILABLE and (w8a8_scaled_gemm is not None or w8a8_gemm is not None):
            try:
                # Check device
                if x.device.type != 'cuda':
                    # Fall through to _int8_mm fallback
                    pass
                else:
                    # Reduce TileLang JIT compilation churn using M-bucketing (similar to W8A16)
                    M_bucket = M
                    if M > 1:
                        if M <= 64:
                            M_bucket = 1 << (M - 1).bit_length()
                        else:
                            M_bucket = ((M + 63) // 64) * 64

                    x_q_for_kernel = x_q
                    if M_bucket != M:
                        x_pad = torch.zeros((M_bucket, K), device=x.device, dtype=torch.int8)
                        x_pad[:M, :] = x_q
                        x_q_for_kernel = x_pad
                        x_scales_pad = torch.zeros((M_bucket,), device=x.device, dtype=torch.float32)
                        x_scales_pad[:M] = x_scales.to(torch.float32)
                        x_scales_for_kernel = x_scales_pad
                    else:
                        x_scales_for_kernel = x_scales.to(torch.float32)

                    # Prefer fused-scale kernel: outputs bf16 directly, avoiding large int32->fp32 postprocessing.
                    if w8a8_scaled_gemm is not None:
                        kernel = w8a8_scaled_gemm(
                            M_bucket,
                            N,
                            K,
                            block_M=64,
                            block_N=64,
                            block_K=128,
                            num_stages=2,
                            threads=128,
                        )
                        out_full = kernel(x_q_for_kernel, qweight, x_scales_for_kernel, w_scales)
                        out = out_full[:M, :] if M_bucket != M else out_full
                    else:
                        # Fallback to int32-output kernel + python scaling
                        kernel = w8a8_gemm(
                            M_bucket,
                            N,
                            K,
                            block_M=64,
                            block_N=64,
                            block_K=128,
                            num_stages=2,
                            threads=128,
                        )
                        out_i32_full = kernel(x_q_for_kernel, qweight)
                        out_i32 = out_i32_full[:M, :] if M_bucket != M else out_i32_full

                        out_fp32 = out_i32.to(torch.float32)
                        out_fp32 = out_fp32 * x_scales.to(torch.float32).unsqueeze(-1)
                        out_fp32 = out_fp32 * w_scales.to(torch.float32).unsqueeze(0)
                        out = out_fp32.to(torch.bfloat16)

                    if bias is not None:
                        out = out + bias
                    return out
            except Exception as e:
                # Fallback to _int8_mm on any kernel error
                import warnings
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                warnings.warn(f"W8A8 TileLang kernel failed, falling back to torch._int_mm: {error_msg}", UserWarning)

        # Fallback: use torch._int_mm
        # Prepare weight transpose for int8 GEMM: [N,K] -> [K,N]
        wt = self._weight_t_cache.get(weight_id)
        if wt is None or wt.device != x.device:
            wt = qweight.t().contiguous()
            self._weight_t_cache[weight_id] = wt

        # Some builds require M > 16 for int8 GEMM; pad minimally.
        if M <= 16:
            M_bucket = 17
            x_pad = torch.zeros((M_bucket, K), device=x.device, dtype=torch.int8)
            x_pad[:M, :] = x_q
            x_q_for_mm = x_pad
        else:
            x_q_for_mm = x_q

        try:
            out_i32_full = _int8_mm(x_q_for_mm, wt)  # [M_bucket, N] int32
        except Exception as e:
            # Fallback: dequant + BF16 GEMM
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            warnings.warn(f"W8A8 int8 GEMM failed, falling back to BF16 F.linear: {msg}", UserWarning)
            deq_w = self.dequantize(qweight, w_scales)
            return F.linear(x, deq_w, bias)

        out_i32 = out_i32_full[:M, :] if M <= 16 else out_i32_full

        # Apply scales: int32 * x_scale[m] * w_scale[n]
        out_fp32 = out_i32.to(torch.float32)
        out_fp32 = out_fp32 * x_scales.to(torch.float32).unsqueeze(-1)
        out_fp32 = out_fp32 * w_scales.to(torch.float32).unsqueeze(0)
        out = out_fp32.to(torch.bfloat16)

        if bias is not None:
            out = out + bias
        return out


