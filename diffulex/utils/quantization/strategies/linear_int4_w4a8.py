"""
W4A8 Linear quantization strategy (int4 weight + int8 activation).

Notes:
- Weight is per-output-channel symmetric int4 packed into int8 (2 values per byte), with per-channel scales.
- Activation is quantized per-row to int8 with per-row scales.
- GEMM is performed by unpacking int4 -> int8 and using `torch._int_mm` (int8 x int8 -> int32).
  For now we cache the unpacked (and transposed) weight to avoid repeated unpack.
- If int8 GEMM is not available, we fall back to unpack+dequant BF16 + cuBLAS (F.linear).
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
        w4a8_gemm,
        w4a8_scaled_gemm,
        w4a8_fused_act_gemm,
        w8a8_act_quant,
    )
    _TILELANG_AVAILABLE = True
except ImportError:
    _TILELANG_AVAILABLE = False
    w4a8_gemm = None
    w4a8_scaled_gemm = None
    w8a8_act_quant = None
    w4a8_fused_act_gemm = None

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
    abs_max = x.abs().amax(dim=-1, keepdim=False)  # [M]
    scales = (abs_max.clamp(min=1e-8) / 127.0).to(torch.float32)  # [M]
    x_q = torch.round(x.to(torch.float32) / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return x_q, scales


def _quantize_per_row_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric int8 quantization with optional TileLang fused kernel.

    Default: use TileLang fused kernel if available, otherwise fall back to torch ops.

    Env:
        - DIFFULEX_W4A8_USE_TL_ACT_QUANT=0 to force torch fallback.
    """
    use_tl = os.getenv("DIFFULEX_W4A8_USE_TL_ACT_QUANT", "1") == "1"
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
            pass
    return _quantize_per_row_int8_torch(x)


def _int8_mm(a_int8: torch.Tensor, b_int8: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "_int_mm"):
        return torch._int_mm(a_int8, b_int8)
    if hasattr(torch.ops.aten, "_int_mm"):
        return torch.ops.aten._int_mm(a_int8, b_int8)
    raise RuntimeError("No int8 GEMM backend found (torch._int_mm / aten._int_mm missing)")


def _unpack_int4_packed_int8(packed: torch.Tensor, *, original_in_features: int) -> torch.Tensor:
    """Unpack int4 weights stored in int8 bytes (2 nibbles per byte) into int8 values in [-8, 7].

    Args:
        packed: int8 [N, ceil(K/2)]
        original_in_features: K
    Returns:
        unpacked: int8 [N, K]
    """
    if packed.dtype != torch.int8:
        raise TypeError(f"packed weight must be int8, got {packed.dtype}")
    N, packed_K = packed.shape
    expected = (original_in_features + 1) // 2
    if packed_K != expected:
        raise ValueError(f"Packed K mismatch: got {packed_K}, expected {expected} for K={original_in_features}")

    # Interpret bytes as uint8 so we can shift/mask predictably.
    p_u8 = packed.view(torch.uint8)
    low = (p_u8 & 0x0F).to(torch.int16)
    high = ((p_u8 >> 4) & 0x0F).to(torch.int16)

    # Convert unsigned nibble [0..15] to signed int4 [-8..7]
    low_s = torch.where(low >= 8, low - 16, low)
    high_s = torch.where(high >= 8, high - 16, high)

    # Interleave low/high along K
    out = torch.empty((N, packed_K * 2), device=packed.device, dtype=torch.int16)
    out[:, 0::2] = low_s
    out[:, 1::2] = high_s
    out = out[:, :original_in_features].to(torch.int8)
    return out


@register_linear_strategy(weight_dtype="int4", act_dtype="int8")
def _build_linear_int4_w4a8() -> LinearQuantizationStrategy:
    return LinearInt4W4A8Strategy()


class LinearInt4W4A8Strategy(LinearQuantizationStrategy):
    def __init__(self):
        super().__init__()
        # bf16 weight id -> (packed_int8[N,ceil(K/2)], scales_bf16[N])
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # (packed_id, K) -> unpacked_int8[N,K]
        self._unpacked_cache: dict[tuple[int, int], torch.Tensor] = {}
        # (packed_id, K) -> unpacked_t_int8[K,N]
        self._unpacked_t_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._dequant_weight_cache: dict[int, torch.Tensor] = {}
        # (device_index, M_bucket, N, K) -> TileLang config dict for fused kernel
        self._tl_fused_cfg_cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "linear_int4_w4a8"

    @property
    def linear_weight_format(self) -> str:
        return "int4"

    @property
    def linear_act_format(self) -> str:
        return "int8"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # stored as packed int8 bytes (2 weights per byte)
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
        self._unpacked_cache.clear()
        self._unpacked_t_cache.clear()
        self._dequant_weight_cache.clear()
        self._tl_fused_cfg_cache.clear()

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        _ = kwargs
        # Per-output-channel symmetric int4 quantization: scale = absmax/7
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [N,1]
        # Keep scales in fp16 to reduce scale quantization error (A8 paths are sensitive).
        scales = (abs_max.clamp(min=1e-8) / 7.0).to(torch.float16)  # [N,1]
        q = torch.round(tensor / scales).clamp(-8, 7).to(torch.int16)  # [N,K]

        # Pack two int4 into one byte: low nibble for even k, high nibble for odd k.
        N, K = q.shape
        packed_K = (K + 1) // 2
        q_even = q[:, 0::2]
        q_odd = q[:, 1::2]
        if q_odd.shape[1] != q_even.shape[1]:
            q_odd = torch.nn.functional.pad(q_odd, (0, 1), value=0)

        q_even_u = (q_even & 0x0F).to(torch.uint8)
        q_odd_u = (q_odd & 0x0F).to(torch.uint8)
        packed_u8 = q_even_u | (q_odd_u << 4)  # [N, packed_K]
        packed_i8 = packed_u8.view(torch.int8)
        return packed_i8, scales.squeeze(-1)

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        original_in_features = kwargs.get("original_in_features", None)
        if original_in_features is None:
            raise ValueError("original_in_features is required for int4 dequantize")
        scales = scale_or_metadata.get("scales") if isinstance(scale_or_metadata, dict) else scale_or_metadata
        if scales is None:
            raise ValueError("scales required for dequantization")
        w_i8 = _unpack_int4_packed_int8(quantized, original_in_features=original_in_features)  # [N,K]
        deq = w_i8.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)
        return deq.to(torch.bfloat16)

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
        original_in_features = kwargs.pop("original_in_features", None)
        if original_in_features is None:
            raise ValueError("W4A8 requires original_in_features for packed int4 weights")

        # Resolve / cache packed weight + scales
        if weight.dtype == torch.int8:
            if quant_scales is None:
                raise ValueError("weight is int8 (packed int4) but quant_scales is None")
            packed = weight if weight.device == x.device else weight.to(device=x.device)
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
                packed, w_scales = self.quantize_weight_for_kernel(weight, device=x.device)
                self._weight_cache[weight_id] = (packed, w_scales)
            else:
                packed, w_scales = cached
                if packed.device != x.device:
                    packed = packed.to(device=x.device)
                    w_scales = w_scales.to(device=x.device)
                    self._weight_cache[weight_id] = (packed, w_scales)

        # Optional: dequant once and use cuBLAS BF16
        if os.getenv("DIFFULEX_W4A8_PREFER_CUBLAS", "0") == "1":
            deq_key = weight_id
            deq_w = self._dequant_weight_cache.get(deq_key)
            if deq_w is None or deq_w.device != x.device:
                deq_w = self.dequantize(packed, w_scales, original_in_features=original_in_features)
                self._dequant_weight_cache[deq_key] = deq_w
            return F.linear(x, deq_w, bias)

        # Quantize activation per-row to int8
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            x = x.to(torch.bfloat16)
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # Try TileLang fused quant + GEMM first (bf16 activation input).
        use_fused = os.getenv("DIFFULEX_W4A8_USE_TL_FUSED_GEMM", "1") == "1"
        if (
            use_fused
            and _TILELANG_AVAILABLE
            and (w4a8_fused_act_gemm is not None)
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.is_contiguous()
        ):
            try:
                M, K = x.shape
                N, packed_K = packed.shape
                expected_packed_K = (original_in_features + 1) // 2
                assert packed_K == expected_packed_K, (
                    f"Packed K mismatch: got {packed_K}, expected {expected_packed_K} for K={original_in_features}"
                )

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
                cfg_key = (dev_idx, M_bucket, N, original_in_features)
                cfg = self._tl_fused_cfg_cache.get(cfg_key)
                kernel = None

                # TileLang autotune (warmup-only): we set real inputs so the autotuner can benchmark configs.
                if cfg is None and is_warming_up() and set_autotune_inputs is not None:
                    try:
                        with set_autotune_inputs([x_for_kernel, packed, w_scales]):
                            kernel = w4a8_fused_act_gemm(M_bucket, N, original_in_features)
                        cfg = kernel.config
                        self._tl_fused_cfg_cache[cfg_key] = cfg
                    except Exception:
                        # Cache a safe default to avoid retriggering autotune for this key.
                        cfg = _DEFAULT_TL_LINEAR_CFG
                        self._tl_fused_cfg_cache[cfg_key] = cfg

                if cfg is None:
                    cfg = _DEFAULT_TL_LINEAR_CFG
                    self._tl_fused_cfg_cache[cfg_key] = cfg

                if kernel is None:
                    kernel = w4a8_fused_act_gemm(M_bucket, N, original_in_features, **cfg)
                out_full = kernel(x_for_kernel, packed, w_scales)
                out = out_full[:M, :] if M_bucket != M else out_full
                if bias is not None:
                    out = out + bias
                return out
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                warnings.warn(
                    f"W4A8 fused quant GEMM failed, falling back to quantize+GEMM: {error_msg}",
                    UserWarning,
                )

        # Step-local cache for activation quantization (reuse within one step for QKV/gate-up, etc.)
        use_cache = os.getenv("DIFFULEX_W4A8_ACT_QUANT_CACHE", "1") == "1"
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
        N, packed_K = packed.shape
        expected_packed_K = (original_in_features + 1) // 2
        assert packed_K == expected_packed_K, f"Packed K mismatch: got {packed_K}, expected {expected_packed_K} for K={original_in_features}"

        # Try TileLang kernel first if available (uses packed weights directly)
        if _TILELANG_AVAILABLE and (w4a8_scaled_gemm is not None or w4a8_gemm is not None):
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

                    # Prefer fused-scale kernel: outputs bf16 directly.
                    if w4a8_scaled_gemm is not None:
                        kernel = w4a8_scaled_gemm(
                            M_bucket,
                            N,
                            original_in_features,
                            block_M=64,
                            block_N=64,
                            block_K=128,
                            num_stages=2,
                            threads=128,
                        )
                        out_full = kernel(x_q_for_kernel, packed, x_scales_for_kernel, w_scales)
                        out = out_full[:M, :] if M_bucket != M else out_full
                    else:
                        # Fallback to int32-output kernel + python scaling
                        kernel = w4a8_gemm(
                            M_bucket,
                            N,
                            original_in_features,
                            block_M=64,
                            block_N=64,
                            block_K=128,
                            num_stages=2,
                            threads=128,
                        )
                        out_i32_full = kernel(x_q_for_kernel, packed)
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
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                warnings.warn(f"W4A8 TileLang kernel failed, falling back to torch._int_mm: {error_msg}", UserWarning)

        # Fallback: unpack weight and use torch._int_mm
        # Unpack weight to int8 and cache
        packed_key = (id(packed), int(original_in_features))
        w_i8 = self._unpacked_cache.get(packed_key)
        if w_i8 is None or w_i8.device != x.device:
            w_i8 = _unpack_int4_packed_int8(packed, original_in_features=original_in_features)
            self._unpacked_cache[packed_key] = w_i8

        wt = self._unpacked_t_cache.get(packed_key)
        if wt is None or wt.device != x.device:
            wt = w_i8.t().contiguous()
            self._unpacked_t_cache[packed_key] = wt

        # Pad small M for backend constraints (M > 16)
        if M <= 16:
            M_bucket = 17
            x_pad = torch.zeros((M_bucket, K), device=x.device, dtype=torch.int8)
            x_pad[:M, :] = x_q
            x_q_for_mm = x_pad
        else:
            x_q_for_mm = x_q

        try:
            out_i32_full = _int8_mm(x_q_for_mm, wt)
        except Exception as e:
            msg = str(e)
            if len(msg) > 200:
                msg = msg[:200] + "..."
            warnings.warn(f"W4A8 int8 GEMM failed, falling back to BF16 F.linear: {msg}", UserWarning)
            deq_w = self.dequantize(packed, w_scales, original_in_features=original_in_features)
            return F.linear(x, deq_w, bias)

        out_i32 = out_i32_full[:M, :] if M <= 16 else out_i32_full
        out_fp32 = out_i32.to(torch.float32)
        out_fp32 = out_fp32 * x_scales.to(torch.float32).unsqueeze(-1)
        out_fp32 = out_fp32 * w_scales.to(torch.float32).unsqueeze(0)
        out = out_fp32.to(torch.bfloat16)
        if bias is not None:
            out = out + bias
        return out


