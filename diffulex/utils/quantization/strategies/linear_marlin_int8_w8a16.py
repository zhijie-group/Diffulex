"""W8A16 Linear quantization strategy using vLLM custom ops.

This strategy uses vLLM's fused AllSpark W8A16 path via `vllm._custom_ops`:
- per-out-channel int8 quantization stored as uint8 (+128 bias)
- one-time N32K16 reorder (AllSpark repack)
- fused dequant + GEMM (AllSpark w8a16 gemm)

Important:
- We intentionally do NOT vendor/compile a local AllSpark/Marlin extension in
  Diffulex anymore. If `vllm._custom_ops` is unavailable, this strategy fails
  fast (instead of silently compiling or falling back to a slow/oom-prone path).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

try:
    import vllm._custom_ops as _vllm_ops
except Exception:
    _vllm_ops = None


def _allspark_is_available() -> bool:
    return bool(
        _vllm_ops is not None
        and hasattr(_vllm_ops, "allspark_w8a16_gemm")
        and hasattr(_vllm_ops, "allspark_repack_weight")
    )

def _allspark_repack_weight(b_qweight_kn: torch.Tensor, scales_1xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack KxN uint8 qweight + 1xN scales into (N_32,K) + (1,N_32) for AllSpark GEMM."""
    if _vllm_ops is None or not hasattr(_vllm_ops, "allspark_repack_weight"):
        raise RuntimeError("vLLM custom ops are unavailable: missing `allspark_repack_weight`.")
    q_reorder, s_reorder, _ = _vllm_ops.allspark_repack_weight(
        b_qweight_kn,
        scales_1xn,
        None,
        False,
    )
    return q_reorder, s_reorder


@register_linear_strategy(weight_dtype="marlin_int8", act_dtype="bf16")
def _build_linear_marlin_int8_w8a16() -> LinearQuantizationStrategy:
    return LinearMarlinInt8W8A16Strategy()


class LinearMarlinInt8W8A16Strategy(LinearQuantizationStrategy):
    """W8A16 strategy using vLLM custom ops (AllSpark fused GEMM + repack)."""

    def __init__(self) -> None:
        super().__init__()
        # Cache for bf16 Parameters only (load-time quantized path bypasses this).
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Cache device info and thresholds to reduce per-call CPU overhead.
        self._sm_info_cache: dict[int, tuple[int, int]] = {}
        self._quant_block_n: int = 256
        self._cublas_m_thr: int = 256
        # One-time availability check (avoid calling `_allspark_is_available()` on every linear).
        self._allspark_available: bool = _allspark_is_available()

    def configure(self, *, diffulex_config: Any | None = None) -> None:
        # Prefer explicit config fields over environment-variable based tuning.
        if diffulex_config is None:
            return
        try:
            bn = int(getattr(diffulex_config, "linear_w8a16_quant_block_n", self._quant_block_n))
            self._quant_block_n = max(1, bn)
        except Exception:
            pass
        try:
            thr = int(getattr(diffulex_config, "linear_w8a16_allspark_cublas_m_threshold", self._cublas_m_thr))
            self._cublas_m_thr = max(1, thr)
        except Exception:
            pass

    @property
    def name(self) -> str:
        # NOTE: Keep strategy naming consistent with the public W8A16 INT8 path.
        # The underlying implementation is a Marlin/AllSpark-style fused kernel,
        # but the user-facing strategy name should not be tied to a particular kernel brand.
        return "linear_int8_w8a16"

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

        # IMPORTANT (OOM fix):
        # Avoid allocating a full [N,K] fp32 copy (and an extra transpose buffer).
        # Quantize in small row blocks and (when using AllSpark) write directly into
        # the repack input layout B_kn=[K,N], so we never materialize q_u8 + transpose.
        block_n = max(1, int(self._quant_block_n))

        if self._allspark_available:
            # AllSpark repack expects B in (K,N) contiguous layout.
            b_kn = torch.empty((k, n), device=weight.device, dtype=torch.uint8)  # [K,N]
            for i in range(0, n, block_n):
                j = min(i + block_n, n)
                w_blk = weight[i:j, :]  # [B,K]
                s_blk = scales[i:j].unsqueeze(-1)  # [B,1]
                # Quantize to signed int in bf16 to minimize temporary memory.
                q_i16 = torch.round(w_blk / s_blk).clamp(-128, 127).to(torch.int16)  # [B,K]
                q_u8_blk = (q_i16 + 128).to(torch.uint8)  # [B,K]
                # Write directly into [K,N] buffer.
                b_kn[:, i:j] = q_u8_blk.transpose(0, 1)
        else:
            # Fallback storage (no reorder). Keep [N,K] and [N] (padded to N_32).
            # Note: forward will detect unavailable allspark and fallback further.
            q_pad = torch.full((n_32, k), 128, device=weight.device, dtype=torch.uint8)
            for i in range(0, n, block_n):
                j = min(i + block_n, n)
                w_blk = weight[i:j, :]  # [B,K]
                s_blk = scales[i:j].unsqueeze(-1)  # [B,1]
                q_i16 = torch.round(w_blk / s_blk).clamp(-128, 127).to(torch.int16)  # [B,K]
                q_pad[i:j, :] = (q_i16 + 128).to(torch.uint8)
            if n_32 != n:
                s_pad = torch.zeros((n_32,), device=scales.device, dtype=torch.bfloat16)
                s_pad[:n] = scales
                return q_pad.contiguous(), s_pad.contiguous()
            return q_pad[:n, :].contiguous(), scales.contiguous()

        # vLLM expects scales in [1, N] layout for repack.
        q_reorder, s_reorder_1xn = _allspark_repack_weight(
            b_kn.contiguous(),
            scales.unsqueeze(0).contiguous(),
        )

        # Store scales as 1D for LinearBase buffers; linear_forward will reshape as needed.
        s_1d = s_reorder_1xn.reshape(-1).to(dtype=torch.bfloat16)
        return q_reorder.contiguous(), s_1d.contiguous()

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
        # get_device_properties is relatively expensive on hot paths; cache per device index.
        try:
            idx = int(device.index) if device.index is not None else int(torch.cuda.current_device())
        except Exception:
            idx = -1
        cached = self._sm_info_cache.get(idx)
        if cached is not None:
            return cached
        try:
            props = torch.cuda.get_device_properties(device)
            sm_count = int(getattr(props, "multi_processor_count", 0))
            sm_version = int(props.major) * 10 + int(props.minor)
            self._sm_info_cache[idx] = (sm_count, sm_version)
            return sm_count, sm_version
        except Exception:
            self._sm_info_cache[idx] = (0, 0)
            return 0, 0

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
        if not self._allspark_available or _vllm_ops is None:
            # correctness fallback only when bf16 weight exists
            if weight is not None and getattr(weight, "dtype", None) in (torch.float16, torch.bfloat16):
                return F.linear(x, weight, bias)
            raise RuntimeError(
                "vLLM AllSpark W8A16 fused kernel is unavailable. "
                "Please ensure vLLM custom ops are installed and loadable (`import vllm._custom_ops`)."
            )

        orig_shape = x.shape
        x2 = x.reshape(-1, x.shape[-1]) if x.dim() != 2 else x
        if x2.device.type != "cuda":
            if weight is not None and getattr(weight, "dtype", None) in (torch.float16, torch.bfloat16):
                return F.linear(x, weight, bias)
            raise RuntimeError("AllSpark W8A16 requires CUDA inputs.")

        if x2.dtype != torch.bfloat16:
            x2 = x2.to(dtype=torch.bfloat16)
        if not x2.is_contiguous():
            x2 = x2.contiguous()

        # Load-time quantized module path: weight is uint8/int8 buffer and scales provided.
        if weight is not None and weight.dtype in (torch.uint8, torch.int8):
            if quant_scales is None:
                raise ValueError("quant_scales is required when weight is quantized")
            qweight = weight
            scales = quant_scales
        else:
            # Safety net for bf16 weights (should be rare in steady-state).
            weight_id = id(weight)
            cached = self._weight_cache.get(weight_id)
            if cached is None or cached[0].device != x2.device:
                qweight, scales = self.quantize_weight_for_kernel(weight, device=x2.device)
                self._weight_cache[weight_id] = (qweight, scales)
            else:
                qweight, scales = cached

        m, k = x2.shape
        n_32, k_w = qweight.shape
        if k_w != k or (k & 15) != 0:
            if weight is not None and getattr(weight, "dtype", None) in (torch.float16, torch.bfloat16):
                y = F.linear(x, weight, bias)
                return y
            raise RuntimeError(f"AllSpark W8A16 requires K%16==0 and matching K. Got x.K={k}, w.K={k_w}.")

        n = int(out_features) if out_features is not None else (int(bias.numel()) if bias is not None else int(min(scales.numel(), n_32)))
        n = n_32 if (n <= 0 or n > n_32) else n
        scales_1xn = scales if scales.dim() == 2 else scales.view(1, -1)

        sm_count, sm_version = self._get_sm_info(x2.device)
        y2 = _vllm_ops.allspark_w8a16_gemm(
            x2,
            qweight,
            scales_1xn,
            None,  # b_qzeros
            n,
            -1,  # group_size (only supports -1)
            sm_count,
            sm_version,
            self._cublas_m_thr,
            False,  # has_zp
            True,  # n32k16_reorder
        )
        if bias is not None:
            y2 = y2 + bias
        if orig_shape == x2.shape:
            return y2
        if x.dim() == 1:
            return y2.squeeze(0)
        return y2.reshape(*orig_shape[:-1], y2.shape[-1])

    # NOTE: We intentionally do not provide a generic dequantize+F.linear fallback for reordered weights.
    # It materializes a full bf16 matrix and is prone to OOM on large models.

