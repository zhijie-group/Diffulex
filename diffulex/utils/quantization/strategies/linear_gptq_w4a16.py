"""
GPTQ W4A16 Linear quantization strategy (GPTQ weight + bf16 activation).

Implementation notes:
- Weight quantization: GPTQ format with groupwise quantization
- Activation: kept as bf16 (no activation quantization)
- Storage: GPTQ uses packed int4 weights (qweight), int4 zeros (qzeros), and per-group scales
- Forward path: Dequantize GPTQ weights to bf16, then use F.linear
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

# Try to import TileLang kernel, fallback to None if not available
_TILELANG_AVAILABLE = False
try:
    from diffulex_kernel.python.linear_kernels import gptq_w4a16_gemm
    _TILELANG_AVAILABLE = True
except ImportError:
    gptq_w4a16_gemm = None

try:
    from diffulex.attention.metadata import is_warming_up
    from tilelang.autotuner import set_autotune_inputs
    _AUTOTUNE_AVAILABLE = True
except ImportError:
    _AUTOTUNE_AVAILABLE = False
    is_warming_up = lambda: False
    set_autotune_inputs = lambda *args, **kwargs: lambda f: f


def _unpack_gptq_int4(
    packed: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor:
    """Unpack GPTQ packed int4 weights into int8 values.

    GPTQ packs 2 int4 values per int8 byte:
    - Lower 4 bits: even columns
    - Upper 4 bits: odd columns

    Args:
        packed: int8 tensor [out_features, (in_features + 1) // 2]
        out_features: Original output features
        in_features: Original input features

    Returns:
        unpacked: int8 tensor [out_features, in_features] with values in [-8, 7]
    """
    if packed.dtype != torch.int8:
        raise TypeError(f"packed weight must be int8, got {packed.dtype}")

    out_features_actual, packed_in = packed.shape
    expected_packed_in = (in_features + 1) // 2
    if packed_in != expected_packed_in:
        raise ValueError(
            f"Packed input dimension mismatch: got {packed_in}, "
            f"expected {expected_packed_in} for in_features={in_features}"
        )
    if out_features_actual != out_features:
        raise ValueError(
            f"Output dimension mismatch: got {out_features_actual}, "
            f"expected {out_features}"
        )

    # Interpret bytes as uint8 for bit manipulation
    p_u8 = packed.view(torch.uint8)
    # Extract lower and upper 4 bits
    low_u8 = (p_u8 & 0x0F)  # [0..15]
    high_u8 = ((p_u8 >> 4) & 0x0F)  # [0..15]

    # Convert unsigned nibble [0..15] to signed int4 [-8..7]
    # Packing: int4 [-8, 7] + 8 -> uint8 [0, 15]
    # Unpacking: uint8 [0, 15] - 8 -> int4 [-8, 7]
    low_s = low_u8.to(torch.int16) - 8
    high_s = high_u8.to(torch.int16) - 8

    # Interleave low/high along in_features
    unpacked = torch.empty((out_features, packed_in * 2), device=packed.device, dtype=torch.int16)
    unpacked[:, 0::2] = low_s
    unpacked[:, 1::2] = high_s
    unpacked = unpacked[:, :in_features].to(torch.int8)
    return unpacked


def _dequantize_gptq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_features: int,
    in_features: int,
    group_size: int = 128,
    g_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize GPTQ weights to bf16.

    GPTQ uses groupwise quantization:
    - Weight is quantized per group (group_size consecutive elements)
    - Each group has its own scale and zero point
    - g_idx (optional) maps each weight element to its group

    Args:
        qweight: int8 tensor [out_features, (in_features + 1) // 2] packed int4
        qzeros: int8 tensor [(out_features + group_size - 1) // group_size, (in_features + 1) // 2] packed int4
        scales: float32 tensor [(out_features + group_size - 1) // group_size, in_features]
        out_features: Output features
        in_features: Input features
        group_size: Group size for quantization (default: 128)
        g_idx: Optional int32 tensor [in_features] mapping each weight to its group

    Returns:
        dequantized: bf16 tensor [out_features, in_features]
    """
    device = qweight.device

    # Unpack qweight to int8 [out_features, in_features]
    w_int8 = _unpack_gptq_int4(qweight, out_features=out_features, in_features=in_features)

    # Unpack qzeros to int8 [num_groups, in_features]
    num_groups = (out_features + group_size - 1) // group_size
    if qzeros.shape[0] != num_groups:
        raise ValueError(
            f"qzeros shape mismatch: got {qzeros.shape[0]} groups, "
            f"expected {num_groups} for out_features={out_features}, group_size={group_size}"
        )
    zeros_int8 = _unpack_gptq_int4(qzeros, out_features=num_groups, in_features=in_features)

    # Ensure scales have correct shape [num_groups, in_features]
    if scales.shape != (num_groups, in_features):
        # If scales is [num_groups] or [num_groups, 1], broadcast to [num_groups, in_features]
        if scales.shape == (num_groups,) or scales.shape == (num_groups, 1):
            scales = scales.unsqueeze(-1).expand(num_groups, in_features)
        else:
            raise ValueError(
                f"scales shape mismatch: got {scales.shape}, "
                f"expected ({num_groups}, {in_features}) or ({num_groups},) or ({num_groups}, 1)"
            )

    # Convert to float32 for dequantization
    w_fp32 = w_int8.to(torch.float32)
    zeros_int8_fp32 = zeros_int8.to(torch.float32)  # Quantized zeros (int8)
    scales_fp32 = scales.to(torch.float32)
    
    # Dequantize zeros: zero = zero_quantized * scale
    # zeros_int8 was quantized as: zero_quantized = round(zero / scale)
    # So to recover: zero = zero_quantized * scale
    zeros_fp32 = zeros_int8_fp32 * scales_fp32  # [num_groups, in_features]

    # Dequantize: (weight - zero) * scale
    # w_int8 is [out_features, in_features]
    # zeros_int8 is [num_groups, in_features]
    # scales_fp32 is [num_groups, in_features]

    # For each output channel, determine which group it belongs to
    if g_idx is not None:
        # g_idx maps each output channel to its group
        if g_idx.shape != (out_features,):
            raise ValueError(
                f"g_idx shape mismatch: got {g_idx.shape}, expected ({out_features},)"
            )
        # g_idx: [out_features] -> group_id for each output channel
        group_ids = g_idx.to(torch.int64)  # [out_features]
        # Clamp group_ids to valid range [0, num_groups-1]
        group_ids = torch.clamp(group_ids, 0, num_groups - 1)
        # Gather zeros and scales for each output channel
        # zeros_fp32: [num_groups, in_features], group_ids: [out_features]
        # We need to index along dimension 0 for each output channel
        zeros_for_channel = zeros_fp32[group_ids]  # [out_features, in_features]
        scales_for_channel = scales_fp32[group_ids]  # [out_features, in_features]
    else:
        # Without g_idx, assume sequential grouping: group_id = out_idx // group_size
        group_ids = torch.arange(out_features, device=device) // group_size  # [out_features]
        # Clamp group_ids to valid range
        group_ids = torch.clamp(group_ids, 0, num_groups - 1)
        zeros_for_channel = zeros_fp32[group_ids]  # [out_features, in_features]
        scales_for_channel = scales_fp32[group_ids]  # [out_features, in_features]

    # Dequantize: quantized * scale + zero
    # Quantization formula: quantized = round((weight - zero) / scale)
    # Dequantization formula: weight = quantized * scale + zero
    dequantized = w_fp32 * scales_for_channel + zeros_for_channel
    return dequantized.to(torch.bfloat16)


@register_linear_strategy(weight_dtype="gptq", act_dtype="bf16")
def _build_linear_gptq_w4a16() -> LinearQuantizationStrategy:
    return LinearGPTQW4A16Strategy()


class LinearGPTQW4A16Strategy(LinearQuantizationStrategy):
    """GPTQ W4A16 Linear strategy: GPTQ weight quantization + bf16 activation.

    Current implementation: Python reference using dequantized weights + F.linear.
    Weight quantization: GPTQ format with groupwise quantization (typically group_size=128).
    Activation: kept as bf16 (no activation quantization).

    Lazy cache: Dequantized weights are cached to avoid re-dequantizing on every forward pass.
    """

    def __init__(self):
        """Initialize strategy (no cache needed when using kernel)."""
        super().__init__()
        # TileLang autotune config cache: (device, M_bucket, N, K, num_groups, group_size) -> config dict
        self._tl_autotune_config_cache: dict[tuple[str, int, int, int, int, int], dict] = {}

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
        # GPTQ weights are stored as packed int8 (2 int4 per byte)
        return torch.int8, 1

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """Return shape of scales tensor for GPTQ groupwise quantization.

        For [out_features, in_features] weight with group_size groups:
        - scales shape is [(out_features + group_size - 1) // group_size, in_features]
        """
        if len(original_shape) < 2:
            raise ValueError(f"Expected weight shape with at least 2 dims, got {original_shape}")
        out_features, in_features = original_shape[0], original_shape[1]
        group_size = kwargs.get("group_size", 128)
        num_groups = (out_features + group_size - 1) // group_size
        return (num_groups, in_features)

    def quantize(self, tensor: torch.Tensor, **kwargs):
        """GPTQ quantization is typically done offline, so this is a placeholder."""
        raise NotImplementedError(
            "GPTQ quantization should be done offline using GPTQ tools. "
            "This strategy only supports loading pre-quantized weights."
        )

    def dequantize(
        self,
        quantized: torch.Tensor,
        scale_or_metadata: Any,
        **kwargs
    ) -> torch.Tensor:
        """Dequantize GPTQ weights.

        Args:
            quantized: Not used (kept for interface compatibility)
            scale_or_metadata: Dict with keys:
                - 'qweight': int8 packed int4 weights
                - 'qzeros': int8 packed int4 zeros
                - 'scales': float32 per-group scales
                - 'out_features': int
                - 'in_features': int
                - 'group_size': int (default: 128)
                - 'g_idx': Optional int32 group indices
            **kwargs: Additional arguments

        Returns:
            Dequantized tensor in bf16
        """
        if not isinstance(scale_or_metadata, dict):
            raise ValueError(
                "GPTQ dequantize requires dict metadata with keys: "
                "qweight, qzeros, scales, out_features, in_features, group_size (optional), g_idx (optional)"
            )

        qweight = scale_or_metadata["qweight"]
        qzeros = scale_or_metadata["qzeros"]
        scales = scale_or_metadata["scales"]
        out_features = scale_or_metadata["out_features"]
        in_features = scale_or_metadata["in_features"]
        group_size = scale_or_metadata.get("group_size", 128)
        g_idx = scale_or_metadata.get("g_idx", None)

        return _dequantize_gptq(
            qweight=qweight,
            qzeros=qzeros,
            scales=scales,
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            g_idx=g_idx,
        )

    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """GPTQ quantization is done offline, so this should not be called."""
        raise NotImplementedError(
            "GPTQ quantization should be done offline. "
            "Use set_offline_quantized_weight() to load pre-quantized weights."
        )

    def quantize_act_for_kernel(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """No activation quantization for W4A16 (activation stays bf16)."""
        if device is not None:
            x = x.to(device=device)
        return x, None

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Linear output using GPTQ quantized weights (W4A16).

        Args:
            x: Activation tensor [M, K] (bf16)
            weight: Either bf16 weight [N, K] (fallback) or GPTQ metadata dict
            bias: Optional bias tensor [N]
            quant_kind: Quantization kind (unused)
            **kwargs: May include:
                - gptq_qweight: int8 packed int4 weights [N, (K+1)//2]
                - gptq_qzeros: int8 packed int4 zeros [num_groups, (K+1)//2]
                - gptq_scales: float32 scales [num_groups, K]
                - gptq_group_size: int (default: 128)
                - gptq_g_idx: Optional int32 group indices [N]
                - out_features: int (N)
                - in_features: int (K)
        """
        _ = quant_kind

        # Check if GPTQ tensors are provided directly via kwargs
        qweight = kwargs.pop("gptq_qweight", None)
        qzeros = kwargs.pop("gptq_qzeros", None)
        scales = kwargs.pop("gptq_scales", None)
        group_size = kwargs.pop("gptq_group_size", 128)
        g_idx = kwargs.pop("gptq_g_idx", None)
        out_features = kwargs.pop("out_features", None)
        in_features = kwargs.pop("in_features", None)

        # If GPTQ tensors are provided, use them
        if qweight is not None and qzeros is not None and scales is not None:
            if out_features is None or in_features is None:
                # Infer from x shape
                M, K = x.shape
                if in_features is None:
                    in_features = K
                if out_features is None:
                    # Infer from qweight shape
                    out_features = qweight.shape[0]

            M, K = x.shape
            N = out_features
            num_groups = (N + group_size - 1) // group_size

            # Handle scales shape: broadcast to [num_groups, in_features] if needed
            if scales.shape == (num_groups,):
                scales = scales.unsqueeze(-1).expand(num_groups, in_features)
            elif scales.shape == (num_groups, 1):
                scales = scales.expand(num_groups, in_features)
            elif scales.shape != (num_groups, in_features):
                raise ValueError(
                    f"scales shape mismatch: got {scales.shape}, "
                    f"expected ({num_groups}, {in_features}), ({num_groups},), or ({num_groups}, 1)"
                )

            # Handle GIdx: if None, create sequential indices
            device = qweight.device
            if g_idx is None:
                g_idx = torch.arange(N, device=device, dtype=torch.int32) // group_size
            else:
                g_idx = g_idx.to(device=device, dtype=torch.int32)

            # Ensure all tensors are on the correct device
            qweight = qweight.to(device=x.device)
            qzeros = qzeros.to(device=x.device)
            scales = scales.to(device=x.device, dtype=torch.float32)
            g_idx = g_idx.to(device=x.device)

            # Try to use TileLang kernel if available
            if _TILELANG_AVAILABLE and gptq_w4a16_gemm is not None:
                try:
                    # Check device
                    if x.device.type != 'cuda':
                        return self._fallback_python_forward(
                            x, qweight, qzeros, scales, bias,
                            out_features=N, in_features=in_features,
                            group_size=group_size, g_idx=g_idx,
                        )

                    # M-bucketing: reduce JIT compilation churn
                    M_bucket = M
                    if M > 1:
                        if M <= 64:
                            M_bucket = 1 << (M - 1).bit_length()
                        else:
                            M_bucket = ((M + 63) // 64) * 64

                    x_for_kernel = x
                    if M_bucket != M:
                        x_pad = torch.zeros((M_bucket, K), device=x.device, dtype=x.dtype)
                        x_pad[:M, :] = x
                        x_for_kernel = x_pad

                    # TileLang autotune: use warmup + config cache pattern
                    cache_key = (str(x.device), M_bucket, N, K, num_groups, group_size)
                    config = self._tl_autotune_config_cache.get(cache_key)
                    
                    if _AUTOTUNE_AVAILABLE and is_warming_up() and config is None:
                        # Warmup phase: run autotune with real inputs
                        try:
                            with set_autotune_inputs([x_for_kernel, qweight, qzeros, scales, g_idx]):
                                kernel = gptq_w4a16_gemm(M_bucket, N, K, num_groups, group_size)
                            config = kernel.config
                            self._tl_autotune_config_cache[cache_key] = config
                        except Exception:
                            # Fallback to default config if autotune fails
                            config = None
                    
                    # Use cached config or default parameters
                    if config is not None:
                        kernel = gptq_w4a16_gemm(M_bucket, N, K, num_groups, group_size, **config)
                    else:
                        # Default config (backward compatible)
                        kernel = gptq_w4a16_gemm(M_bucket, N, K, num_groups, group_size, block_M=64, block_N=64, block_K=128, num_stages=2, threads=128)

                    # Call kernel - out_idx=[5] means output is the 6th parameter
                    output_full = kernel(x_for_kernel, qweight, qzeros, scales, g_idx)
                    output = output_full[:M, :] if M_bucket != M else output_full

                    # Add bias if present
                    if bias is not None:
                        output = output + bias

                    return output
                except Exception as e:
                    # Fallback to Python implementation on any error
                    import warnings
                    error_msg = str(e)

                    # Extract meaningful error information
                    if 'sm_' in error_msg and ('not defined' in error_msg or 'fatal' in error_msg):
                        # CUDA architecture not supported - silently fallback
                        pass
                    elif 'Compilation error' in error_msg:
                        # Extract the actual error
                        idx = error_msg.find('Compilation error')
                        after = error_msg[idx + len('Compilation error'):]
                        lines = after.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#') and ('error:' in line.lower() or 'fatal' in line.lower()):
                                error_msg = f"CUDA compilation error: {line[:200]}"
                                break
                        else:
                            error_msg = "CUDA compilation error (see logs for details)"
                        warnings.warn(
                            f"TileLang GPTQ kernel failed, falling back to Python implementation: {error_msg}",
                            UserWarning,
                        )
                    elif 'pipeline' in error_msg.lower() and 'stage' in error_msg.lower():
                        # Pipeline stages mismatch - silently fallback
                        pass
                    else:
                        # Warn for unexpected errors
                        if len(error_msg) > 200:
                            error_msg = error_msg[:200] + "..."
                        warnings.warn(
                            f"TileLang GPTQ kernel failed, falling back to Python implementation: {error_msg}",
                            UserWarning,
                        )
                    return self._fallback_python_forward(
                        x, qweight, qzeros, scales, bias,
                        out_features=N, in_features=in_features,
                        group_size=group_size, g_idx=g_idx,
                    )
            else:
                # TileLang not available, use Python fallback
                return self._fallback_python_forward(
                    x, qweight, qzeros, scales, bias,
                    out_features=N, in_features=in_features,
                    group_size=group_size, g_idx=g_idx,
                )

        # Fallback: if weight is a regular bf16 tensor, use it directly
        if isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16:
            return F.linear(x, weight, bias)

        raise ValueError(
            "GPTQ strategy requires gptq_qweight, gptq_qzeros, and gptq_scales to be provided "
            "via kwargs or weight must be a bf16 tensor (fallback mode)"
        )

    def _fallback_python_forward(
        self,
        x: torch.Tensor,
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        out_features: int,
        in_features: int,
        group_size: int,
        g_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fallback Python implementation: dequantize + F.linear."""
        dequant_weight = _dequantize_gptq(
            qweight=qweight.to(device=x.device),
            qzeros=qzeros.to(device=x.device),
            scales=scales.to(device=x.device),
            out_features=out_features,
            in_features=in_features,
            group_size=group_size,
            g_idx=g_idx.to(device=x.device) if g_idx is not None else None,
        )
        return F.linear(x, dequant_weight, bias)

    def clear_cache(self) -> None:
        """Clear cache (no-op, kept for compatibility)."""
        pass
