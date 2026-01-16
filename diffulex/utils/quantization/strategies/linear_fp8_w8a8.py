"""
FP8 W8A8 Linear quantization strategy (FP8 weight + FP8 activation).

Implementation notes:
- Weight quantization: per-output-channel FP8 quantization (fp8_e4m3 or fp8_e5m2)
- Activation quantization: per-row FP8 quantization
- Storage: FP8 weights and activations use uint8 storage + view(fp8_dtype) pattern
- Scale management:
  - Weight scales: per-channel [out_features], dtype: float16
  - Activation scales: per-row [M], dtype: float32
- Forward path: Python fallback (dequantize both FP8 weight and activation → bf16, then F.linear)
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy
from diffulex.utils.quantization.kv_cache_dtype import (
    parse_kv_cache_dtype,
    _get_fp8_e4m3_dtype,
    _get_fp8_e5m2_dtype,
)

# Try to import TileLang kernels, fallback to None if not available
_TILELANG_AVAILABLE = False
_fp8_e4m3_w8a8_gemm = None
_fp8_e5m2_w8a8_gemm = None

try:
    from diffulex_kernel.python.linear_kernels import (
        fp8_e4m3_w8a8_gemm,
        fp8_e5m2_w8a8_gemm,
    )
    _TILELANG_AVAILABLE = True
    _fp8_e4m3_w8a8_gemm = fp8_e4m3_w8a8_gemm
    _fp8_e5m2_w8a8_gemm = fp8_e5m2_w8a8_gemm
except ImportError:
    pass

try:
    from diffulex.attention.metadata import is_warming_up
    from tilelang.autotuner import set_autotune_inputs
    _AUTOTUNE_AVAILABLE = True
except ImportError:
    _AUTOTUNE_AVAILABLE = False
    is_warming_up = lambda: False
    set_autotune_inputs = lambda *args, **kwargs: lambda f: f


def _quantize_per_row_fp8(
    x: torch.Tensor,
    fp8_view_dtype: torch.dtype,
    fp8_min: float,
    fp8_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row symmetric FP8 quantization.
    
    Args:
        x: Input tensor [M, K] in bf16/fp16/fp32
        fp8_view_dtype: FP8 dtype (e.g., torch.float8_e4m3fn)
        fp8_min: Minimum FP8 value
        fp8_max: Maximum FP8 value
    
    Returns:
        x_q: uint8 [M, K] (FP8 storage)
        x_scales: float32 [M] where dequant is x_q.view(fp8_dtype).float() * x_scales[:, None]
    """
    # x: [M, K]
    abs_max = x.abs().amax(dim=-1, keepdim=False)  # [M]
    eps = 1e-8
    scales = (abs_max.clamp(min=eps) / fp8_max).to(torch.float32)  # [M]
    
    # Quantize: clamp(x / scale, fp8_min, fp8_max).to(fp8_dtype).view(uint8)
    descale = 1.0 / scales.unsqueeze(-1)  # [M, 1]
    quantized = (x.to(torch.float32) * descale).clamp(
        min=fp8_min,
        max=fp8_max
    )
    quantized_fp8 = quantized.to(fp8_view_dtype)
    quantized_uint8 = quantized_fp8.view(torch.uint8)
    
    return quantized_uint8, scales


@register_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
def _build_linear_fp8_e4m3_w8a8() -> LinearQuantizationStrategy:
    return LinearFP8W8A8Strategy("fp8_e4m3", "fp8_e4m3")


@register_linear_strategy(weight_dtype="fp8_e5m2", act_dtype="fp8_e5m2")
def _build_linear_fp8_e5m2_w8a8() -> LinearQuantizationStrategy:
    return LinearFP8W8A8Strategy("fp8_e5m2", "fp8_e5m2")


class LinearFP8W8A8Strategy(LinearQuantizationStrategy):
    """FP8 W8A8 Linear strategy: FP8 weight + FP8 activation quantization, output bf16.
    
    Current implementation: Python reference using dequantized weights and activations + F.linear.
    Weight quantization: per-output-channel FP8 quantization.
    Activation quantization: per-row FP8 quantization.
    """
    
    def __init__(self, weight_dtype: str = "fp8_e4m3", act_dtype: str = "fp8_e4m3"):
        """
        Initialize FP8 W8A8 strategy.
        
        Args:
            weight_dtype: FP8 dtype string for weights ("fp8_e4m3" or "fp8_e5m2")
            act_dtype: FP8 dtype string for activations ("fp8_e4m3" or "fp8_e5m2")
        """
        super().__init__()
        self.weight_dtype_str = weight_dtype
        self.act_dtype_str = act_dtype
        self.weight_spec = parse_kv_cache_dtype(weight_dtype)
        self.act_spec = parse_kv_cache_dtype(act_dtype)
        if not self.weight_spec.is_fp8 or not self.act_spec.is_fp8:
            raise ValueError(f"Expected FP8 dtypes, got weight={weight_dtype}, act={act_dtype}")
        
        # Cache: weight_id -> (quantized_weight_uint8, scales_float16)
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Optional cache: weight_id -> bf16 dequantized weight (speed-first; uses extra memory)
        self._dequant_weight_cache: dict[int, torch.Tensor] = {}
        # TileLang autotune config cache: (device, M_bucket, N, K) -> config dict
        self._tl_autotune_config_cache: dict[tuple[str, int, int, int], dict] = {}
    
    @property
    def name(self) -> str:
        return f"linear_fp8_{self.weight_dtype_str}_w8a8"
    
    @property
    def linear_weight_format(self) -> str:
        return self.weight_dtype_str
    
    @property
    def linear_act_format(self) -> str:
        return self.act_dtype_str
    
    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # FP8 weights are stored as uint8 (1 byte per element)
        return torch.uint8, 1
    
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
        self._dequant_weight_cache.clear()
    
    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """Quantize tensor to FP8 with per-channel (per-output) scales.
        
        Args:
            tensor: Weight tensor of shape [out_features, in_features]
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            (quantized_tensor_uint8, scales_float16): quantized_tensor is uint8 (FP8 storage),
                scales is float16 [out_features]
        """
        _ = kwargs
        assert self.weight_spec.fp8_view_dtype is not None
        assert self.weight_spec.fp8_min is not None and self.weight_spec.fp8_max is not None
        
        # Per-output-channel quantization: compute scale for each output channel
        # shape: [out_features, in_features] -> scales shape: [out_features]
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [out_features, 1]
        eps = 1e-8
        fp8_max = float(self.weight_spec.fp8_max)
        
        # Compute scales: abs_max / fp8_max
        # Use float16 for weight scales (W8A8 paths are sensitive to scale precision)
        scales = (abs_max.clamp(min=eps) / fp8_max).to(torch.float16)  # [out_features, 1]
        
        # Quantize: clamp(tensor / scale, fp8_min, fp8_max).to(fp8_dtype).view(uint8)
        descale = 1.0 / scales  # [out_features, 1]
        quantized = (tensor.to(torch.float32) * descale).clamp(
            min=float(self.weight_spec.fp8_min),
            max=float(self.weight_spec.fp8_max)
        )
        quantized_fp8 = quantized.to(self.weight_spec.fp8_view_dtype)
        quantized_uint8 = quantized_fp8.view(torch.uint8)
        
        scales_1d = scales.squeeze(-1)  # [out_features]
        
        return quantized_uint8, scales_1d
    
    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """Dequantize FP8 tensor back to bf16 using per-channel scales.
        
        Args:
            quantized: uint8 tensor [out_features, in_features] (FP8 storage)
            scale_or_metadata: scales tensor [out_features] or dict with 'scales'
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            Dequantized tensor in bf16
        """
        _ = kwargs
        assert self.weight_spec.fp8_view_dtype is not None
        
        if isinstance(scale_or_metadata, dict):
            scales = scale_or_metadata.get("scales")
        else:
            scales = scale_or_metadata
        
        if scales is None:
            raise ValueError("scales required for dequantization")
        
        # View uint8 as FP8 dtype
        fp8_tensor = quantized.view(self.weight_spec.fp8_view_dtype).to(torch.float32)
        
        # Ensure scales have correct shape for broadcasting
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)  # [out_features, 1]
        
        # Dequantize: quantized * scales
        dequantized = fp8_tensor * scales.to(torch.float32)
        return dequantized.to(torch.bfloat16)
    
    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Quantize weight to FP8 with per-channel scales.
        
        Returns:
            (quantized_weight_uint8, scales_float16): quantized_weight is uint8 [out, in],
                scales is float16 [out]
        """
        _ = kwargs
        if device is not None:
            weight = weight.to(device=device)
        
        quantized, scales = self.quantize(weight)
        return quantized, scales
    
    def quantize_act_for_kernel(
        self,
        x: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Quantize activation to FP8 with per-row scales.
        
        Returns:
            (quantized_act_uint8, scales_float32): quantized_act is uint8 [M, K],
                scales is float32 [M]
        """
        if device is not None:
            x = x.to(device=device)
        
        assert self.act_spec.fp8_view_dtype is not None
        assert self.act_spec.fp8_min is not None and self.act_spec.fp8_max is not None
        
        # Ensure input is in a compatible dtype
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            x = x.to(torch.bfloat16)
        
        quantized, scales = _quantize_per_row_fp8(
            x,
            self.act_spec.fp8_view_dtype,
            float(self.act_spec.fp8_min),
            float(self.act_spec.fp8_max),
        )
        return quantized, scales
    
    def linear_forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute Linear output using quantized FP8 weights and activations (W8A8).
        
        Uses Python reference implementation (dequantize both + F.linear).
        Future: can integrate TileLang kernel if available.
        """
        _ = quant_kind
        
        quant_scales = kwargs.pop("quant_scales", None)
        
        # Resolve / cache quantized weight + scales
        if weight.dtype == torch.uint8:
            if quant_scales is None:
                raise ValueError("weight is uint8 (FP8) but quant_scales is None; expected per-channel scales tensor")
            qweight = weight if weight.device == x.device else weight.to(device=x.device)
            w_scales = quant_scales
            # Prefer float16 scales for quality
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
        import os
        if os.getenv("DIFFULEX_FP8_W8A8_PREFER_CUBLAS", "0") == "1":
            deq_key = weight_id
            deq_w = self._dequant_weight_cache.get(deq_key)
            if deq_w is None or deq_w.device != x.device:
                deq_w = self.dequantize(qweight, w_scales)
                self._dequant_weight_cache[deq_key] = deq_w
            # Also dequantize activation
            x_q_temp, x_scales_temp = self.quantize_act_for_kernel(x, device=x.device)
            x_deq = self._dequantize_act(x_q_temp, x_scales_temp)
            return F.linear(x_deq, deq_w, bias)
        
        # Quantize activation per-row
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            x = x.to(torch.bfloat16)
        x_q, x_scales = self.quantize_act_for_kernel(x, device=x.device)
        
        # Try to use TileLang kernel if available
        # For W8A8, weight_dtype and act_dtype should match (both e4m3 or both e5m2)
        fp8_w8a8_gemm = None
        if self.weight_dtype_str == "fp8_e4m3" and self.act_dtype_str == "fp8_e4m3":
            fp8_w8a8_gemm = _fp8_e4m3_w8a8_gemm
        elif self.weight_dtype_str == "fp8_e5m2" and self.act_dtype_str == "fp8_e5m2":
            fp8_w8a8_gemm = _fp8_e5m2_w8a8_gemm
        
        if _TILELANG_AVAILABLE and fp8_w8a8_gemm is not None:
            try:
                # Check device
                if x.device.type != 'cuda':
                    return self._fallback_python_forward(x_q, x_scales, qweight, w_scales, bias)
                
                # Get shapes
                M, K = x_q.shape
                N, K_w = qweight.shape
                assert K == K_w, f"K dimension mismatch: {K} != {K_w}"
                
                # Bucket M to reduce compilation churn
                M_bucket = M
                if M > 1:
                    if M <= 64:
                        M_bucket = 1 << (M - 1).bit_length()
                    else:
                        M_bucket = ((M + 63) // 64) * 64

                x_q_for_kernel = x_q
                if M_bucket != M:
                    x_q_pad = torch.zeros((M_bucket, K), device=x_q.device, dtype=x_q.dtype)
                    x_q_pad[:M, :] = x_q
                    x_q_for_kernel = x_q_pad
                    # Pad scales as well
                    x_scales_pad = torch.zeros((M_bucket,), device=x_scales.device, dtype=x_scales.dtype)
                    x_scales_pad[:M] = x_scales
                    x_scales = x_scales_pad

                # TileLang autotune: use warmup + config cache pattern
                cache_key = (str(x.device), M_bucket, N, K)
                config = self._tl_autotune_config_cache.get(cache_key)
                
                if _AUTOTUNE_AVAILABLE and is_warming_up() and config is None:
                    # Warmup phase: run autotune with real inputs
                    try:
                        assert self.act_spec.fp8_view_dtype is not None
                        assert self.weight_spec.fp8_view_dtype is not None
                        x_fp8 = x_q_for_kernel.view(self.act_spec.fp8_view_dtype)
                        w_fp8 = qweight.view(self.weight_spec.fp8_view_dtype)
                        with set_autotune_inputs([x_fp8, w_fp8, x_scales, w_scales]):
                            kernel = fp8_w8a8_gemm(M_bucket, N, K)
                        config = kernel.config
                        self._tl_autotune_config_cache[cache_key] = config
                    except Exception:
                        # Fallback to default config if autotune fails
                        config = None
                
                # Use cached config or default parameters
                assert self.act_spec.fp8_view_dtype is not None
                assert self.weight_spec.fp8_view_dtype is not None
                x_fp8 = x_q_for_kernel.view(self.act_spec.fp8_view_dtype)
                w_fp8 = qweight.view(self.weight_spec.fp8_view_dtype)
                if config is not None:
                    kernel = fp8_w8a8_gemm(M_bucket, N, K, **config)
                else:
                    # Default config (backward compatible)
                    kernel = fp8_w8a8_gemm(M_bucket, N, K, block_M=64, block_N=64, block_K=128, num_stages=2, threads=128)
                
                # Call kernel - out_idx=[4] means output is the 5th parameter
                # Inputs: A/B are fp8 tensors (viewed from uint8 storage), scales are float32/float16.
                assert self.act_spec.fp8_view_dtype is not None
                assert self.weight_spec.fp8_view_dtype is not None
                x_fp8 = x_q_for_kernel.view(self.act_spec.fp8_view_dtype)
                w_fp8 = qweight.view(self.weight_spec.fp8_view_dtype)
                output_full = kernel(x_fp8, w_fp8, x_scales, w_scales)
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
                elif 'pipeline' in error_msg.lower() and 'stage' in error_msg.lower():
                    # Pipeline stages mismatch - silently fallback
                    pass
                else:
                    # Truncate very long error messages
                    if len(error_msg) > 200:
                        error_msg = error_msg[:200] + "..."
                
                # Only warn for unexpected errors
                if 'CUDA architecture not supported' not in error_msg and 'sm_' not in error_msg and 'Pipeline stages' not in error_msg:
                    warnings.warn(
                        f"TileLang kernel failed, falling back to Python implementation: {error_msg}",
                        UserWarning,
                    )
                return self._fallback_python_forward(x_q, x_scales, qweight, w_scales, bias)
        else:
            # TileLang not available, use Python reference
            return self._fallback_python_forward(x_q, x_scales, qweight, w_scales, bias)
    
    def _fallback_python_forward(
        self,
        x_q: torch.Tensor,
        x_scales: torch.Tensor,
        qweight: torch.Tensor,
        w_scales: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fallback Python implementation: dequantize both + F.linear."""
        # Dequantize both weight and activation
        deq_w = self.dequantize(qweight, w_scales)
        deq_x = self._dequantize_act(x_q, x_scales)
        
        # Compute linear output
        return F.linear(deq_x, deq_w, bias)
    
    def _dequantize_act(
        self,
        quantized: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Dequantize FP8 activation tensor.
        
        Args:
            quantized: uint8 tensor [M, K] (FP8 storage)
            scales: float32 tensor [M] (per-row scales)
        
        Returns:
            Dequantized tensor in bf16 [M, K]
        """
        assert self.act_spec.fp8_view_dtype is not None
        
        # View uint8 as FP8 dtype
        fp8_tensor = quantized.view(self.act_spec.fp8_view_dtype).to(torch.float32)
        
        # Reshape scales to broadcast: [M] -> [M, 1]
        scales_view = scales.to(torch.float32).unsqueeze(-1)  # [M, 1]
        
        # Dequantize: value * scale
        dequantized = fp8_tensor * scales_view
        return dequantized.to(torch.bfloat16)

