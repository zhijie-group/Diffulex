"""
FP8 W8A16 Linear quantization strategy (FP8 weight + bf16 activation).

Implementation notes:
- Weight quantization: per-output-channel FP8 quantization (fp8_e4m3 or fp8_e5m2)
- Activation: kept as bf16 (no activation quantization)
- Storage: FP8 weights use uint8 storage + view(fp8_dtype) pattern
- Scale management: per-channel weight scales (shape: [out_features]), dtype: float32
- Forward path: Python fallback (dequantize FP8 weight → bf16, then F.linear)
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
_fp8_e4m3_w8a16_gemm = None
_fp8_e5m2_w8a16_gemm = None

try:
    from diffulex_kernel.python.linear_kernels import (
        fp8_e4m3_w8a16_gemm,
        fp8_e5m2_w8a16_gemm,
    )
    _TILELANG_AVAILABLE = True
    _fp8_e4m3_w8a16_gemm = fp8_e4m3_w8a16_gemm
    _fp8_e5m2_w8a16_gemm = fp8_e5m2_w8a16_gemm
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


@register_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
def _build_linear_fp8_e4m3_w8a16() -> LinearQuantizationStrategy:
    return LinearFP8W8A16Strategy("fp8_e4m3")


@register_linear_strategy(weight_dtype="fp8_e5m2", act_dtype="bf16")
def _build_linear_fp8_e5m2_w8a16() -> LinearQuantizationStrategy:
    return LinearFP8W8A16Strategy("fp8_e5m2")


class LinearFP8W8A16Strategy(LinearQuantizationStrategy):
    """FP8 W8A16 Linear strategy: FP8 weight quantization + bf16 activation.
    
    Current implementation: Python reference using dequantized weights + F.linear.
    Weight quantization: per-output-channel FP8 quantization (fp8_e4m3 or fp8_e5m2).
    Activation: kept as bf16 (no activation quantization).
    
    Lazy cache: Quantized weights are cached per weight tensor (by id) to avoid
    re-quantizing on every forward pass.
    """
    
    def __init__(self, weight_dtype: str = "fp8_e4m3"):
        """
        Initialize FP8 W8A16 strategy.
        
        Args:
            weight_dtype: FP8 dtype string ("fp8_e4m3" or "fp8_e5m2")
        """
        super().__init__()
        self.weight_dtype_str = weight_dtype
        self.spec = parse_kv_cache_dtype(weight_dtype)
        if not self.spec.is_fp8:
            raise ValueError(f"Expected FP8 dtype, got {weight_dtype}")
        
        # Cache: weight_id -> (quantized_weight_uint8, scales_float32)
        # Using id(weight) as key since the same Parameter object is reused across forwards
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Optional cache: weight_id -> bf16 dequantized weight (speed-first; uses extra memory)
        self._dequant_weight_cache: dict[int, torch.Tensor] = {}
        # TileLang autotune config cache: (device, M_bucket, N, K) -> config dict
        self._tl_autotune_config_cache: dict[tuple[str, int, int, int], dict] = {}
    
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
        # FP8 weights are stored as uint8 (1 byte per element)
        return torch.uint8, 1
    
    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """Quantize tensor to FP8 with per-channel (per-output) scales.
        
        Args:
            tensor: Weight tensor of shape [out_features, in_features]
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            (quantized_tensor_uint8, scales_float32): quantized_tensor is uint8 (FP8 storage),
                scales is [out_features]
        """
        _ = kwargs
        assert self.spec.fp8_view_dtype is not None
        assert self.spec.fp8_min is not None and self.spec.fp8_max is not None
        
        # Per-output-channel quantization: compute scale for each output channel
        # shape: [out_features, in_features] -> scales shape: [out_features]
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [out_features, 1]
        eps = 1e-8
        fp8_max = float(self.spec.fp8_max)
        
        # Compute scales: abs_max / fp8_max
        scales = (abs_max.clamp(min=eps) / fp8_max).to(torch.float32)  # [out_features, 1]
        
        # Quantize: clamp(tensor / scale, fp8_min, fp8_max).to(fp8_dtype).view(uint8)
        descale = 1.0 / scales  # [out_features, 1]
        quantized = (tensor.to(torch.float32) * descale).clamp(
            min=float(self.spec.fp8_min),
            max=float(self.spec.fp8_max)
        )
        quantized_fp8 = quantized.to(self.spec.fp8_view_dtype)
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
        assert self.spec.fp8_view_dtype is not None
        
        if isinstance(scale_or_metadata, dict):
            scales = scale_or_metadata.get("scales")
        else:
            scales = scale_or_metadata
        
        if scales is None:
            raise ValueError("scales required for dequantization")
        
        # View uint8 as FP8 dtype
        fp8_tensor = quantized.view(self.spec.fp8_view_dtype).to(torch.float32)
        
        # Ensure scales have correct shape for broadcasting
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)  # [out_features, 1]
        
        # Dequantize: quantized * scales
        dequantized = fp8_tensor * scales.to(torch.float32)
        return dequantized.to(torch.bfloat16)
    
    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        """Return shape of scales tensor for per-channel quantization.
        
        For [out_features, in_features] weight, scales shape is [out_features].
        """
        _ = kwargs
        if len(original_shape) < 2:
            raise ValueError(f"Expected weight shape with at least 2 dims, got {original_shape}")
        # Per-output-channel: scales shape is [out_features]
        return (original_shape[0],)
    
    def quantize_weight_for_kernel(
        self,
        weight: torch.Tensor,
        *,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Quantize weight to FP8 with per-channel scales.
        
        Returns:
            (quantized_weight_uint8, scales_float32): quantized_weight is uint8 [out, in],
                scales is float32 [out]
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
        """No activation quantization for W8A16 (activation stays bf16)."""
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
        """Compute Linear output using quantized FP8 weights (W8A16).
        
        Uses Python reference implementation (dequant + F.linear).
        Future: can integrate TileLang kernel if available.
        """
        _ = quant_kind
        
        # If caller provides a pre-quantized uint8 weight + scales (e.g., load-time quantized module),
        # use them directly and DO NOT populate the lazy cache (to avoid double-storage).
        quant_scales = kwargs.pop("quant_scales", None)
        if weight.dtype == torch.uint8:
            if quant_scales is None:
                raise ValueError("weight is uint8 (FP8) but quant_scales is None; expected per-channel scales tensor")
            quantized_weight = weight
            scales = quant_scales
            if scales.dtype != torch.float32:
                scales = scales.to(dtype=torch.float32)
            if quantized_weight.device != x.device:
                quantized_weight = quantized_weight.to(device=x.device)
            if scales.device != x.device:
                scales = scales.to(device=x.device)
        else:
            # Lazy cache: use weight tensor id as key (only for bf16/fp16/fp32 weights)
            weight_id = id(weight)
            
            # Check cache
            if weight_id in self._weight_cache:
                quantized_weight, scales = self._weight_cache[weight_id]
                # Ensure cached tensors are on the correct device
                if quantized_weight.device != x.device:
                    quantized_weight = quantized_weight.to(device=x.device)
                    scales = scales.to(device=x.device)
            else:
                # Quantize weight and cache it
                quantized_weight, scales = self.quantize_weight_for_kernel(weight, device=x.device)
                # Cache the quantized weight and scales
                self._weight_cache[weight_id] = (quantized_weight, scales)
        
        # Speed-first option: cache dequantized bf16 weight for F.linear (cuBLAS)
        # This trades extra GPU memory for throughput.
        import os
        if os.getenv("DIFFULEX_FP8_W8A16_PREFER_CUBLAS", "0") == "1":
            deq_key = id(weight) if weight.dtype != torch.uint8 else id(quantized_weight)
            deq_w = self._dequant_weight_cache.get(deq_key)
            if deq_w is None or deq_w.device != x.device:
                # Dequantize: FP8[N,K] * scales[N] -> bf16[N,K]
                deq_w = self.dequantize(quantized_weight, scales)
                self._dequant_weight_cache[deq_key] = deq_w
            return F.linear(x, deq_w, bias)
        
        # Try to use TileLang kernel if available
        fp8_w8a16_gemm = None
        if self.weight_dtype_str == "fp8_e4m3":
            fp8_w8a16_gemm = _fp8_e4m3_w8a16_gemm
        elif self.weight_dtype_str == "fp8_e5m2":
            fp8_w8a16_gemm = _fp8_e5m2_w8a16_gemm
        
        if _TILELANG_AVAILABLE and fp8_w8a16_gemm is not None:
            try:
                # Check device
                if x.device.type != 'cuda':
                    return self._fallback_python_forward(x, quantized_weight, scales, bias)
                
                # Get shapes
                M, K = x.shape
                N, K_w = quantized_weight.shape
                assert K == K_w, f"K dimension mismatch: {K} != {K_w}"
                
                # Bucket M to reduce compilation churn
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
                cache_key = (str(x.device), M_bucket, N, K)
                config = self._tl_autotune_config_cache.get(cache_key)
                
                if _AUTOTUNE_AVAILABLE and is_warming_up() and config is None:
                    # Warmup phase: run autotune with real inputs
                    try:
                        assert self.spec.fp8_view_dtype is not None
                        qweight_fp8 = quantized_weight.view(self.spec.fp8_view_dtype)
                        with set_autotune_inputs([x_for_kernel, qweight_fp8, scales]):
                            kernel = fp8_w8a16_gemm(M_bucket, N, K)
                        config = kernel.config
                        self._tl_autotune_config_cache[cache_key] = config
                    except Exception:
                        # Fallback to default config if autotune fails
                        config = None
                
                # Use cached config or default parameters
                assert self.spec.fp8_view_dtype is not None
                qweight_fp8 = quantized_weight.view(self.spec.fp8_view_dtype)
                if config is not None:
                    kernel = fp8_w8a16_gemm(M_bucket, N, K, **config)
                else:
                    # Default config (backward compatible)
                    kernel = fp8_w8a16_gemm(M_bucket, N, K, block_M=64, block_N=64, block_K=128, num_stages=2, threads=128)
                
                # Call kernel - out_idx=[3] means output is the 4th parameter
                assert self.spec.fp8_view_dtype is not None
                qweight_fp8 = quantized_weight.view(self.spec.fp8_view_dtype)
                output_full = kernel(x_for_kernel, qweight_fp8, scales)
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
                return self._fallback_python_forward(x, quantized_weight, scales, bias)
        else:
            # TileLang not available, use Python reference
            return self._fallback_python_forward(x, quantized_weight, scales, bias)
    
    def _fallback_python_forward(
        self,
        x: torch.Tensor,
        quantized_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fallback Python implementation: dequantize + F.linear."""
        # Dequantize for reference implementation
        dequantized_weight = self.dequantize(quantized_weight, scales)
        
        # Compute linear output
        return F.linear(x, dequantized_weight, bias)
    
    def clear_cache(self) -> None:
        """Clear the weight quantization cache.
        
        Useful for memory management or when weights are updated (e.g., fine-tuning).
        """
        self._weight_cache.clear()
        self._dequant_weight_cache.clear()

