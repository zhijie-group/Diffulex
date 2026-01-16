"""
W8A16 Linear quantization strategy (int8 weight + bf16 activation).

Reference implementation using Python dequantization + torch.nn.functional.linear.
Future optimizations:
- Lazy cache quantized weights per module instance
- Replace F.linear with custom Triton/TileLang kernel for int8 GEMM
"""

from __future__ import annotations

from typing import Any, Optional

import os
import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

# Try to import TileLang kernel, fallback to None if not available
try:
    from diffulex_kernel.python.linear_kernels import w8a16_gemm
    _TILELANG_AVAILABLE = True
except ImportError:
    _TILELANG_AVAILABLE = False
    w8a16_gemm = None

try:
    from diffulex_kernel.python.linear_kernels import w8a16_gemm_bias
except ImportError:
    w8a16_gemm_bias = None

try:
    from diffulex.attention.metadata import is_warming_up
    from tilelang.autotuner import set_autotune_inputs
    _AUTOTUNE_AVAILABLE = True
except ImportError:
    _AUTOTUNE_AVAILABLE = False
    is_warming_up = lambda: False
    set_autotune_inputs = lambda *args, **kwargs: lambda f: f


@register_linear_strategy(weight_dtype="int8", act_dtype="bf16")
def _build_linear_int8_w8a16() -> LinearQuantizationStrategy:
    return LinearInt8W8A16Strategy()


class LinearInt8W8A16Strategy(LinearQuantizationStrategy):
    """W8A16 Linear strategy: int8 weight quantization + bf16 activation.

    Current implementation: Python reference using dequantized weights + F.linear.
    Weight quantization: per-output-channel symmetric quantization to int8.
    Activation: kept as bf16 (no activation quantization).
    
    Lazy cache: Quantized weights are cached per weight tensor (by id) to avoid
    re-quantizing on every forward pass.
    """
    
    def __init__(self):
        """Initialize strategy with empty weight cache."""
        super().__init__()
        # Cache: weight_id -> (quantized_weight, scales)
        # Using id(weight) as key since the same Parameter object is reused across forwards
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Optional cache: weight_id -> bf16 dequantized weight (speed-first; uses extra memory)
        self._dequant_weight_cache: dict[int, torch.Tensor] = {}
        # bias cache for fused-bias kernel (store fp16 copy on device)
        self._bias_f16_cache: dict[int, torch.Tensor] = {}
        # TileLang autotune config cache: (device, M_bucket, N, K) -> config dict
        self._tl_autotune_config_cache: dict[tuple[str, int, int, int], dict] = {}
        # Lightweight runtime observability (opt-in by env var)
        self._rt_call_count: int = 0
        self._rt_fallback_count: int = 0
        self._rt_m_hist_le64: dict[int, int] = {}

    def _rt_enabled(self) -> bool:
        return os.getenv("DIFFULEX_LINEAR_PROFILE", "0") == "1"

    def _rt_log_every(self) -> int:
        try:
            return int(os.getenv("DIFFULEX_LINEAR_PROFILE_EVERY", "200"))
        except Exception:
            return 200

    def _rt_on_call(self, *, m: int, n: int, k: int) -> None:
        if not self._rt_enabled():
            return
        self._rt_call_count += 1
        if m <= 64:
            self._rt_m_hist_le64[m] = self._rt_m_hist_le64.get(m, 0) + 1
        every = self._rt_log_every()
        if every > 0 and (self._rt_call_count % every == 0):
            top = sorted(self._rt_m_hist_le64.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
            top_str = ", ".join([f"M={mm}:{cc}" for mm, cc in top]) if top else "empty"
            print(
                f"[DIFFULEX_LINEAR_PROFILE][w8a16] calls={self._rt_call_count} "
                f"fallbacks={self._rt_fallback_count} last(M,N,K)=({m},{n},{k}) "
                f"M_hist_le64_top={top_str}",
                flush=True,
            )

    def _rt_on_fallback(self, *, m: int, n: int, k: int, reason: str) -> None:
        if not self._rt_enabled():
            return
        self._rt_fallback_count += 1
        # Avoid spam: only print first few fallbacks, then rely on periodic summary.
        max_print = 5
        try:
            max_print = int(os.getenv("DIFFULEX_LINEAR_FALLBACK_MAX_PRINT", "5"))
        except Exception:
            pass
        if self._rt_fallback_count <= max_print:
            print(
                f"[DIFFULEX_LINEAR_PROFILE][w8a16][FALLBACK] "
                f"count={self._rt_fallback_count} (M,N,K)=({m},{n},{k}) reason={reason}",
                flush=True,
            )

    @property
    def name(self) -> str:
        return "linear_int8_w8a16"

    @property
    def linear_weight_format(self) -> str:
        return "int8"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # Weights are stored as int8 (1 byte per element)
        return torch.int8, 1

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """Quantize tensor to int8 with per-channel (per-output) scales.
        
        Args:
            tensor: Weight tensor of shape [out_features, in_features]
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            (quantized_tensor, scales): quantized_tensor is int8, scales is [out_features]
        """
        _ = kwargs
        # Per-output-channel quantization: compute scale for each output channel
        # shape: [out_features, in_features] -> scales shape: [out_features]
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [out_features, 1]
        # Avoid division by zero
        scales = abs_max.clamp(min=1e-8) / 127.0  # [out_features, 1]
        
        # Quantize: round(clamp(tensor / scales, -128, 127))
        quantized = torch.round(tensor / scales).clamp(-128, 127).to(torch.int8)
        scales_1d = scales.squeeze(-1)  # [out_features]
        
        return quantized, scales_1d

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """Dequantize int8 tensor back to bf16 using per-channel scales.
        
        Args:
            quantized: int8 tensor [out_features, in_features]
            scale_or_metadata: scales tensor [out_features] or dict with 'scales'
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            Dequantized tensor in bf16
        """
        _ = kwargs
        if isinstance(scale_or_metadata, dict):
            scales = scale_or_metadata.get("scales")
        else:
            scales = scale_or_metadata
        
        if scales is None:
            raise ValueError("scales required for dequantization")
        
        # Ensure scales have correct shape for broadcasting
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)  # [out_features, 1]
        
        # Dequantize: quantized * scales
        dequantized = quantized.to(torch.float32) * scales
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
        """Quantize weight to int8 with per-channel scales.
        
        Returns:
            (quantized_weight, scales): quantized_weight is int8 [out, in], scales is [out]
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
        """Compute Linear output using quantized weights (W8A16).
        
        Uses TileLang kernel if available and conditions are met, otherwise falls back
        to Python reference implementation (dequant + F.linear).
        
        Conditions for using TileLang kernel:
        - TileLang is available
        - Device is CUDA
        - (Kernel supports tail sizes; no K%128 constraint required)
        """
        _ = quant_kind

        # If caller provides a pre-quantized int8 weight + scales (e.g., load-time quantized module),
        # use them directly and DO NOT populate the lazy cache (to avoid double-storage).
        quant_scales = kwargs.pop("quant_scales", None)
        if weight.dtype == torch.int8:
            if quant_scales is None:
                raise ValueError("weight is int8 but quant_scales is None; expected per-channel scales tensor")
            quantized_weight = weight
            scales = quant_scales
            if scales.dtype != torch.bfloat16:
                scales = scales.to(dtype=torch.bfloat16)
            if quantized_weight.device != x.device:
                quantized_weight = quantized_weight.to(device=x.device)
            if scales.device != x.device:
                scales = scales.to(device=x.device)
        else:
            # Lazy cache: use weight tensor id as key (only for bf16/fp16 weights)
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
        
        # Speed-first option:
        # Using the TileLang kernel can be slower than cuBLAS BF16 GEMM for small/typical decode shapes.
        # If enabled, we dequantize once and reuse a cached bf16 weight for F.linear (cuBLAS).
        # This trades extra GPU memory for throughput.
        if os.getenv("DIFFULEX_W8A16_PREFER_CUBLAS", "0") == "1":
            # Key by the actual weight object we received (bf16 Parameter or int8 buffer).
            deq_key = id(weight)
            deq_w = self._dequant_weight_cache.get(deq_key)
            if deq_w is None or deq_w.device != x.device:
                # Dequantize: int8[N,K] * scales[N] -> bf16[N,K]
                s = scales
                if s.dim() == 1:
                    s = s.unsqueeze(-1)
                deq_w = (quantized_weight.to(torch.float32) * s.to(torch.float32)).to(torch.bfloat16)
                self._dequant_weight_cache[deq_key] = deq_w
            return F.linear(x, deq_w, bias)
        
        # Try to use TileLang kernel if available
        if _TILELANG_AVAILABLE and w8a16_gemm is not None:
            try:
                # Check device
                if x.device.type != 'cuda':
                    return self._fallback_python_forward(x, quantized_weight, scales, bias)
                
                # Check CUDA compute capability (skip kernel if unsupported)
                # sm_89 (Hopper) requires CUDA 11.8+, sm_90+ requires CUDA 12.0+
                # If CUDA toolkit doesn't support the GPU architecture, skip kernel attempt
                try:
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(x.device.index or 0)
                        compute_cap = (props.major, props.minor)
                        # sm_89 requires CUDA 11.8+, sm_90+ requires CUDA 12.0+
                        # For now, we'll let TileLang handle the check and fallback gracefully
                        # This is a conservative approach - we try the kernel and let it fail gracefully
                        pass
                except Exception:
                    # If we can't check compute capability, still try the kernel
                    pass
                
                # Get shapes
                M, K = x.shape
                N, K_w = quantized_weight.shape
                assert K == K_w, f"K dimension mismatch: {K} != {K_w}"
                self._rt_on_call(m=M, n=N, k=K)
                
                # Reduce TileLang JIT compilation churn without killing small-M decode performance.
                # Previous logic padded *any* M!=1 to 64/128/256, which can turn decode M=2/4 into M=64.
                # We instead bucket to a small stable set:
                # - for M<=64: next power-of-two (2,4,8,16,32,64)
                # - for M>64: round up to a multiple of 64
                M_bucket = M
                if M > 1:
                    if M <= 64:
                        M_bucket = 1 << (M - 1).bit_length()
                    else:
                        M_bucket = ((M + 63) // 64) * 64
                else:
                    M_bucket = 1

                # TileLang MMA GEMM requires M divisible by 16.
                # For decode small-M (1/2/4/8), pad minimally to 16 (much cheaper than padding to 64).
                if M_bucket < 16:
                    M_bucket = 16

                x_for_kernel = x
                if M_bucket != M:
                    x_pad = torch.zeros((M_bucket, K), device=x.device, dtype=x.dtype)
                    x_pad[:M, :] = x
                    x_for_kernel = x_pad

                # Choose a small-M friendly block_M to reduce wasted work in decode.
                # Keep variants bounded to avoid compilation churn and satisfy MMA constraints:
                # use only {16, 32, 64} so M is always divisible by 16.
                if M_bucket <= 16:
                    block_m = 16
                elif M_bucket <= 32:
                    block_m = 32
                else:
                    block_m = 64

                # TileLang autotune: use warmup + config cache pattern
                # NOTE: fused-bias kernel currently regresses decode throughput significantly on typical workloads.
                # Keep it disabled by default; can be enabled for experimentation.
                fuse_bias = os.getenv("DIFFULEX_W8A16_FUSE_BIAS", "0") == "1"
                use_bias_kernel = fuse_bias and (bias is not None) and (w8a16_gemm_bias is not None)
                
                cache_key = (str(x.device), M_bucket, N, K)
                config = self._tl_autotune_config_cache.get(cache_key)
                
                if _AUTOTUNE_AVAILABLE and is_warming_up() and config is None:
                    # Warmup phase: run autotune with real inputs
                    try:
                        if use_bias_kernel:
                            b_key = id(bias)
                            b = self._bias_f16_cache.get(b_key)
                            if b is None or b.device != x.device:
                                b = bias.to(device=x.device, dtype=torch.float16)
                                self._bias_f16_cache[b_key] = b
                            with set_autotune_inputs([x_for_kernel, quantized_weight, scales, b]):
                                kernel = w8a16_gemm_bias(M_bucket, N, K)
                        else:
                            with set_autotune_inputs([x_for_kernel, quantized_weight, scales]):
                                kernel = w8a16_gemm(M_bucket, N, K)
                        config = kernel.config
                        self._tl_autotune_config_cache[cache_key] = config
                    except Exception:
                        # Fallback to default config if autotune fails
                        config = None
                
                # Use cached config or default parameters
                if config is not None:
                    if use_bias_kernel:
                        kernel = w8a16_gemm_bias(M_bucket, N, K, **config)
                    else:
                        kernel = w8a16_gemm(M_bucket, N, K, **config)
                else:
                    # Default config (backward compatible)
                    if use_bias_kernel:
                        kernel = w8a16_gemm_bias(
                            M_bucket,
                            N,
                            K,
                            block_M=block_m,
                            block_N=64,
                            block_K=128,
                            num_stages=2,
                            threads=128,
                        )
                    else:
                        kernel = w8a16_gemm(
                            M_bucket,
                            N,
                            K,
                            block_M=block_m,
                            block_N=64,
                            block_K=128,
                            num_stages=2,
                            threads=128,
                        )
                
                # Call kernel - out_idx=[3] means output is the 4th parameter,
                # so we only pass inputs (x, quantized_weight, scales), and kernel returns output
                tag_kernel = os.getenv("DIFFULEX_PROFILE_TAG_W8A16", "0") == "1"
                tag_name = (
                    f"{'w8a16_gemm_bias' if use_bias_kernel else 'w8a16_gemm'}"
                    f"[M={M} Mb={M_bucket} N={N} K={K} bm={block_m} bn=64 bk=128 st=2 th=128]"
                )
                if use_bias_kernel:
                    # out_idx=[4] -> output is 5th arg (returned). Inputs: A, B, Scales, Bias
                    # NOTE: kernel expects fp16 bias (see kernel signature).
                    b_key = id(bias)
                    b = self._bias_f16_cache.get(b_key)
                    if b is None or b.device != x.device:
                        b = bias.to(device=x.device, dtype=torch.float16)
                        self._bias_f16_cache[b_key] = b
                    if tag_kernel:
                        with torch.profiler.record_function(tag_name):
                            output_full = kernel(x_for_kernel, quantized_weight, scales, b)
                    else:
                        output_full = kernel(x_for_kernel, quantized_weight, scales, b)
                else:
                    if tag_kernel:
                        with torch.profiler.record_function(tag_name):
                            output_full = kernel(x_for_kernel, quantized_weight, scales)
                    else:
                        output_full = kernel(x_for_kernel, quantized_weight, scales)
                output = output_full[:M, :] if M_bucket != M else output_full
                
                # Add bias if present
                if (bias is not None) and (not use_bias_kernel):
                    output = output + bias
                
                return output
            except Exception as e:
                # Fallback to Python implementation on any error
                # This includes kernel compilation errors, execution errors, etc.
                import warnings
                error_msg = str(e)
                
                # Extract meaningful error information
                # Check for common error types
                if 'sm_' in error_msg and ('not defined' in error_msg or 'fatal' in error_msg):
                    # CUDA architecture not supported
                    import re
                    arch_match = re.search(r"sm_(\d+)", error_msg)
                    if arch_match:
                        arch = arch_match.group(1)
                        error_msg = f"CUDA architecture sm_{arch} not supported by current CUDA toolkit"
                    else:
                        error_msg = "CUDA architecture not supported by current CUDA toolkit"
                elif 'Compilation error' in error_msg:
                    # Extract the actual error after "Compilation error:"
                    idx = error_msg.find('Compilation error')
                    after = error_msg[idx + len('Compilation error'):]
                    # Find the first meaningful error line
                    lines = after.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and ('error:' in line.lower() or 'fatal' in line.lower()):
                            error_msg = f"CUDA compilation error: {line[:200]}"
                            break
                    else:
                        error_msg = "CUDA compilation error (see logs for details)"
                elif 'pipeline' in error_msg.lower() and 'stage' in error_msg.lower():
                    # Pipeline stages mismatch
                    import re
                    match = re.search(r'Got (\d+) stages and (\d+) pipeline stages', error_msg)
                    if match:
                        error_msg = f"Pipeline stages mismatch: detected {match.group(1)} stages, expected {match.group(2)}"
                    else:
                        error_msg = "Pipeline stages configuration error"
                else:
                    # Truncate very long error messages (like CUDA source code)
                    if len(error_msg) > 200:
                        error_msg = error_msg[:200] + "..."
                
                # Only warn for unexpected errors
                # For known issues (like unsupported CUDA architecture), silently fallback
                # This prevents spam warnings when the environment doesn't support the kernel
                if 'CUDA architecture not supported' in error_msg or 'sm_' in error_msg:
                    # Silently fallback for unsupported architectures (expected in some environments)
                    # The Python fallback is fully functional, so this is acceptable
                    pass
                elif 'Pipeline stages' in error_msg:
                    # Pipeline stages mismatch - this might be fixable, but for now silently fallback
                    pass
                else:
                    # Warn for unexpected errors that might indicate a real problem
                    warnings.warn(
                        f"TileLang kernel failed, falling back to Python implementation: {error_msg}",
                        UserWarning,
                    )
                # Count fallback and expose reason (opt-in).
                try:
                    m, k = x.shape
                    n = int(quantized_weight.shape[0])
                except Exception:
                    m, n, k = -1, -1, -1
                self._rt_on_fallback(m=m, n=n, k=k, reason=error_msg)
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

