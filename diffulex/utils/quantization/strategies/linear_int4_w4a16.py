"""
W4A16 Linear quantization strategy (int4 weight + bf16 activation).

Reference implementation using Python dequantization + torch.nn.functional.linear.
Int4 weights are packed into int8 (2 int4 values per int8 byte).

Future optimizations:
- Replace F.linear with custom Triton/TileLang kernel for int4 GEMM
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
    from diffulex_kernel.python.linear_kernels import w4a16_gemm
    _TILELANG_AVAILABLE = True
except ImportError:
    _TILELANG_AVAILABLE = False
    w4a16_gemm = None

try:
    from diffulex.attention.metadata import is_warming_up
    from tilelang.autotuner import set_autotune_inputs
    _AUTOTUNE_AVAILABLE = True
except ImportError:
    _AUTOTUNE_AVAILABLE = False
    is_warming_up = lambda: False
    set_autotune_inputs = lambda *args, **kwargs: lambda f: f


@register_linear_strategy(weight_dtype="int4", act_dtype="bf16")
def _build_linear_int4_w4a16() -> LinearQuantizationStrategy:
    return LinearInt4W4A16Strategy()


class LinearInt4W4A16Strategy(LinearQuantizationStrategy):
    """W4A16 Linear strategy: int4 weight quantization + bf16 activation.

    Current implementation: Python reference using dequantized weights + F.linear.
    Weight quantization: per-output-channel symmetric quantization to int4.
    Activation: kept as bf16 (no activation quantization).
    
    Int4 packing: Each int8 byte stores 2 int4 values (lower 4 bits and upper 4 bits).
    Packed weight shape: [out_features, (in_features + 1) // 2] (int8)
    
    Lazy cache: Quantized weights are cached per weight tensor (by id) to avoid
    re-quantizing on every forward pass.
    """
    
    def __init__(self):
        """Initialize strategy with empty weight cache."""
        super().__init__()
        # Cache: weight_id -> (packed_weight_int8, scales)
        # Using id(weight) as key since the same Parameter object is reused across forwards
        self._weight_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # Optional cache: weight_id -> bf16 dequantized weight (speed-first; uses extra memory)
        self._dequant_weight_cache: dict[int, torch.Tensor] = {}
        # TileLang autotune config cache: (device, M_bucket, N, K) -> config dict
        self._tl_autotune_config_cache: dict[tuple[str, int, int, int], dict] = {}

    @property
    def name(self) -> str:
        return "linear_int4_w4a16"

    @property
    def linear_weight_format(self) -> str:
        return "int4"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # Weights are stored as int8 (1 byte per element), but each byte contains 2 int4 values
        # So effective storage is 0.5 bytes per int4 weight element
        return torch.int8, 1  # Physical storage is int8, but logical is int4

    @staticmethod
    def _pack_int4_to_int8(int4_tensor: torch.Tensor) -> torch.Tensor:
        """Pack int4 tensor into int8 format.
        
        Args:
            int4_tensor: int8 tensor with values in range [-8, 7] (representing int4)
                        shape: [out_features, in_features]
        
        Returns:
            Packed int8 tensor, shape: [out_features, (in_features + 1) // 2]
            Each int8 byte contains 2 int4 values: lower 4 bits (first) and upper 4 bits (second)
        """
        out_features, in_features = int4_tensor.shape
        
        # Clamp to int4 range [-8, 7]
        int4_tensor = int4_tensor.clamp(-8, 7)
        
        # Convert to uint8 for easier bit manipulation
        # Map [-8, 7] to [0, 15] by adding 8
        uint8_tensor = (int4_tensor + 8).to(torch.uint8)
        
        # Pad in_features to even number if needed
        if in_features % 2 != 0:
            # Pad with zeros (value 8 in uint8, which represents 0 in int4)
            pad_size = 1
            padding = torch.zeros(out_features, pad_size, dtype=torch.uint8, device=uint8_tensor.device) + 8
            uint8_tensor = torch.cat([uint8_tensor, padding], dim=1)
            padded_in_features = in_features + pad_size
        else:
            padded_in_features = in_features
        
        # Reshape to [out_features, in_features // 2, 2]
        reshaped = uint8_tensor.view(out_features, padded_in_features // 2, 2)
        
        # Pack: first element in lower 4 bits, second element in upper 4 bits
        # packed[i, j] = reshaped[i, j, 0] | (reshaped[i, j, 1] << 4)
        packed = reshaped[:, :, 0] | (reshaped[:, :, 1] << 4)
        
        # Convert back to int8
        return packed.to(torch.int8)

    @staticmethod
    def _unpack_int8_to_int4(packed_int8: torch.Tensor, original_in_features: int) -> torch.Tensor:
        """Unpack int8 tensor back to int4 format.
        
        Args:
            packed_int8: Packed int8 tensor, shape: [out_features, packed_size]
            original_in_features: Original in_features dimension (before padding)
        
        Returns:
            Unpacked int4 tensor (as int8 with values in range [-8, 7]), shape: [out_features, original_in_features]
        """
        out_features, packed_size = packed_int8.shape
        
        # Convert to uint8 for bit manipulation
        uint8_packed = packed_int8.to(torch.uint8)
        
        # Extract lower and upper 4 bits
        lower = uint8_packed & 0x0F  # Lower 4 bits
        upper = (uint8_packed >> 4) & 0x0F  # Upper 4 bits
        
        # Stack: [out_features, packed_size, 2]
        unpacked_uint8 = torch.stack([lower, upper], dim=-1)
        
        # Reshape to [out_features, packed_size * 2]
        unpacked_uint8 = unpacked_uint8.view(out_features, packed_size * 2)
        
        # Slice to original size (remove padding if any)
        unpacked_uint8 = unpacked_uint8[:, :original_in_features]
        
        # Convert back to int4 range: [0, 15] -> [-8, 7]
        unpacked_int4 = unpacked_uint8.to(torch.int8) - 8
        
        return unpacked_int4

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        """Quantize tensor to int4 with per-channel (per-output) scales.
        
        Args:
            tensor: Weight tensor of shape [out_features, in_features]
            **kwargs: Additional arguments (unused for now)
        
        Returns:
            (packed_weight_int8, scales): 
            - packed_weight_int8: int8 tensor shape [out_features, (in_features + 1) // 2]
            - scales: [out_features]
        """
        _ = kwargs
        # Per-output-channel quantization: compute scale for each output channel
        # shape: [out_features, in_features] -> scales shape: [out_features]
        abs_max = torch.abs(tensor).max(dim=-1, keepdim=True)[0]  # [out_features, 1]
        # Avoid division by zero
        scales = abs_max.clamp(min=1e-8) / 7.0  # [out_features, 1] (int4 range is -8 to 7, so max abs is 7)
        
        # Quantize: round(clamp(tensor / scales, -8, 7))
        quantized_int4 = torch.round(tensor / scales).clamp(-8, 7).to(torch.int8)
        scales_1d = scales.squeeze(-1)  # [out_features]
        
        # Pack int4 into int8
        packed_weight = self._pack_int4_to_int8(quantized_int4)
        
        return packed_weight, scales_1d

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        """Dequantize packed int4 tensor back to bf16 using per-channel scales.
        
        Args:
            quantized: Packed int8 tensor [out_features, packed_size]
            scale_or_metadata: scales tensor [out_features] or dict with 'scales' and 'original_in_features'
            **kwargs: Additional arguments, may include 'original_in_features'
        
        Returns:
            Dequantized tensor in bf16, shape [out_features, original_in_features]
        """
        _ = kwargs
        if isinstance(scale_or_metadata, dict):
            scales = scale_or_metadata.get("scales")
            original_in_features = scale_or_metadata.get("original_in_features")
        else:
            scales = scale_or_metadata
            # Try to infer original_in_features from quantized shape
            # packed_size = (in_features + 1) // 2, so in_features = packed_size * 2 or packed_size * 2 - 1
            packed_size = quantized.shape[1]
            # We'll use the maximum possible (packed_size * 2), caller should provide original_in_features if needed
            original_in_features = packed_size * 2
        
        if scales is None:
            raise ValueError("scales required for dequantization")
        
        # Get original_in_features from kwargs if provided
        original_in_features = kwargs.get("original_in_features", original_in_features)
        
        # Unpack int4 from int8
        unpacked_int4 = self._unpack_int8_to_int4(quantized, original_in_features)
        
        # Ensure scales have correct shape for broadcasting
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)  # [out_features, 1]
        
        # Dequantize: quantized * scales
        dequantized = unpacked_int4.to(torch.float32) * scales
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
        """Quantize weight to int4 (packed as int8) with per-channel scales.
        
        Returns:
            (packed_weight_int8, scales): 
            - packed_weight_int8: int8 [out, (in + 1) // 2]
            - scales: [out]
        """
        _ = kwargs
        if device is not None:
            weight = weight.to(device=device)
        
        packed_weight, scales = self.quantize(weight)
        return packed_weight, scales

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
        """Compute Linear output using quantized weights (W4A16).
        
        Uses Python reference implementation (dequant + F.linear).
        Future: Replace with TileLang kernel for int4 GEMM.
        
        Args:
            x: Activation tensor [M, K] (bf16)
            weight: Either bf16 weight [N, K] or packed int8 weight [N, (K + 1) // 2]
            bias: Optional bias tensor [N]
            quant_kind: Quantization kind (unused)
            **kwargs: May include quant_scales and original_in_features for load-time quantized weights
        """
        _ = quant_kind

        # If caller provides a pre-quantized packed int8 weight + scales (e.g., load-time quantized module),
        # use them directly and DO NOT populate the lazy cache (to avoid double-storage).
        quant_scales = kwargs.pop("quant_scales", None)
        original_in_features = kwargs.pop("original_in_features", None)
        
        if weight.dtype == torch.int8:
            if quant_scales is None:
                raise ValueError("weight is int8 (packed int4) but quant_scales is None; expected per-channel scales tensor")
            # We have activation K; that's the real in_features for this matmul.
            # Using packed_size*2 is fragile (it breaks if the int4 weights are stored "unpacked" as int8[N, K]).
            M, K = x.shape
            if original_in_features is None:
                original_in_features = K

            # Accept both representations:
            # - packed int4: int8[N, (K+1)//2] where each byte holds 2 int4
            # - unpacked int4: int8[N, K] where each element is an int4 value stored in int8
            expected_packed_K = (K + 1) // 2
            if weight.shape[1] == expected_packed_K:
                packed_weight = weight
            elif weight.shape[1] == K:
                # Unpacked int4 -> pack on-the-fly so we can use the same kernel path.
                # Support both [-8, 7] (signed int4) and [0, 15] (uint4 stored in int8).
                w = weight
                if (w.min() >= 0) and (w.max() <= 15):
                    w = (w.to(torch.int16) - 8).to(torch.int8)
                packed_weight = self._pack_int4_to_int8(w)
            else:
                raise ValueError(
                    f"Unexpected int4 weight shape for int8 weight: got {tuple(weight.shape)}, "
                    f"expected (N,{expected_packed_K}) for packed or (N,{K}) for unpacked."
                )
            scales = quant_scales
            if scales.dtype != torch.bfloat16:
                scales = scales.to(dtype=torch.bfloat16)
            if packed_weight.device != x.device:
                packed_weight = packed_weight.to(device=x.device)
            if scales.device != x.device:
                scales = scales.to(device=x.device)
        else:
            # Lazy cache: use weight tensor id as key (only for bf16/fp16 weights)
            weight_id = id(weight)

            # Check cache
            if weight_id in self._weight_cache:
                packed_weight, scales = self._weight_cache[weight_id]
                # Ensure cached tensors are on the correct device
                if packed_weight.device != x.device:
                    packed_weight = packed_weight.to(device=x.device)
                    scales = scales.to(device=x.device)
                # Get original_in_features from cached metadata or infer
                if original_in_features is None:
                    # Infer: packed_size = (in_features + 1) // 2
                    packed_size = packed_weight.shape[1]
                    original_in_features = packed_size * 2
            else:
                # Quantize weight and cache it
                packed_weight, scales = self.quantize_weight_for_kernel(weight, device=x.device)
                # Cache the packed weight and scales
                self._weight_cache[weight_id] = (packed_weight, scales)
                # Store original_in_features for later use
                original_in_features = weight.shape[1]

        # Speed-first option:
        # If enabled, dequantize once and reuse a cached bf16 weight for F.linear (cuBLAS).
        # This trades extra GPU memory for throughput.
        if os.getenv("DIFFULEX_W4A16_PREFER_CUBLAS", "0") == "1":
            deq_key = id(weight)
            deq_w = self._dequant_weight_cache.get(deq_key)
            if deq_w is None or deq_w.device != x.device:
                deq_w = self.dequantize(
                    packed_weight,
                    scales,
                    original_in_features=original_in_features,
                )
                if deq_w.device != x.device:
                    deq_w = deq_w.to(device=x.device)
                self._dequant_weight_cache[deq_key] = deq_w
            return F.linear(x, deq_w, bias)
        
        # Try to use TileLang kernel if available
        if _TILELANG_AVAILABLE and w4a16_gemm is not None:
            try:
                # Check device
                if x.device.type != 'cuda':
                    return self._fallback_python_forward(x, packed_weight, scales, bias, original_in_features=original_in_features)
                
                # Check CUDA compute capability (skip kernel if unsupported)
                try:
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(x.device.index or 0)
                        compute_cap = (props.major, props.minor)
                        # Let TileLang handle the check and fallback gracefully
                        pass
                except Exception:
                    # If we can't check compute capability, still try the kernel
                    pass
                
                # Get shapes
                M, K = x.shape
                N, packed_K = packed_weight.shape
                # Verify packed_K matches expected packed size for K
                expected_packed_K = (original_in_features + 1) // 2
                assert packed_K == expected_packed_K, f"Packed K dimension mismatch: {packed_K} != {expected_packed_K}"
                
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
                        with set_autotune_inputs([x_for_kernel, packed_weight, scales]):
                            kernel = w4a16_gemm(M_bucket, N, K)
                        config = kernel.config
                        self._tl_autotune_config_cache[cache_key] = config
                    except Exception:
                        # Fallback to default config if autotune fails
                        config = None
                
                # Use cached config or default parameters
                if config is not None:
                    kernel = w4a16_gemm(M_bucket, N, K, **config)
                else:
                    # Default config (backward compatible)
                    kernel = w4a16_gemm(M_bucket, N, K, block_M=64, block_N=64, block_K=128, num_stages=2, threads=128)
                
                # Call kernel - out_idx=[3] means output is the 4th parameter,
                # so we only pass inputs (x, packed_weight, scales), and kernel returns output
                output_full = kernel(x_for_kernel, packed_weight, scales)
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
                        f"TileLang W4A16 kernel failed, falling back to Python implementation: {error_msg}",
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
                        f"TileLang W4A16 kernel failed, falling back to Python implementation: {error_msg}",
                        UserWarning,
                    )
                return self._fallback_python_forward(x, packed_weight, scales, bias, original_in_features=original_in_features)
        else:
            # TileLang not available, use Python reference
            return self._fallback_python_forward(x, packed_weight, scales, bias, original_in_features=original_in_features)
    
    def _fallback_python_forward(
        self,
        x: torch.Tensor,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        original_in_features: int,
    ) -> torch.Tensor:
        """Fallback Python implementation: unpack + dequantize + F.linear."""
        # Unpack and dequantize
        dequantized_weight = self.dequantize(
            packed_weight, 
            scales, 
            original_in_features=original_in_features
        )
        
        # Compute linear output
        return F.linear(x, dequantized_weight, bias)
    
    def clear_cache(self) -> None:
        """Clear the weight quantization cache.
        
        Useful for memory management or when weights are updated (e.g., fine-tuning).
        """
        self._weight_cache.clear()
        self._dequant_weight_cache.clear()

