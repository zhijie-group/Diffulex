"""
Unit tests for FP8 Linear quantization strategies.
"""

import pytest
import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import create_linear_strategy
from diffulex.utils.quantization.context import get_quantization_context


def test_linear_strategy_registry_fp8_e4m3_w8a16():
    """Test that fp8_e4m3+bf16 returns the real FP8 W8A16 strategy."""
    s = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    assert s.name == "linear_fp8_fp8_e4m3_w8a16"
    assert s.linear_weight_format == "fp8_e4m3"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_fp8_e5m2_w8a16():
    """Test that fp8_e5m2+bf16 returns the real FP8 W8A16 strategy."""
    s = create_linear_strategy(weight_dtype="fp8_e5m2", act_dtype="bf16")
    assert s.name == "linear_fp8_fp8_e5m2_w8a16"
    assert s.linear_weight_format == "fp8_e5m2"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_fp8_e4m3_w8a8():
    """Test that fp8_e4m3+fp8_e4m3 returns the real FP8 W8A8 strategy."""
    s = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
    assert s.name == "linear_fp8_fp8_e4m3_w8a8"
    assert s.linear_weight_format == "fp8_e4m3"
    assert s.linear_act_format == "fp8_e4m3"


def test_linear_strategy_registry_fp8_e5m2_w8a8():
    """Test that fp8_e5m2+fp8_e5m2 returns the real FP8 W8A8 strategy."""
    s = create_linear_strategy(weight_dtype="fp8_e5m2", act_dtype="fp8_e5m2")
    assert s.name == "linear_fp8_fp8_e5m2_w8a8"
    assert s.linear_weight_format == "fp8_e5m2"
    assert s.linear_act_format == "fp8_e5m2"


def test_fp8_w8a16_quantize_dequantize_roundtrip():
    """Test FP8 W8A16 quantization and dequantization roundtrip."""
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    
    # Create a test weight tensor
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    
    # Quantize
    quantized, scales = strategy.quantize(weight)
    
    # Check output types and shapes
    assert quantized.dtype == torch.uint8
    assert quantized.shape == weight.shape
    assert scales.dtype == torch.float32
    assert scales.shape == (weight.shape[0],)
    
    # Dequantize
    dequantized = strategy.dequantize(quantized, scales)
    
    # Check output type and shape
    assert dequantized.dtype == torch.bfloat16
    assert dequantized.shape == weight.shape
    
    # Check approximate recovery (FP8 has limited precision)
    # Use relaxed tolerance for FP8
    max_error = torch.abs(dequantized - weight).max()
    relative_error = torch.abs((dequantized - weight) / (weight.abs() + 1e-8)).max()
    # FP8 has ~3-4 bits of precision, so we expect some error
    assert max_error < 0.5  # Relaxed tolerance
    assert relative_error < 0.3  # 30% relative error is acceptable for FP8


def test_fp8_w8a16_forward():
    """Test FP8 W8A16 forward pass."""
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    
    # Create test tensors
    M, K, N = 4, 256, 128
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)
    
    # Compute reference output (bf16)
    ref_out = F.linear(x, weight, bias)
    
    # Compute FP8 quantized output
    fp8_out = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    
    # Check output shape
    assert fp8_out.shape == ref_out.shape
    assert fp8_out.dtype == torch.bfloat16
    
    # Check approximate correctness (FP8 has limited precision)
    max_error = torch.abs(fp8_out - ref_out).max()
    # FP8 quantization introduces error, but output should be reasonable
    # FP8 has ~3-4 bits of precision, so we use more relaxed tolerance
    # Only check absolute error to avoid issues with near-zero values
    assert max_error < 2.0  # Relaxed tolerance for FP8
    # Check that outputs are in similar range (not completely broken)
    assert fp8_out.abs().max() < ref_out.abs().max() * 3  # Output shouldn't be 3x larger


def test_fp8_w8a16_lazy_cache():
    """Test FP8 W8A16 lazy cache behavior."""
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    
    # Create test tensors
    M, K, N = 4, 256, 128
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)
    
    # First forward pass should quantize and cache
    out1 = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    assert len(strategy._weight_cache) == 1
    
    # Second forward pass should use cached quantized weight
    out2 = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    assert len(strategy._weight_cache) == 1  # Cache size unchanged
    
    # Outputs should be identical (same quantization)
    assert torch.allclose(out1, out2, atol=1e-5, rtol=1e-5)
    
    # Clear cache
    strategy.clear_cache()
    assert len(strategy._weight_cache) == 0


def test_fp8_w8a8_quantize_dequantize_roundtrip():
    """Test FP8 W8A8 quantization and dequantization roundtrip."""
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
    
    # Test weight quantization
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    quantized_weight, w_scales = strategy.quantize(weight)
    
    assert quantized_weight.dtype == torch.uint8
    assert quantized_weight.shape == weight.shape
    assert w_scales.dtype == torch.float16
    assert w_scales.shape == (weight.shape[0],)
    
    dequantized_weight = strategy.dequantize(quantized_weight, w_scales)
    assert dequantized_weight.dtype == torch.bfloat16
    assert dequantized_weight.shape == weight.shape
    
    # Test activation quantization
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    quantized_x, x_scales = strategy.quantize_act_for_kernel(x)
    
    assert quantized_x.dtype == torch.uint8
    assert quantized_x.shape == x.shape
    assert x_scales.dtype == torch.float32
    assert x_scales.shape == (x.shape[0],)
    
    # Dequantize activation
    dequantized_x = strategy._dequantize_act(quantized_x, x_scales)
    assert dequantized_x.dtype == torch.bfloat16
    assert dequantized_x.shape == x.shape


def test_fp8_w8a8_forward():
    """Test FP8 W8A8 forward pass."""
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
    
    # Create test tensors
    M, K, N = 4, 256, 128
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)
    
    # Compute reference output (bf16)
    ref_out = F.linear(x, weight, bias)
    
    # Compute FP8 quantized output
    fp8_out = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    
    # Check output shape
    assert fp8_out.shape == ref_out.shape
    assert fp8_out.dtype == torch.bfloat16
    
    # Check approximate correctness (FP8 has limited precision)
    max_error = torch.abs(fp8_out - ref_out).max()
    # FP8 W8A8 quantization introduces larger error since both weights and activations are quantized
    # FP8 has ~3-4 bits of precision, so we use more relaxed tolerance for W8A8
    # Only check absolute error to avoid issues with near-zero values
    assert max_error < 3.0  # More relaxed tolerance for FP8 W8A8 (both W and A quantized)
    # Check that outputs are in similar range (not completely broken)
    assert fp8_out.abs().max() < ref_out.abs().max() * 3  # Output shouldn't be 3x larger


def test_fp8_w8a8_lazy_cache():
    """Test FP8 W8A8 lazy cache behavior."""
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
    
    # Create test tensors
    M, K, N = 4, 256, 128
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)
    
    # First forward pass should quantize and cache weight
    out1 = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    assert len(strategy._weight_cache) == 1
    
    # Second forward pass should use cached quantized weight
    out2 = strategy.linear_forward(x, weight, bias, quant_kind="attn")
    assert len(strategy._weight_cache) == 1  # Cache size unchanged
    
    # Outputs should be identical (same quantization)
    assert torch.allclose(out1, out2, atol=1e-5, rtol=1e-5)
    
    # Clear cache
    strategy.clear_cache()
    assert len(strategy._weight_cache) == 0


def test_fp8_w8a16_load_time_quantization(monkeypatch):
    """Test FP8 W8A16 load-time quantization (quantized weight buffer)."""
    import torch.distributed as dist
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    
    from diffulex.layer.linear import ReplicatedLinear
    from diffulex.utils.quantization.context import get_quantization_context
    
    # Set up FP8 W8A16 strategy
    ctx = get_quantization_context()
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    ctx.set_linear_strategy("attn", strategy)
    
    # Create Linear layer
    linear = ReplicatedLinear(256, 128, bias=False, quant_kind="attn")
    
    # Load weight (should trigger quantization)
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    linear.weight.data.copy_(weight)
    linear.weight_loader(linear.weight, weight)
    
    # Check that bf16 weight Parameter is removed
    assert linear.weight is None or not hasattr(linear.weight, "data")
    
    # Check that quantized weight buffer is set
    assert linear.has_quantized_weight()
    assert linear.quant_weight_int8.dtype == torch.uint8
    assert linear.quant_weight_int8.shape == weight.shape
    assert linear.quant_scales.dtype == torch.float32
    assert linear.quant_scales.shape == (weight.shape[0],)
    
    # Test forward with quantized weight
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    out = linear(x)
    assert out.shape == (4, 128)
    assert out.dtype == torch.bfloat16


def test_fp8_w8a8_load_time_quantization(monkeypatch):
    """Test FP8 W8A8 load-time quantization (quantized weight buffer)."""
    import torch.distributed as dist
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    
    from diffulex.layer.linear import ReplicatedLinear
    from diffulex.utils.quantization.context import get_quantization_context
    
    # Set up FP8 W8A8 strategy
    ctx = get_quantization_context()
    strategy = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
    ctx.set_linear_strategy("attn", strategy)
    
    # Create Linear layer
    linear = ReplicatedLinear(256, 128, bias=False, quant_kind="attn")
    
    # Load weight (should trigger quantization)
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    linear.weight.data.copy_(weight)
    linear.weight_loader(linear.weight, weight)
    
    # Check that bf16 weight Parameter is removed
    assert linear.weight is None or not hasattr(linear.weight, "data")
    
    # Check that quantized weight buffer is set
    assert linear.has_quantized_weight()
    assert linear.quant_weight_int8.dtype == torch.uint8
    assert linear.quant_weight_int8.shape == weight.shape
    assert linear.quant_scales.dtype == torch.float16  # FP8 W8A8 uses float16 scales
    assert linear.quant_scales.shape == (weight.shape[0],)
    
    # Test forward with quantized weight
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    out = linear(x)
    assert out.shape == (4, 128)
    assert out.dtype == torch.bfloat16


def test_fp8_different_shapes():
    """Test FP8 strategies with different tensor shapes."""
    strategy_w8a16 = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    strategy_w8a8 = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="fp8_e4m3")
    
    # Test various shapes
    shapes = [
        (1, 64, 32),   # Small decode
        (4, 128, 64),  # Small batch
        (16, 256, 128), # Medium batch
        (32, 512, 256), # Large batch
    ]
    
    for M, K, N in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16)
        weight = torch.randn(N, K, dtype=torch.bfloat16)
        bias = torch.randn(N, dtype=torch.bfloat16)
        
        # Test W8A16
        out_w8a16 = strategy_w8a16.linear_forward(x, weight, bias, quant_kind="attn")
        assert out_w8a16.shape == (M, N)
        assert out_w8a16.dtype == torch.bfloat16
        
        # Test W8A8
        out_w8a8 = strategy_w8a8.linear_forward(x, weight, bias, quant_kind="attn")
        assert out_w8a8.shape == (M, N)
        assert out_w8a8.dtype == torch.bfloat16


def test_fp8_e5m2_vs_e4m3():
    """Test both FP8 formats (e4m3 and e5m2)."""
    # Test W8A16 with both formats
    strategy_e4m3 = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    strategy_e5m2 = create_linear_strategy(weight_dtype="fp8_e5m2", act_dtype="bf16")
    
    M, K, N = 4, 256, 128
    x = torch.randn(M, K, dtype=torch.bfloat16)
    weight = torch.randn(N, K, dtype=torch.bfloat16)
    bias = torch.randn(N, dtype=torch.bfloat16)
    
    out_e4m3 = strategy_e4m3.linear_forward(x, weight, bias, quant_kind="attn")
    out_e5m2 = strategy_e5m2.linear_forward(x, weight, bias, quant_kind="attn")
    
    # Both should produce valid outputs
    assert out_e4m3.shape == (M, N)
    assert out_e5m2.shape == (M, N)
    assert out_e4m3.dtype == torch.bfloat16
    assert out_e5m2.dtype == torch.bfloat16

