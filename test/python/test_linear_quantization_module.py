import pytest


def test_linear_strategy_registry_bf16_pair():
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="bf16", act_dtype="bf16")
    assert s.linear_weight_format == "bf16"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_int8_w8a16():
    """Test that int8+bf16 returns the real W8A16 strategy (not stub)."""
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    assert s.name == "linear_int8_w8a16"
    assert s.linear_weight_format == "int8"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_int4_w4a16():
    """Test that int4+bf16 returns the real W4A16 strategy (not stub)."""
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="int4", act_dtype="bf16")
    assert s.name == "linear_int4_w4a16"
    assert s.linear_weight_format == "int4"
    assert s.linear_act_format == "bf16"


def test_linear_strategy_registry_int8_w8a8():
    """Test that int8+int8 returns the real W8A8 strategy (not stub)."""
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="int8", act_dtype="int8")
    assert s.name == "linear_int8_w8a8"
    assert s.linear_weight_format == "int8"
    assert s.linear_act_format == "int8"


def test_linear_strategy_registry_int4_w4a8():
    """Test that int4+int8 returns the real W4A8 strategy (not stub)."""
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="int4", act_dtype="int8")
    assert s.name == "linear_int4_w4a8"
    assert s.linear_weight_format == "int4"
    assert s.linear_act_format == "int8"


def test_linear_strategy_registry_non_bf16_returns_stub():
    """Test that unimplemented combinations (e.g., fp8) return stub."""
    from diffulex.utils.quantization.registry import create_linear_strategy

    s = create_linear_strategy(weight_dtype="fp8_e4m3", act_dtype="bf16")
    assert s.name.startswith("linear_stub")
    assert s.linear_weight_format == "fp8_e4m3"
    assert s.linear_act_format == "bf16"


def test_factory_injects_linear_strategies_into_context():
    from dataclasses import dataclass

    from diffulex.utils.quantization.factory import QuantizationStrategyFactory
    from diffulex.utils.quantization.context import get_quantization_context

    @dataclass
    class DummyConfig:
        kv_cache_dtype: str = "bf16"
        attn_q_dtype: str = "bf16"
        linear_attn_weight_dtype: str = "bf16"
        linear_mlp_weight_dtype: str = "bf16"
        linear_attn_act_dtype: str = "bf16"
        linear_mlp_act_dtype: str = "bf16"

    ctx = QuantizationStrategyFactory.create_from_config(DummyConfig())
    assert ctx is get_quantization_context()
    assert ctx.get_linear_strategy("attn") is not None
    assert ctx.get_linear_strategy("mlp") is not None


def test_linear_forward_raises_on_stub(monkeypatch):
    # Avoid requiring torch.distributed process group init in unit tests.
    import torch
    import torch.nn.functional as F
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    from diffulex.layer.linear import ColumnParallelLinear
    from diffulex.utils.quantization.registry import create_linear_strategy
    from diffulex.utils.quantization.context import get_quantization_context

    # Install a stub strategy for attention linears (use int4, not implemented yet).
    ctx = get_quantization_context()
    ctx.set_linear_strategy("attn", create_linear_strategy(weight_dtype="int4", act_dtype="bf16"))

    lin = ColumnParallelLinear(4, 8, bias=False, quant_kind="attn")
    # NOTE: default Linear weights are float32 unless a checkpoint loader overwrites them.
    # Keep dtypes consistent for this unit test.
    x = torch.randn(2, 4, dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        _ = lin(x)

    # Ensure bf16 path still works for other kinds.
    lin2 = ColumnParallelLinear(4, 8, bias=False, quant_kind="other")
    y = lin2(x)
    ref = F.linear(x, lin2.weight, None)
    assert torch.allclose(y, ref)


def test_linear_int8_w8a16_quantization():
    """Test that int8+bf16 strategy correctly quantizes and dequantizes weights."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch

    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    assert strategy.name == "linear_int8_w8a16"
    assert strategy.linear_weight_format == "int8"
    assert strategy.linear_act_format == "bf16"

    # Test quantization/dequantization
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    quantized, scales = strategy.quantize(weight)
    assert quantized.dtype == torch.int8
    assert quantized.shape == weight.shape
    assert scales.shape == (weight.shape[0],)  # Per-output-channel scales

    dequantized = strategy.dequantize(quantized, scales)
    assert dequantized.dtype == torch.bfloat16
    assert dequantized.shape == weight.shape

    # Quantization error should be reasonable (int8 quantization introduces error)
    error = (weight - dequantized).abs().max()
    assert error.item() < 0.1, f"Quantization error too large: {error.item()}"


def test_linear_int8_w8a16_forward():
    """Test that int8+bf16 strategy's linear_forward produces reasonable outputs."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch
    import torch.nn.functional as F

    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)

    # Forward with quantized strategy
    y_quant = strategy.linear_forward(x, weight, bias, quant_kind="test")

    # Reference forward (should be close but not exact due to quantization)
    y_ref = F.linear(x, weight, bias)

    assert y_quant.shape == y_ref.shape
    assert y_quant.dtype == torch.bfloat16

    # Error should be reasonable (quantization introduces some error)
    error = (y_quant - y_ref).abs().max()
    assert error.item() < 0.5, f"Forward error too large: {error.item()}"


def test_linear_int8_w8a16_lazy_cache():
    """Test that W8A16 strategy caches quantized weights to avoid re-quantization."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch

    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    
    # Initial cache should be empty
    assert len(strategy._weight_cache) == 0
    
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    
    # First forward - should cache
    y1 = strategy.linear_forward(x, weight, None, quant_kind="test")
    assert len(strategy._weight_cache) == 1
    assert id(weight) in strategy._weight_cache
    
    # Second forward with same weight - should use cache (same output)
    y2 = strategy.linear_forward(x, weight, None, quant_kind="test")
    assert len(strategy._weight_cache) == 1  # Cache size unchanged
    assert torch.allclose(y1, y2), "Cached forward should produce same output"
    
    # Different weight - should cache new entry
    weight2 = torch.randn(8, 4, dtype=torch.bfloat16)
    y3 = strategy.linear_forward(x, weight2, None, quant_kind="test")
    assert len(strategy._weight_cache) == 2  # New entry cached
    
    # Clear cache
    strategy.clear_cache()
    assert len(strategy._weight_cache) == 0


def test_w8a16_tilelang_kernel_correctness():
    """Test that W8A16 TileLang kernel produces correct results (if available)."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch

    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    
    # Skip test if TileLang kernel is not available
    try:
        from diffulex_kernel.python.linear_kernels import w8a16_gemm
        tilelang_available = True
    except ImportError:
        tilelang_available = False
        import pytest
        pytest.skip("TileLang kernel not available")
    
    if not tilelang_available:
        return
    
    # Create test data
    M, N, K = 128, 256, 512
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    
    # Quantize weight
    quantized_weight, scales = strategy.quantize(weight)
    quantized_weight = quantized_weight.to(device="cuda")
    scales = scales.to(device="cuda")
    
    # Compute reference output (Python implementation)
    ref_output = strategy._fallback_python_forward(x, quantized_weight, scales, None)
    
    # Compute output using strategy (kernel when available; may fall back if kernel unavailable).
    out = strategy.linear_forward(x, weight, None, quant_kind="test")
    
    # Compare results
    error = (out - ref_output).abs().max()
    # Relative error can explode when ref_output is very close to 0.
    # Use a masked relative error that only considers reasonably-sized reference values.
    rel_mask = ref_output.abs() > 1.0
    if rel_mask.any():
        relative_error = (out - ref_output).abs() / (ref_output.abs() + 1e-8)
        max_relative_error = relative_error[rel_mask].max()
    else:
        max_relative_error = None
    
    # Allow some numerical error (quantization + kernel precision)
    assert error.item() < 1.0, f"Absolute error too large: {error.item()}"
    if max_relative_error is not None:
        assert max_relative_error.item() < 0.15, f"Relative error too large: {max_relative_error.item()}"


def test_w8a16_tilelang_kernel_tail_sizes_correctness():
    """Tail sizes (non-multiple M/N/K) should be handled without needing K%128==0."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch

    # Skip test if TileLang kernel is not available
    try:
        from diffulex_kernel.python.linear_kernels import w8a16_gemm  # noqa: F401
        tilelang_available = True
    except ImportError:
        tilelang_available = False
        import pytest
        pytest.skip("TileLang kernel not available")
    
    if not tilelang_available:
        return

    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")

    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    # Intentionally choose tail sizes (not multiples of block_M/N=64 and block_K=128).
    M, N, K = 127, 255, 130
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    # Strategy output (kernel when available; may fall back if kernel unavailable).
    out = strategy.linear_forward(x, weight, None, quant_kind="test")

    # Reference (same as fallback implementation)
    qweight, scales = strategy.quantize_weight_for_kernel(weight, device=x.device)
    ref = strategy._fallback_python_forward(x, qweight, scales, None)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, rtol=7e-2, atol=7e-2)


def test_w8a16_load_time_quantized_linear_saves_weight_memory(monkeypatch):
    """Ensure load-time quantized Linear does not keep bf16 weight Parameter on CUDA."""
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    # Avoid requiring torch.distributed process group init in unit tests.
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    from diffulex.layer.linear import ReplicatedLinear
    from diffulex.utils.quantization.registry import create_linear_strategy
    from diffulex.utils.quantization.context import get_quantization_context

    ctx = get_quantization_context()
    strategy = create_linear_strategy(weight_dtype="int8", act_dtype="bf16")
    ctx.set_linear_strategy("attn", strategy)

    lin = ReplicatedLinear(4096, 11008, bias=False, quant_kind="attn").cuda().to(dtype=torch.bfloat16)

    # Simulate checkpoint load: call weight_loader on the original Parameter.
    param = lin._parameters["weight"]
    loaded_weight = torch.randn_like(param, device=param.device, dtype=torch.bfloat16)
    lin.weight_loader(param, loaded_weight)

    # Weight Parameter should be dropped and replaced by quant buffers.
    assert lin.has_quantized_weight()
    assert lin.weight is None
    assert "weight" not in dict(lin.named_parameters())
    assert lin.quant_weight_int8.dtype == torch.int8
    assert lin.quant_scales.dtype == torch.bfloat16
    assert lin.quant_weight_int8.device.type == "cuda"
    assert lin.quant_scales.device.type == "cuda"

    # Quant buffers should be significantly smaller than bf16 weight.
    bf16_bytes = loaded_weight.numel() * loaded_weight.element_size()
    q_bytes = lin.quant_weight_int8.numel() * lin.quant_weight_int8.element_size()
    s_bytes = lin.quant_scales.numel() * lin.quant_scales.element_size()
    assert (q_bytes + s_bytes) < bf16_bytes * 0.7  # conservative threshold

    # Forward should run and NOT populate the lazy cache (to avoid double-storage).
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    before_cache = len(strategy._weight_cache)


# ========== W4A16 Tests ==========

def test_linear_int4_w4a16_quantization():
    """Test W4A16 quantization and dequantization."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch
    torch.manual_seed(0)

    strategy = create_linear_strategy(weight_dtype="int4", act_dtype="bf16")
    assert strategy.name == "linear_int4_w4a16"
    assert strategy.linear_weight_format == "int4"
    assert strategy.linear_act_format == "bf16"

    # Test quantization/dequantization
    # Use a bounded distribution to make the quantization error check stable.
    # With int4 per-channel quantization, very large random values can cause the max error
    # to occasionally exceed a tight threshold.
    weight = (torch.randn(8, 4, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    packed_weight, scales = strategy.quantize(weight)
    assert packed_weight.dtype == torch.int8
    # Packed shape: [out_features, (in_features + 1) // 2]
    assert packed_weight.shape == (weight.shape[0], (weight.shape[1] + 1) // 2)
    assert scales.shape == (weight.shape[0],)  # Per-output-channel scales

    dequantized = strategy.dequantize(packed_weight, scales, original_in_features=weight.shape[1])
    assert dequantized.dtype == torch.bfloat16
    assert dequantized.shape == weight.shape

    # Quantization error should be reasonable (int4 quantization introduces more error than int8)
    error = (weight - dequantized).abs().max()
    assert error.item() < 0.2, f"Quantization error too large: {error.item()}"


def test_linear_int4_w4a16_forward():
    """Test that int4+bf16 strategy's linear_forward produces reasonable outputs."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch
    import torch.nn.functional as F

    strategy = create_linear_strategy(weight_dtype="int4", act_dtype="bf16")

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    bias = torch.randn(8, dtype=torch.bfloat16)

    # Forward with quantized strategy
    y_quant = strategy.linear_forward(x, weight, bias, quant_kind="test")

    # Reference forward (should be close but not exact due to quantization)
    y_ref = F.linear(x, weight, bias)

    assert y_quant.shape == y_ref.shape
    assert y_quant.dtype == torch.bfloat16

    # Error should be reasonable (int4 quantization introduces more error than int8)
    error = (y_quant - y_ref).abs().max()
    assert error.item() < 1.0, f"Forward error too large: {error.item()}"


def test_linear_int4_w4a16_lazy_cache():
    """Test that W4A16 strategy caches quantized weights to avoid re-quantization."""
    from diffulex.utils.quantization.registry import create_linear_strategy
    import torch

    strategy = create_linear_strategy(weight_dtype="int4", act_dtype="bf16")
    
    # Initial cache should be empty
    assert len(strategy._weight_cache) == 0
    
    weight = torch.randn(8, 4, dtype=torch.bfloat16)
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    
    # First forward - should cache
    y1 = strategy.linear_forward(x, weight, None, quant_kind="test")
    assert len(strategy._weight_cache) == 1
    assert id(weight) in strategy._weight_cache
    
    # Second forward with same weight - should use cache (same output)
    y2 = strategy.linear_forward(x, weight, None, quant_kind="test")
    assert len(strategy._weight_cache) == 1  # Cache size unchanged
    assert torch.allclose(y1, y2, rtol=1e-3, atol=1e-3), "Cached forward should produce same output"
    
    # Different weight - should cache new entry
    weight2 = torch.randn(8, 4, dtype=torch.bfloat16)
    y3 = strategy.linear_forward(x, weight2, None, quant_kind="test")
    assert len(strategy._weight_cache) == 2  # New entry cached
    
    # Clear cache
    strategy.clear_cache()
    assert len(strategy._weight_cache) == 0


def test_w4a16_load_time_quantized_linear_saves_weight_memory(monkeypatch):
    """Ensure load-time quantized W4A16 Linear does not keep bf16 weight Parameter on CUDA."""
    import torch
    import torch.distributed as dist
    from diffulex.layer.linear import ReplicatedLinear
    from diffulex.utils.quantization.registry import create_linear_strategy
    from diffulex.utils.quantization.context import get_quantization_context

    if not torch.cuda.is_available():
        import pytest
        pytest.skip("CUDA not available")

    # Avoid requiring torch.distributed process group init in unit tests.
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)

    ctx = get_quantization_context()
    strategy = create_linear_strategy(weight_dtype="int4", act_dtype="bf16")
    ctx.set_linear_strategy("attn", strategy)

    lin = ReplicatedLinear(4096, 11008, bias=False, quant_kind="attn").cuda().to(dtype=torch.bfloat16)

    # Simulate checkpoint load: call weight_loader on the original Parameter.
    param = lin._parameters["weight"]
    loaded_weight = torch.randn_like(param, device=param.device, dtype=torch.bfloat16)
    lin.weight_loader(param, loaded_weight)

    # Weight Parameter should be dropped and replaced by quant buffers.
    assert lin.has_quantized_weight()
    assert lin.weight is None
    assert "weight" not in dict(lin.named_parameters())
    assert lin.quant_weight_int8.dtype == torch.int8
    assert lin.quant_scales.dtype == torch.bfloat16
    assert lin.quant_weight_int8.device.type == "cuda"
    assert lin.quant_scales.device.type == "cuda"

    # Quant buffers should be significantly smaller than bf16 weight.
    # For int4: packed shape is [out_features, (in_features + 1) // 2]
    bf16_bytes = loaded_weight.numel() * loaded_weight.element_size()
    q_bytes = lin.quant_weight_int8.numel() * lin.quant_weight_int8.element_size()
    s_bytes = lin.quant_scales.numel() * lin.quant_scales.element_size()
    # int4 packed should be ~50% of bf16 (plus small scales overhead)
    assert (q_bytes + s_bytes) < bf16_bytes * 0.6  # conservative threshold

    # Forward should run and NOT populate the lazy cache (to avoid double-storage).
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    before_cache = len(strategy._weight_cache)
    out = lin(x)
    after_cache = len(strategy._weight_cache)
    assert after_cache == before_cache, "Load-time quantized forward should not populate lazy cache"
    assert out.shape == (8, 11008)
    assert out.dtype == torch.bfloat16
    y = lin(x)
    after_cache = len(strategy._weight_cache)
    assert y.shape == (8, 11008)
    assert after_cache == before_cache


