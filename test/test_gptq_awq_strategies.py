"""
Unit tests for GPTQ/AWQ quantization strategies.

These tests verify the dequantization correctness for GPTQ and AWQ formats.
"""

import pytest
import torch
import torch.nn.functional as F

from diffulex.utils.quantization.strategies.linear_gptq_w4a16 import (
    LinearGPTQW4A16Strategy,
    _dequantize_gptq,
    _unpack_gptq_int4,
)
from diffulex.utils.quantization.strategies.linear_awq_w4a16 import (
    LinearAWQW4A16Strategy,
    _dequantize_awq,
    _unpack_awq_int4,
)


def _pack_int4_to_int8(int4_tensor: torch.Tensor) -> torch.Tensor:
    """Pack int4 tensor into int8 format for testing.
    
    This matches the unpack implementation in _unpack_gptq_int4:
    - Lower 4 bits: even columns (0, 2, 4, ...)
    - Upper 4 bits: odd columns (1, 3, 5, ...)
    """
    out_features, in_features = int4_tensor.shape
    
    # Clamp to int4 range [-8, 7]
    int4_tensor = int4_tensor.clamp(-8, 7)
    
    # Pad in_features to even number if needed
    if in_features % 2 != 0:
        pad_size = 1
        padding = torch.zeros(out_features, pad_size, dtype=int4_tensor.dtype, device=int4_tensor.device)
        int4_tensor = torch.cat([int4_tensor, padding], dim=1)
        padded_in_features = in_features + pad_size
    else:
        padded_in_features = in_features
    
    # Convert to uint8 for bit manipulation
    # Map [-8, 7] to [0, 15] by adding 8
    uint8_tensor = (int4_tensor + 8).to(torch.uint8)
    
    # Reshape to [out_features, in_features // 2, 2]
    reshaped = uint8_tensor.view(out_features, padded_in_features // 2, 2)
    
    # Pack: even columns (reshaped[:, :, 0]) in lower 4 bits, odd columns (reshaped[:, :, 1]) in upper 4 bits
    # This matches unpack: low = p_u8 & 0x0F (even), high = (p_u8 >> 4) & 0x0F (odd)
    packed = reshaped[:, :, 0] | (reshaped[:, :, 1] << 4)
    
    # Convert back to int8
    return packed.to(torch.int8)


@pytest.mark.parametrize("out_features,in_features,group_size", [
    (128, 256, 128),
    (256, 512, 128),
    (128, 128, 128),
])
def test_gptq_unpack_pack_roundtrip(out_features, in_features, group_size):
    """Test that unpack and pack operations are inverse."""
    # Create random int4 weights
    weight_int4 = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    
    # Pack to int8
    packed = _pack_int4_to_int8(weight_int4)
    
    # Unpack back
    unpacked = _unpack_gptq_int4(packed, out_features=out_features, in_features=in_features)
    
    # Verify roundtrip
    assert unpacked.shape == weight_int4.shape
    torch.testing.assert_close(unpacked, weight_int4, rtol=0, atol=0)


@pytest.mark.parametrize("out_features,in_features,group_size", [
    (128, 256, 128),
    (256, 512, 128),
    (128, 128, 128),
])
def test_awq_unpack_pack_roundtrip(out_features, in_features, group_size):
    """Test that unpack and pack operations are inverse."""
    # Create random int4 weights
    weight_int4 = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    
    # Pack to int8
    packed = _pack_int4_to_int8(weight_int4)
    
    # Unpack back
    unpacked = _unpack_awq_int4(packed, out_features=out_features, in_features=in_features)
    
    # Verify roundtrip
    assert unpacked.shape == weight_int4.shape
    torch.testing.assert_close(unpacked, weight_int4, rtol=0, atol=0)


@pytest.mark.parametrize("out_features,in_features,group_size", [
    (128, 256, 128),
    (256, 512, 128),
    (128, 128, 128),
])
def test_gptq_dequantize_correctness(out_features, in_features, group_size):
    """Test GPTQ dequantization correctness."""
    device = torch.device("cpu")
    
    # Create reference float weights
    weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32)
    
    # Simulate GPTQ quantization
    num_groups = (out_features + group_size - 1) // group_size
    
    # Quantize per group
    qweight_list = []
    qzeros_list = []
    scales_list = []
    
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, out_features)
        group_weight = weight_fp32[start_idx:end_idx]  # [group_size, in_features]
        
        # Compute scale per group (per input channel for GPTQ/AWQ)
        # GPTQ/AWQ typically uses per-channel scales: [in_features]
        abs_max_per_channel = torch.abs(group_weight).max(dim=0, keepdim=False)[0]  # [in_features]
        scales_per_channel = (abs_max_per_channel.clamp(min=1e-8) / 7.0).to(torch.float32)  # [in_features]
        
        # Per-group zero point (typically zero for symmetric quantization)
        zeros_per_channel = torch.zeros(in_features, dtype=torch.float32)
        
        # Quantize weight for this group
        qweight_group = torch.round(group_weight / scales_per_channel.unsqueeze(0)).clamp(-8, 7).to(torch.int8)
        # Quantize zeros (should be zero, but compute for consistency)
        qzeros_per_channel = torch.round(zeros_per_channel / scales_per_channel).clamp(-8, 7).to(torch.int8)
        
        qweight_list.append(qweight_group)
        qzeros_list.append(qzeros_per_channel.unsqueeze(0))  # [1, in_features]
        scales_list.append(scales_per_channel.unsqueeze(0))  # [1, in_features]
    
    # Concatenate groups
    qweight = torch.cat(qweight_list, dim=0)  # [out_features, in_features]
    qzeros = torch.cat(qzeros_list, dim=0)  # [num_groups, in_features]
    scales = torch.cat(scales_list, dim=0)  # [num_groups, in_features]
    
    # Ensure shapes are correct
    assert qzeros.shape == (num_groups, in_features), f"qzeros shape mismatch: got {qzeros.shape}, expected ({num_groups}, {in_features})"
    assert scales.shape == (num_groups, in_features), f"scales shape mismatch: got {scales.shape}, expected ({num_groups}, {in_features})"
    
    # Pack to int8
    qweight_packed = _pack_int4_to_int8(qweight)
    qzeros_packed = _pack_int4_to_int8(qzeros)
    
    # Dequantize
    dequantized = _dequantize_gptq(
        qweight=qweight_packed,
        qzeros=qzeros_packed,
        scales=scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        g_idx=None,
    )
    
    # Verify approximate correctness (allow small quantization error)
    assert dequantized.shape == weight_fp32.shape
    # Note: Exact match is not expected due to quantization, but should be close
    error = torch.abs(dequantized.float() - weight_fp32)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    # Allow reasonable quantization error
    assert max_error < 1.0, f"Max quantization error too large: {max_error}"
    assert mean_error < 0.5, f"Mean quantization error too large: {mean_error}"


@pytest.mark.parametrize("out_features,in_features,group_size", [
    (128, 256, 128),
    (256, 512, 128),
    (128, 128, 128),
])
def test_awq_dequantize_correctness(out_features, in_features, group_size):
    """Test AWQ dequantization correctness."""
    device = torch.device("cpu")
    
    # Create reference float weights
    weight_fp32 = torch.randn(out_features, in_features, dtype=torch.float32)
    
    # Simulate AWQ quantization
    num_groups = (out_features + group_size - 1) // group_size
    
    # Quantize per group (sequential grouping)
    qweight_list = []
    qzeros_list = []
    scales_list = []
    
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, out_features)
        group_weight = weight_fp32[start_idx:end_idx]  # [group_size, in_features]
        
        # Compute scale per group (per input channel for AWQ)
        # AWQ typically uses per-channel scales: [in_features]
        abs_max_per_channel = torch.abs(group_weight).max(dim=0, keepdim=False)[0]  # [in_features]
        scales_per_channel = (abs_max_per_channel.clamp(min=1e-8) / 7.0).to(torch.float32)  # [in_features]
        
        # Per-group zero point (typically zero for symmetric quantization)
        zeros_per_channel = torch.zeros(in_features, dtype=torch.float32)
        
        # Quantize weight for this group
        qweight_group = torch.round(group_weight / scales_per_channel.unsqueeze(0)).clamp(-8, 7).to(torch.int8)
        # Quantize zeros (should be zero, but compute for consistency)
        qzeros_per_channel = torch.round(zeros_per_channel / scales_per_channel).clamp(-8, 7).to(torch.int8)
        
        qweight_list.append(qweight_group)
        qzeros_list.append(qzeros_per_channel.unsqueeze(0))  # [1, in_features]
        scales_list.append(scales_per_channel.unsqueeze(0))  # [1, in_features]
    
    # Concatenate groups
    qweight = torch.cat(qweight_list, dim=0)  # [out_features, in_features]
    qzeros = torch.cat(qzeros_list, dim=0)  # [num_groups, in_features]
    scales = torch.cat(scales_list, dim=0)  # [num_groups, in_features]
    
    # Ensure shapes are correct
    assert qzeros.shape == (num_groups, in_features), f"qzeros shape mismatch: got {qzeros.shape}, expected ({num_groups}, {in_features})"
    assert scales.shape == (num_groups, in_features), f"scales shape mismatch: got {scales.shape}, expected ({num_groups}, {in_features})"
    
    # Pack to int8
    qweight_packed = _pack_int4_to_int8(qweight)
    qzeros_packed = _pack_int4_to_int8(qzeros)
    
    # Dequantize
    dequantized = _dequantize_awq(
        qweight=qweight_packed,
        qzeros=qzeros_packed,
        scales=scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
    )
    
    # Verify approximate correctness
    assert dequantized.shape == weight_fp32.shape
    error = torch.abs(dequantized.float() - weight_fp32)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    # Allow reasonable quantization error
    assert max_error < 1.0, f"Max quantization error too large: {max_error}"
    assert mean_error < 0.5, f"Mean quantization error too large: {mean_error}"


def test_gptq_strategy_linear_forward():
    """Test GPTQ strategy linear forward pass."""
    strategy = LinearGPTQW4A16Strategy()
    
    out_features, in_features = 128, 256
    group_size = 128
    num_groups = (out_features + group_size - 1) // group_size
    
    # Create mock GPTQ tensors
    qweight = torch.randint(-128, 127, (out_features, (in_features + 1) // 2), dtype=torch.int8)
    qzeros = torch.randint(-128, 127, (num_groups, (in_features + 1) // 2), dtype=torch.int8)
    scales = torch.randn(num_groups, in_features, dtype=torch.float32).abs() + 0.1
    
    # Create input
    batch_size = 4
    x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
    
    # Forward pass
    output = strategy.linear_forward(
        x=x,
        weight=None,
        bias=None,
        quant_kind="other",
        gptq_qweight=qweight,
        gptq_qzeros=qzeros,
        gptq_scales=scales,
        gptq_group_size=group_size,
        out_features=out_features,
        in_features=in_features,
    )
    
    # Verify output shape
    assert output.shape == (batch_size, out_features)
    assert output.dtype == torch.bfloat16


def test_awq_strategy_linear_forward():
    """Test AWQ strategy linear forward pass."""
    strategy = LinearAWQW4A16Strategy()
    
    out_features, in_features = 128, 256
    group_size = 128
    num_groups = (out_features + group_size - 1) // group_size
    
    # Create mock AWQ tensors
    qweight = torch.randint(-128, 127, (out_features, (in_features + 1) // 2), dtype=torch.int8)
    qzeros = torch.randint(-128, 127, (num_groups, (in_features + 1) // 2), dtype=torch.int8)
    scales = torch.randn(num_groups, in_features, dtype=torch.float32).abs() + 0.1
    
    # Create input
    batch_size = 4
    x = torch.randn(batch_size, in_features, dtype=torch.bfloat16)
    
    # Forward pass
    output = strategy.linear_forward(
        x=x,
        weight=None,
        bias=None,
        quant_kind="other",
        awq_qweight=qweight,
        awq_qzeros=qzeros,
        awq_scales=scales,
        awq_group_size=group_size,
        out_features=out_features,
        in_features=in_features,
    )
    
    # Verify output shape
    assert output.shape == (batch_size, out_features)
    assert output.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
