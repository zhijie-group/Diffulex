"""
端到端测试：验证量化策略在实际使用场景中的集成
"""
import torch
from types import SimpleNamespace

from diffulex.utils.quantization.factory import QuantizationStrategyFactory
from diffulex.utils.quantization.context import get_kv_cache_strategy
from diffulex.attention.metadata import AttnMetaDataBase


def test_bf16_e2e():
    """端到端测试：BF16路径的完整流程"""
    print("\n=== BF16 端到端测试 ===")
    
    # 1. 配置初始化
    config = SimpleNamespace(
        kv_cache_dtype="bf16",
        attn_q_dtype="bf16",
    )
    ctx = QuantizationStrategyFactory.create_from_config(config)
    strategy = get_kv_cache_strategy()
    
    # 2. 验证存储dtype
    storage_dtype, itemsize = strategy.get_storage_dtype()
    assert storage_dtype == torch.bfloat16
    print(f"✓ 存储dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # 3. 模拟KV cache分配（类似ModelRunner.allocate_kv_cache）
    num_layers = 2
    num_blocks = 4
    block_size = 32
    num_kv_heads = 8
    head_dim = 128
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 分配KV cache（unified layout）
    kv_cache = torch.zeros(
        2, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
        dtype=storage_dtype, device=device
    )
    print(f"✓ KV cache分配: shape={kv_cache.shape}, dtype={kv_cache.dtype}")
    
    # 4. 验证不需要scales
    k_scale_init, v_scale_init = strategy.init_scales(num_kv_heads, device)
    assert k_scale_init is None or v_scale_init is None
    print("✓ BF16不需要scales")
    
    # 5. 模拟attention forward（类似Attention.forward）
    seq_len = 16
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    # 模拟scale更新（应该被跳过，因为BF16不需要）
    k_scale, v_scale = None, None
    if strategy.requires_kv_cache_scales:
        k_scale, v_scale = strategy.update_scales(k, v, k_scale, v_scale, num_kv_heads, device)
    assert k_scale is None or v_scale is None
    print("✓ Scale更新被正确跳过")
    
    # 6. 模拟metadata设置
    attn_metadata = AttnMetaDataBase()
    strategy.maybe_set_attn_metadata_scales(attn_metadata, k_scale=k_scale, v_scale=v_scale)
    assert attn_metadata.k_scale is None
    assert attn_metadata.v_scale is None
    print("✓ Metadata scales未设置（符合预期）")
    
    # 7. 验证cache view（应该直接返回原cache）
    cache_view = strategy.view_kv_cache_for_kernels(kv_cache[0, 0, 0])
    assert cache_view is kv_cache[0, 0, 0] or torch.equal(cache_view, kv_cache[0, 0, 0])
    print("✓ Cache view正确（直接返回原cache）")
    
    print("✅ BF16端到端测试通过")


def test_fp8_e2e():
    """端到端测试：FP8路径的完整流程"""
    print("\n=== FP8 端到端测试 ===")
    
    # 检查FP8支持
    has_fp8 = hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")
    if not has_fp8:
        print("⚠ 当前PyTorch版本不支持FP8，跳过FP8端到端测试")
        return True
    
    # 1. 配置初始化
    config = SimpleNamespace(
        kv_cache_dtype="fp8",
        attn_q_dtype="bf16",
    )
    ctx = QuantizationStrategyFactory.create_from_config(config)
    strategy = get_kv_cache_strategy()
    
    # 2. 验证存储dtype
    storage_dtype, itemsize = strategy.get_storage_dtype()
    assert storage_dtype == torch.uint8
    assert itemsize == 1
    print(f"✓ 存储dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # 3. 模拟KV cache分配
    num_layers = 2
    num_blocks = 4
    block_size = 32
    num_kv_heads = 8
    head_dim = 128
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 分配KV cache（unified layout，uint8存储）
    kv_cache = torch.zeros(
        2, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
        dtype=storage_dtype, device=device
    )
    print(f"✓ KV cache分配: shape={kv_cache.shape}, dtype={kv_cache.dtype}")
    
    # 4. 分配scales（类似ModelRunner.allocate_kv_cache）
    k_scale_init, v_scale_init = strategy.init_scales(num_kv_heads, device)
    assert k_scale_init is not None and v_scale_init is not None
    
    k_scale = torch.zeros(num_layers, num_kv_heads, dtype=torch.float32, device=device)
    v_scale = torch.zeros(num_layers, num_kv_heads, dtype=torch.float32, device=device)
    k_scale[:] = k_scale_init[None, :]
    v_scale[:] = v_scale_init[None, :]
    print(f"✓ Scales分配: k_scale={k_scale.shape}, v_scale={v_scale.shape}")
    
    # 5. 模拟attention forward
    seq_len = 16
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    # 模拟scale更新（类似Attention.forward中的逻辑）
    layer_id = 0
    k_scale_layer = k_scale[layer_id]
    v_scale_layer = v_scale[layer_id]
    
    k_scale_updated, v_scale_updated = strategy.update_scales(
        k, v, k_scale_layer, v_scale_layer, num_kv_heads, device
    )
    assert k_scale_updated is not None and v_scale_updated is not None
    assert k_scale_updated.shape == (num_kv_heads,)
    assert v_scale_updated.shape == (num_kv_heads,)
    print(f"✓ Scale更新: k_scale范围=[{k_scale_updated.min():.4f}, {k_scale_updated.max():.4f}]")
    
    # 更新全局scales
    k_scale[layer_id] = k_scale_updated
    v_scale[layer_id] = v_scale_updated
    
    # 6. 模拟metadata设置（类似Attention.forward）
    attn_metadata = AttnMetaDataBase()
    strategy.maybe_set_attn_metadata_scales(
        attn_metadata, k_scale=k_scale_layer, v_scale=v_scale_layer
    )
    assert attn_metadata.k_scale is not None
    assert attn_metadata.v_scale is not None
    print("✓ Metadata scales已设置")
    
    # 7. 验证cache view（应该返回float8 view）
    cache_view = strategy.view_kv_cache_for_kernels(kv_cache[0, 0, 0])
    assert cache_view.dtype != torch.uint8
    print(f"✓ Cache view dtype: {cache_view.dtype}")
    
    # 8. 模拟quantize_kv_for_store（类似store_kvcache中的逻辑）
    k_quantized, v_quantized = strategy.quantize_kv_for_store(
        k, v, k_scale=k_scale_layer, v_scale=v_scale_layer
    )
    assert k_quantized.dtype == torch.uint8
    assert v_quantized.dtype == torch.uint8
    assert k_quantized.shape == k.shape
    assert v_quantized.shape == v.shape
    print(f"✓ KV量化: k={k_quantized.shape}, v={v_quantized.shape}")
    
    print("✅ FP8端到端测试通过")


if __name__ == "__main__":
    print("开始端到端测试...")
    
    try:
        test_bf16_e2e()
        test_fp8_e2e()
        print("\n✅ 所有端到端测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

