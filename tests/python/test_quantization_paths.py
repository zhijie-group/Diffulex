"""
验证量化路径：bf16路径和bf16+fp8 kv路径
"""
import torch
from types import SimpleNamespace

from diffulex.utils.quantization.factory import QuantizationStrategyFactory
from diffulex.utils.quantization.context import (
    get_kv_cache_strategy,
    get_attn_q_strategy,
    QuantizationContext,
)


def test_bf16_path():
    """测试bf16路径（默认，无量化）"""
    print("\n=== 测试 BF16 路径 ===")
    
    # 创建配置
    config = SimpleNamespace(
        kv_cache_dtype="bf16",
        attn_q_dtype="bf16",
    )
    
    # 初始化量化上下文
    ctx = QuantizationStrategyFactory.create_from_config(config)
    
    # 获取策略
    kv_strategy = get_kv_cache_strategy()
    attn_q_strategy = get_attn_q_strategy()
    
    assert kv_strategy is not None, "KV cache策略应该被创建"
    assert attn_q_strategy is not None, "Attn-Q策略应该被创建"
    
    print(f"KV Cache策略: {kv_strategy.name}")
    print(f"KV Cache格式: {kv_strategy.kv_cache_format}")
    print(f"需要scales: {kv_strategy.requires_kv_cache_scales}")
    
    print(f"Attn-Q策略: {attn_q_strategy.name}")
    print(f"Attn-Q格式: {attn_q_strategy.attn_q_format}")
    
    # 验证存储dtype
    storage_dtype, itemsize = kv_strategy.get_storage_dtype()
    assert storage_dtype == torch.bfloat16, f"期望bfloat16，得到{storage_dtype}"
    assert itemsize == 2, f"期望itemsize=2，得到{itemsize}"
    print(f"存储dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # 验证不需要scales
    assert not kv_strategy.requires_kv_cache_scales, "BF16不应该需要scales"
    
    # 验证scale初始化
    num_kv_heads = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_scale, v_scale = kv_strategy.init_scales(num_kv_heads, device)
    assert k_scale is None or v_scale is None, "BF16策略应该返回None scales"
    print("✓ BF16路径验证通过")


def test_bf16_with_fp8_kv_path():
    """测试bf16 + fp8 kv路径"""
    print("\n=== 测试 BF16 + FP8 KV 路径 ===")
    
    # 检查是否支持FP8
    has_fp8 = hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")
    if not has_fp8:
        print("⚠ 当前PyTorch版本不支持FP8，跳过FP8测试")
        return True
    
    # 创建配置
    config = SimpleNamespace(
        kv_cache_dtype="fp8",  # 或 "fp8_e4m3"
        attn_q_dtype="bf16",
    )
    
    # 初始化量化上下文
    ctx = QuantizationStrategyFactory.create_from_config(config)
    
    # 获取策略
    kv_strategy = get_kv_cache_strategy()
    attn_q_strategy = get_attn_q_strategy()
    
    assert kv_strategy is not None, "KV cache策略应该被创建"
    assert attn_q_strategy is not None, "Attn-Q策略应该被创建"
    
    print(f"KV Cache策略: {kv_strategy.name}")
    print(f"KV Cache格式: {kv_strategy.kv_cache_format}")
    print(f"需要scales: {kv_strategy.requires_kv_cache_scales}")
    
    print(f"Attn-Q策略: {attn_q_strategy.name}")
    print(f"Attn-Q格式: {attn_q_strategy.attn_q_format}")
    
    # 验证存储dtype（FP8应该用uint8存储）
    storage_dtype, itemsize = kv_strategy.get_storage_dtype()
    assert storage_dtype == torch.uint8, f"期望uint8，得到{storage_dtype}"
    assert itemsize == 1, f"期望itemsize=1，得到{itemsize}"
    print(f"存储dtype: {storage_dtype}, itemsize: {itemsize}")
    
    # 验证需要scales
    assert kv_strategy.requires_kv_cache_scales, "FP8应该需要scales"
    
    # 验证scale初始化
    num_kv_heads = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    k_scale, v_scale = kv_strategy.init_scales(num_kv_heads, device)
    assert k_scale is not None and v_scale is not None, "FP8策略应该返回非None scales"
    assert k_scale.shape == (num_kv_heads,), f"k_scale形状应该是({num_kv_heads},)，得到{k_scale.shape}"
    assert v_scale.shape == (num_kv_heads,), f"v_scale形状应该是({num_kv_heads},)，得到{v_scale.shape}"
    print(f"初始scales形状: k_scale={k_scale.shape}, v_scale={v_scale.shape}")
    
    # 验证scale更新逻辑
    seq_len = 32
    head_dim = 128
    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device)
    
    # 第一次更新（从None开始）
    k_scale_new, v_scale_new = kv_strategy.update_scales(
        k, v, None, None, num_kv_heads, device
    )
    assert k_scale_new is not None and v_scale_new is not None
    assert k_scale_new.shape == (num_kv_heads,)
    assert v_scale_new.shape == (num_kv_heads,)
    print(f"第一次更新scales: k_scale范围=[{k_scale_new.min():.4f}, {k_scale_new.max():.4f}]")
    
    # 第二次更新（使用已有scales）
    k_scale_updated, v_scale_updated = kv_strategy.update_scales(
        k, v, k_scale_new, v_scale_new, num_kv_heads, device
    )
    assert k_scale_updated is not None and v_scale_updated is not None
    print(f"第二次更新scales: k_scale范围=[{k_scale_updated.min():.4f}, {k_scale_updated.max():.4f}]")
    
    # 验证view_kv_cache_for_kernels
    cache_u8 = torch.empty((16,), dtype=torch.uint8, device=device)
    cache_view = kv_strategy.view_kv_cache_for_kernels(cache_u8)
    assert cache_view.dtype != torch.uint8, "view应该返回非uint8的dtype"
    print(f"Cache view dtype: {cache_view.dtype}")
    
    # 验证quantize_kv_for_store
    k_quantized, v_quantized = kv_strategy.quantize_kv_for_store(
        k, v, k_scale=k_scale_new, v_scale=v_scale_new
    )
    assert k_quantized.dtype == torch.uint8, f"量化后的K应该是uint8，得到{k_quantized.dtype}"
    assert v_quantized.dtype == torch.uint8, f"量化后的V应该是uint8，得到{v_quantized.dtype}"
    assert k_quantized.shape == k.shape, f"量化后的K形状应该保持不变"
    assert v_quantized.shape == v.shape, f"量化后的V形状应该保持不变"
    print(f"量化后形状: k={k_quantized.shape}, v={v_quantized.shape}")
    
    print("✓ BF16 + FP8 KV路径验证通过")


def test_metadata_integration():
    """测试与AttnMetaData的集成"""
    print("\n=== 测试 Metadata 集成 ===")
    
    from diffulex.attention.metadata import AttnMetaDataBase
    
    # BF16路径
    config_bf16 = SimpleNamespace(kv_cache_dtype="bf16", attn_q_dtype="bf16")
    QuantizationStrategyFactory.create_from_config(config_bf16)
    kv_strategy_bf16 = get_kv_cache_strategy()
    
    md_bf16 = AttnMetaDataBase()
    kv_strategy_bf16.maybe_set_attn_metadata_scales(md_bf16, k_scale=None, v_scale=None)
    assert md_bf16.k_scale is None, "BF16不应该设置scales"
    assert md_bf16.v_scale is None, "BF16不应该设置scales"
    print("✓ BF16 metadata集成正常")
    
    # FP8路径（如果支持）
    has_fp8 = hasattr(torch, "float8_e4m3fn") or hasattr(torch, "float8_e4m3fnuz")
    if has_fp8:
        config_fp8 = SimpleNamespace(kv_cache_dtype="fp8", attn_q_dtype="bf16")
        QuantizationStrategyFactory.create_from_config(config_fp8)
        kv_strategy_fp8 = get_kv_cache_strategy()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        k_scale = torch.ones((8,), dtype=torch.float32, device=device)
        v_scale = torch.ones((8,), dtype=torch.float32, device=device) * 2
        
        md_fp8 = AttnMetaDataBase()
        kv_strategy_fp8.maybe_set_attn_metadata_scales(md_fp8, k_scale=k_scale, v_scale=v_scale)
        assert md_fp8.k_scale is k_scale, "FP8应该设置k_scale"
        assert md_fp8.v_scale is v_scale, "FP8应该设置v_scale"
        print("✓ FP8 metadata集成正常")


if __name__ == "__main__":
    print("开始验证量化路径...")
    
    try:
        test_bf16_path()
        test_bf16_with_fp8_kv_path()
        test_metadata_integration()
        print("\n✅ 所有路径验证通过！")
    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

