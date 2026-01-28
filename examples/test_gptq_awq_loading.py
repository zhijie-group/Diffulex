#!/usr/bin/env python3
"""测试 GPTQ/AWQ 离线量化权重加载功能

此脚本演示如何加载 GPTQ/AWQ 格式的量化 checkpoint 并验证权重是否正确加载。

使用方法:
    # 测试 GPTQ checkpoint 加载
    python test_gptq_awq_loading.py --format gptq --model-path /path/to/gptq/checkpoint

    # 测试 AWQ checkpoint 加载
    python test_gptq_awq_loading.py --format awq --model-path /path/to/awq/checkpoint

    # 列出所有线性层及其量化状态
    python test_gptq_awq_loading.py --format gptq --model-path /path/to/checkpoint --list-layers
"""
import os
import sys
import argparse
from pathlib import Path

# Make stdout/stderr line-buffered so progress logs are visible even when redirected/captured.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# 确保从当前仓库导入
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diffulex.config import Config
from diffulex.model import AutoModelForDiffusionLM
from diffulex.utils.quantization.context import get_linear_strategy


def list_quantized_layers(model, format_name: str):
    """列出所有线性层及其量化状态."""
    print("\n" + "=" * 80)
    print(f"线性层量化状态 ({format_name.upper()})")
    print("=" * 80)
    print(f"{'模块名称':<50} {'类型':<15} {'量化状态':<15}")
    print("-" * 80)
    
    gptq_count = 0
    awq_count = 0
    other_count = 0
    no_quant_count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, "has_offline_quantized_weight"):
            if module.has_offline_quantized_weight():
                format_val = int(module._offline_quant_format.item()) if module._offline_quant_format.numel() > 0 else 0
                if format_val == 1:
                    quant_status = "GPTQ (离线)"
                    gptq_count += 1
                elif format_val == 2:
                    quant_status = "AWQ (离线)"
                    awq_count += 1
                else:
                    quant_status = "未知"
                    other_count += 1
                module_type = module.__class__.__name__
                print(f"{name:<50} {module_type:<15} {quant_status:<15}")
            elif hasattr(module, "has_quantized_weight") and module.has_quantized_weight():
                quant_status = "运行时量化"
                module_type = module.__class__.__name__
                print(f"{name:<50} {module_type:<15} {quant_status:<15}")
                other_count += 1
            elif hasattr(module, "weight") and module.weight is not None:
                quant_status = "未量化"
                module_type = module.__class__.__name__
                if "Linear" in module_type:
                    print(f"{name:<50} {module_type:<15} {quant_status:<15}")
                    no_quant_count += 1
    
    print("-" * 80)
    print(f"\n统计:")
    print(f"  - GPTQ 离线量化层: {gptq_count}")
    print(f"  - AWQ 离线量化层: {awq_count}")
    print(f"  - 运行时量化层: {other_count}")
    print(f"  - 未量化层: {no_quant_count}")
    print(f"  - 总计: {gptq_count + awq_count + other_count + no_quant_count}")


def test_model_forward(model, config, num_test_inputs: int = 2):
    """测试模型前向传播."""
    print("\n" + "=" * 80)
    print("测试模型前向传播")
    print("=" * 80)
    
    # 获取模型的输入大小（从第一个线性层的 input_size 推断）
    hidden_size = None
    for name, module in model.named_modules():
        if hasattr(module, "input_size"):
            hidden_size = module.input_size
            break
    
    if hidden_size is None:
        print("⚠ 无法确定模型的 hidden_size，跳过前向传播测试")
        return
    
    print(f"使用 hidden_size={hidden_size}")
    
    try:
        import torch
        import torch.nn.functional as F
        
        # 创建测试输入
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_inputs = []
        for i in range(num_test_inputs):
            x = torch.randn(1, hidden_size, dtype=torch.bfloat16, device=device)
            test_inputs.append(x)
        
        print(f"\n运行 {len(test_inputs)} 个测试输入...")
        for i, x in enumerate(test_inputs):
            print(f"\n  测试输入 {i+1}/{len(test_inputs)}: shape={x.shape}, dtype={x.dtype}")
            
            # 测试第一个线性层的 forward
            found_linear = False
            for name, module in model.named_modules():
                if hasattr(module, "forward") and hasattr(module, "quant_kind"):
                    try:
                        output = module(x)
                        print(f"    ✓ {name}: output shape={output.shape}, dtype={output.dtype}")
                        found_linear = True
                        break
                    except Exception as e:
                        print(f"    ✗ {name}: 错误 - {e}")
                        import traceback
                        traceback.print_exc()
                        break
            
            if not found_linear:
                print(f"    ⚠ 未找到可测试的线性层")
        
        print("\n✓ 前向传播测试完成")
        
    except Exception as e:
        print(f"\n✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="测试 GPTQ/AWQ 离线量化权重加载功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --format gptq --model-path /path/to/gptq/checkpoint
  %(prog)s --format awq --model-path /path/to/awq/checkpoint
  %(prog)s --format gptq --model-path /path/to/checkpoint --list-layers --test-forward
        """
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["gptq", "awq"],
        required=True,
        help="量化格式: 'gptq' 或 'awq'"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型 checkpoint 路径（包含 .safetensors 文件）"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dream",
        help="模型名称（默认: 'dream'）"
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="列出所有线性层及其量化状态"
    )
    parser.add_argument(
        "--test-forward",
        action="store_true",
        help="测试模型前向传播"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (默认: 1，仅 TP=1 支持离线量化权重加载)"
    )
    
    args = parser.parse_args()
    
    # 验证模型路径
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 模型路径不存在: {model_path}")
        sys.exit(1)
    
    safetensors_files = list(model_path.glob("*.safetensors"))
    if not safetensors_files:
        print(f"警告: 在 {model_path} 中未找到 .safetensors 文件")
    
    print("=" * 80)
    print("GPTQ/AWQ 离线量化权重加载测试")
    print("=" * 80)
    print(f"量化格式: {args.format.upper()}")
    print(f"模型路径: {model_path}")
    print(f"模型名称: {args.model_name}")
    print(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    print(f"找到 {len(safetensors_files)} 个 .safetensors 文件")
    print("=" * 80)
    
    # 检查 safetensors 文件中是否包含 GPTQ/AWQ keys
    if safetensors_files:
        print("\n检查 checkpoint 中的量化 keys...")
        gptq_keys = []
        awq_keys = []
        for file in safetensors_files:
            from safetensors import safe_open
            with safe_open(file, "pt", "cpu") as f:
                for key in f.keys():
                    if key.endswith(".qweight"):
                        gptq_keys.append(key)
                        awq_keys.append(key)
                    elif key.endswith(".qzeros"):
                        gptq_keys.append(key)
                        awq_keys.append(key)
                    elif key.endswith(".scales"):
                        gptq_keys.append(key)
                        awq_keys.append(key)
                    elif key.endswith(".g_idx"):
                        gptq_keys.append(key)
        
        print(f"  找到 {len(set(k.rsplit('.', 1)[0] for k in gptq_keys if k.endswith('.qweight')))} 个可能的量化层")
        if gptq_keys and args.format == "gptq":
            print(f"  找到 {len([k for k in gptq_keys if k.endswith('.g_idx')])} 个 g_idx keys (GPTQ)")
    
    # 创建配置
    try:
        config = Config(
            model=str(model_path),
            model_name=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            data_parallel_size=1,
            linear_attn_weight_dtype=args.format,
            linear_mlp_weight_dtype=args.format,
            linear_attn_act_dtype="bf16",
            linear_mlp_act_dtype="bf16",
            use_lora=False,
            gpu_memory_utilization=0.3,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            max_model_len=1024,
            decoding_strategy="d2f",
            enforce_eager=True,
        )
        print("\n✓ 配置创建成功")
    except Exception as e:
        print(f"\n✗ 配置创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 检查 TP 支持
    if args.tensor_parallel_size > 1:
        print("\n⚠ 警告: Tensor Parallel > 1 目前不完全支持离线量化权重加载")
        print("  如果遇到问题，请使用 --tensor-parallel-size 1")
    
    # 加载模型
    print("\n加载模型...")
    try:
        model = AutoModelForDiffusionLM.from_config(config)
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"\n✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 列出量化层
    if args.list_layers:
        list_quantized_layers(model, args.format)
    
    # 测试前向传播
    if args.test_forward:
        test_model_forward(model, config)
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
