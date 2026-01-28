#!/usr/bin/env python3
"""统一的量化策略文本生成测试脚本

支持测试以下量化策略组合：
- BF16 + BF16 KV
- BF16 + FP8 KV
- W8A16 + BF16 KV
- W8A16 + FP8 KV
- W4A16 + BF16 KV
- W4A16 + FP8 KV
- W8A8 + BF16 KV
- W8A8 + FP8 KV
- W4A8 + BF16 KV
- W4A8 + FP8 KV
- FP8 W8A16 (e4m3) + BF16 KV
- FP8 W8A16 (e4m3) + FP8 KV
- FP8 W8A16 (e5m2) + BF16 KV
- FP8 W8A16 (e5m2) + FP8 KV
- FP8 W8A8 (e4m3) + BF16 KV
- FP8 W8A8 (e4m3) + FP8 KV
- FP8 W8A8 (e5m2) + BF16 KV
- FP8 W8A8 (e5m2) + FP8 KV
- GPTQ W4A16 (离线量化) + BF16 KV
- GPTQ W4A16 (离线量化) + FP8 KV
- AWQ W4A16 (离线量化) + BF16 KV
- AWQ W4A16 (离线量化) + FP8 KV

使用方法:
    # 运行所有策略
    python test_quantization_generation.py --all

    # 只运行 BF16 相关策略
    python test_quantization_generation.py --bf16

    # 只运行 W8A16 相关策略
    python test_quantization_generation.py --w8a16

    # 只运行 W4A16 相关策略
    python test_quantization_generation.py --w4a16

    # 只运行 W8A8 相关策略
    python test_quantization_generation.py --w8a8

    # 只运行 W4A8 相关策略
    python test_quantization_generation.py --w4a8

    # 只运行 FP8 W8A16 相关策略
    python test_quantization_generation.py --fp8_w8a16

    # 只运行 FP8 W8A8 相关策略
    python test_quantization_generation.py --fp8_w8a8

    # 只运行 GPTQ 相关策略（需要先运行量化脚本生成离线权重）
    python test_quantization_generation.py --gptq

    # 只运行 AWQ 相关策略（需要先运行量化脚本生成离线权重）
    python test_quantization_generation.py --awq

    # 自定义选择（用逗号分隔）
    python test_quantization_generation.py --strategies bf16_bf16kv,w8a16_bf16kv

    # 只测试某个策略
    python test_quantization_generation.py --strategies w4a16_fp8kv

    # 使用量化后的模型路径（如果先运行了量化脚本）
    python test_quantization_generation.py --gptq --model-path /path/to/quantized/model
"""
import os
import sys
import time
import argparse
import gc
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple

# Make stdout/stderr line-buffered so progress logs are visible even when redirected/captured.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

#
# NOTE:
# 这个脚本不应假设本机 CUDA 安装路径或默认 GPU 号。
# 如需指定 CUDA/设备，请在运行前自行设置：
#   - CUDA_HOME / CUDA_PATH / PATH / LD_LIBRARY_PATH
#   - CUDA_VISIBLE_DEVICES
# 或者在你自己的 wrapper 脚本里处理。

# 确保从当前仓库导入
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer
from diffulex import Diffulex, SamplingParams


# 支持的策略配置
STRATEGY_CONFIGS = {
    'bf16_bf16kv': {
        'name': 'BF16 + BF16 KV',
        'linear_attn_weight_dtype': 'bf16',
        'linear_mlp_weight_dtype': 'bf16',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'bf16',
    },
    'bf16_fp8kv': {
        'name': 'BF16 + FP8 KV',
        'linear_attn_weight_dtype': 'bf16',
        'linear_mlp_weight_dtype': 'bf16',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'fp8',
    },
    'w8a16_bf16kv': {
        'name': 'W8A16 + BF16 KV',
        'linear_attn_weight_dtype': 'int8',
        'linear_mlp_weight_dtype': 'int8',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'bf16',
    },
    'w8a16_fp8kv': {
        'name': 'W8A16 + FP8 KV',
        'linear_attn_weight_dtype': 'int8',
        'linear_mlp_weight_dtype': 'int8',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'fp8',
    },
    'w4a16_bf16kv': {
        'name': 'W4A16 + BF16 KV',
        'linear_attn_weight_dtype': 'int4',
        'linear_mlp_weight_dtype': 'int4',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'bf16',
    },
    'w4a16_fp8kv': {
        'name': 'W4A16 + FP8 KV',
        'linear_attn_weight_dtype': 'int4',
        'linear_mlp_weight_dtype': 'int4',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'fp8',
    },
    'w8a8_bf16kv': {
        'name': 'W8A8 + BF16 KV',
        'linear_attn_weight_dtype': 'int8',
        'linear_mlp_weight_dtype': 'int8',
        'linear_attn_act_dtype': 'int8',
        'linear_mlp_act_dtype': 'int8',
        'kv_cache_dtype': 'bf16',
    },
    'w8a8_fp8kv': {
        'name': 'W8A8 + FP8 KV',
        'linear_attn_weight_dtype': 'int8',
        'linear_mlp_weight_dtype': 'int8',
        'linear_attn_act_dtype': 'int8',
        'linear_mlp_act_dtype': 'int8',
        'kv_cache_dtype': 'fp8',
    },
    'w4a8_bf16kv': {
        'name': 'W4A8(MLP A8) + W4A16(Attn A16) + BF16 KV',
        'linear_attn_weight_dtype': 'int4',
        'linear_mlp_weight_dtype': 'int4',
        # Pure W4A8 (int4 weight + int8 act) tends to severely hurt generation quality without calibration.
        # Minimal quality-first tweak: keep attention activation at bf16 (W4A16), while keeping MLP at int8 act (W4A8).
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'int8',
        'kv_cache_dtype': 'bf16',
    },
    'w4a8_fp8kv': {
        'name': 'W4A8(MLP A8) + W4A16(Attn A16) + FP8 KV',
        'linear_attn_weight_dtype': 'int4',
        'linear_mlp_weight_dtype': 'int4',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'int8',
        'kv_cache_dtype': 'fp8',
    },
    # FP8 W8A16 strategies
    'fp8_w8a16_e4m3_bf16kv': {
        'name': 'FP8 W8A16 (e4m3) + BF16 KV',
        'linear_attn_weight_dtype': 'fp8_e4m3',
        'linear_mlp_weight_dtype': 'fp8_e4m3',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'bf16',
    },
    'fp8_w8a16_e4m3_fp8kv': {
        'name': 'FP8 W8A16 (e4m3) + FP8 KV',
        'linear_attn_weight_dtype': 'fp8_e4m3',
        'linear_mlp_weight_dtype': 'fp8_e4m3',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'fp8',
    },
    'fp8_w8a16_e5m2_bf16kv': {
        'name': 'FP8 W8A16 (e5m2) + BF16 KV',
        'linear_attn_weight_dtype': 'fp8_e5m2',
        'linear_mlp_weight_dtype': 'fp8_e5m2',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'bf16',
    },
    'fp8_w8a16_e5m2_fp8kv': {
        'name': 'FP8 W8A16 (e5m2) + FP8 KV',
        'linear_attn_weight_dtype': 'fp8_e5m2',
        'linear_mlp_weight_dtype': 'fp8_e5m2',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'fp8',
    },
    # FP8 W8A8 strategies
    'fp8_w8a8_e4m3_bf16kv': {
        'name': 'FP8 W8A8 (e4m3) + BF16 KV',
        'linear_attn_weight_dtype': 'fp8_e4m3',
        'linear_mlp_weight_dtype': 'fp8_e4m3',
        'linear_attn_act_dtype': 'fp8_e4m3',
        'linear_mlp_act_dtype': 'fp8_e4m3',
        'kv_cache_dtype': 'bf16',
    },
    'fp8_w8a8_e4m3_fp8kv': {
        'name': 'FP8 W8A8 (e4m3) + FP8 KV',
        'linear_attn_weight_dtype': 'fp8_e4m3',
        'linear_mlp_weight_dtype': 'fp8_e4m3',
        'linear_attn_act_dtype': 'fp8_e4m3',
        'linear_mlp_act_dtype': 'fp8_e4m3',
        'kv_cache_dtype': 'fp8',
    },
    'fp8_w8a8_e5m2_bf16kv': {
        'name': 'FP8 W8A8 (e5m2) + BF16 KV',
        'linear_attn_weight_dtype': 'fp8_e5m2',
        'linear_mlp_weight_dtype': 'fp8_e5m2',
        'linear_attn_act_dtype': 'fp8_e5m2',
        'linear_mlp_act_dtype': 'fp8_e5m2',
        'kv_cache_dtype': 'bf16',
    },
    'fp8_w8a8_e5m2_fp8kv': {
        'name': 'FP8 W8A8 (e5m2) + FP8 KV',
        'linear_attn_weight_dtype': 'fp8_e5m2',
        'linear_mlp_weight_dtype': 'fp8_e5m2',
        'linear_attn_act_dtype': 'fp8_e5m2',
        'linear_mlp_act_dtype': 'fp8_e5m2',
        'kv_cache_dtype': 'fp8',
    },
    # GPTQ W4A16 strategies (offline quantized)
    'gptq_w4a16_bf16kv': {
        'name': 'GPTQ W4A16 (离线量化) + BF16 KV',
        'linear_attn_weight_dtype': 'gptq',
        'linear_mlp_weight_dtype': 'gptq',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'bf16',
    },
    'gptq_w4a16_fp8kv': {
        'name': 'GPTQ W4A16 (离线量化) + FP8 KV',
        'linear_attn_weight_dtype': 'gptq',
        'linear_mlp_weight_dtype': 'gptq',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'fp8',
    },
    # AWQ W4A16 strategies (offline quantized)
    'awq_w4a16_bf16kv': {
        'name': 'AWQ W4A16 (离线量化) + BF16 KV',
        'linear_attn_weight_dtype': 'awq',
        'linear_mlp_weight_dtype': 'awq',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'bf16',
    },
    'awq_w4a16_fp8kv': {
        'name': 'AWQ W4A16 (离线量化) + FP8 KV',
        'linear_attn_weight_dtype': 'awq',
        'linear_mlp_weight_dtype': 'awq',
        'linear_attn_act_dtype': 'bf16',
        'linear_mlp_act_dtype': 'bf16',
        'kv_cache_dtype': 'fp8',
    },
}

# 策略组定义
STRATEGY_GROUPS = {
    'bf16': ['bf16_bf16kv', 'bf16_fp8kv'],
    'w8a16': ['w8a16_bf16kv', 'w8a16_fp8kv'],
    'w4a16': ['w4a16_bf16kv', 'w4a16_fp8kv'],
    'w8a8': ['w8a8_bf16kv', 'w8a8_fp8kv'],
    'w4a8': ['w4a8_bf16kv', 'w4a8_fp8kv'],
    'fp8_w8a16': [
        'fp8_w8a16_e4m3_bf16kv',
        'fp8_w8a16_e4m3_fp8kv',
        'fp8_w8a16_e5m2_bf16kv',
        'fp8_w8a16_e5m2_fp8kv',
    ],
    'fp8_w8a8': [
        'fp8_w8a8_e4m3_bf16kv',
        'fp8_w8a8_e4m3_fp8kv',
        'fp8_w8a8_e5m2_bf16kv',
        'fp8_w8a8_e5m2_fp8kv',
    ],
    'gptq': [
        'gptq_w4a16_bf16kv',
        'gptq_w4a16_fp8kv',
    ],
    'awq': [
        'awq_w4a16_bf16kv',
        'awq_w4a16_fp8kv',
    ],
    'all': list(STRATEGY_CONFIGS.keys()),
}


def test_generation(
    llm: Diffulex,
    tokenizer: AutoTokenizer,
    test_name: str,
    prompts: List[str],
    warmup: bool = False,
    max_tokens: int = 30,
) -> Optional[Dict[str, float]]:
    """运行文本生成测试
    
    Args:
        llm: Diffulex 模型实例
        tokenizer: Tokenizer 实例
        test_name: 测试名称
        prompts: 输入 prompts 列表
        warmup: 如果为 True，只运行 warmup，不报告详细结果
        max_tokens: 最大生成 token 数
    
    Returns:
        如果是 warmup，返回 True/False
        如果不是 warmup，返回包含性能指标的字典，或 None（如果失败）
    """
    if not warmup:
        print("\n" + "=" * 70)
        print(f"测试: {test_name}")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"Warmup: {test_name} (排除 kernel 编译影响)")
        print("=" * 70)
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    
    # 添加 BOS token（如果需要）
    prompts_with_bos = []
    for p in prompts:
        if tokenizer.bos_token and not p.startswith(tokenizer.bos_token):
            prompts_with_bos.append(tokenizer.bos_token + p)
        else:
            prompts_with_bos.append(p)
    
    if not warmup:
        print(f"输入 prompts ({len(prompts_with_bos)} 个):")
        for i, p in enumerate(prompts_with_bos, 1):
            print(f"  {i}. {p[:60]}...")
        print(f"\n开始生成...")
    else:
        print(f"运行 warmup 生成（kernel 编译中，不报告速度）...")
    
    start_time = time.time()
    
    try:
        outputs = llm.generate(prompts_with_bos, sampling_params)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = sum(len(o.get('token_ids', [])) for o in outputs)
        
        if warmup:
            print(f"✓ Warmup 完成 (kernel 已编译，耗时 {total_time:.2f} 秒)")
            return True
        
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\n✓ 生成成功!")
        print(f"  - 总时间: {total_time:.2f} 秒")
        print(f"  - 总 token 数: {total_tokens}")
        print(f"  - 平均 TPS: {avg_tps:.2f} tok/s")
        
        print(f"\n生成结果:")
        for i, output in enumerate(outputs, 1):
            generated_text = output.get('text', '')
            token_ids = output.get('token_ids', [])
            print(f"\n  [{i}] 输入: {prompts[i-1][:50]}...")
            print(f"       输出: {generated_text[:150]}...")
            print(f"       Token数: {len(token_ids)}")
        
        return {
            'total_time': total_time,
            'total_tokens': total_tokens,
            'avg_tps': avg_tps,
        }
    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def _cleanup_llm(llm: Optional[Diffulex], force_cleanup: bool = False):
    """Best-effort cleanup to release GPU memory and NCCL resources even on exceptions.
    
    Args:
        llm: Diffulex instance to cleanup
        force_cleanup: If True, performs more aggressive cleanup including delays
    """
    try:
        if llm is not None:
            llm.exit()
    except Exception:
        pass
    
    try:
        import torch
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        if force_cleanup:
            # Force synchronization to ensure cleanup is complete
            torch.cuda.synchronize()
    except Exception:
        pass
    
    # Clear quantization strategy caches if available
    if force_cleanup:
        try:
            from diffulex.utils.quantization.context import get_quantization_context
            ctx = get_quantization_context()
            # QuantizationContext stores strategies in ctx._strategies (linear_attn/linear_mlp/linear_other/...).
            if hasattr(ctx, "_strategies") and isinstance(ctx._strategies, dict):
                for strategy in ctx._strategies.values():
                    if strategy is not None and hasattr(strategy, "_weight_cache"):
                        strategy._weight_cache.clear()
        except Exception:
            pass
    
    try:
        gc.collect()
        if force_cleanup:
            # Additional cleanup pass
            gc.collect()
    except Exception:
        pass
    
    if force_cleanup:
        # Small delay to allow resources to be released
        import time
        time.sleep(0.5)


def run_strategy(
    strategy_key: str,
    model_path: str,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    common_kwargs: Dict,
    max_tokens: int = 30,
) -> Tuple[str, Optional[Dict[str, float]]]:
    """运行单个策略的测试
    
    Returns:
        (strategy_name, result_dict) 或 (strategy_name, None) 如果失败
    """
    if strategy_key not in STRATEGY_CONFIGS:
        print(f"✗ 未知策略: {strategy_key}")
        return (strategy_key, None)
    
    config = STRATEGY_CONFIGS[strategy_key]
    strategy_name = config['name']
    is_w4a16 = 'w4a16' in strategy_key.lower()
    is_w4a8 = 'w4a8' in strategy_key.lower()
    is_gptq = 'gptq' in strategy_key.lower()
    is_awq = 'awq' in strategy_key.lower()
    needs_special_cleanup = is_w4a16 or is_w4a8 or is_gptq or is_awq  # W4A16/W4A8/GPTQ/AWQ may need extra cleanup
    
    print("\n" + "=" * 70)
    print(f"测试: {strategy_name}")
    print("=" * 70)
    
    # For W4A16/W4A8 strategies, add a delay before starting to ensure previous strategy is fully cleaned up
    if needs_special_cleanup:
        import time
        print("等待资源清理...")
        # Additional cleanup before W4A16/W4A8
        _cleanup_llm(None, force_cleanup=True)
        time.sleep(2.0)
    
    llm = None
    try:
        # 构建 Diffulex 配置
        llm_kwargs = {
            **common_kwargs,
            'kv_cache_dtype': config['kv_cache_dtype'],
            'kv_cache_layout': 'unified',  # FP8 kernel 只支持 unified layout
            'linear_attn_weight_dtype': config['linear_attn_weight_dtype'],
            'linear_mlp_weight_dtype': config['linear_mlp_weight_dtype'],
            'linear_attn_act_dtype': config['linear_attn_act_dtype'],
            'linear_mlp_act_dtype': config['linear_mlp_act_dtype'],
        }
        
        llm = Diffulex(model_path, **llm_kwargs)
        print(f"✓ {strategy_name} 模型初始化成功")
        
        # 第一轮：Warmup（排除 kernel 编译影响）
        test_generation(llm, tokenizer, strategy_name, prompts, warmup=True, max_tokens=max_tokens)
        
        # 第二轮：实际测试（kernel 已编译，看稳态性能）
        result = test_generation(llm, tokenizer, strategy_name, prompts, warmup=False, max_tokens=max_tokens)
        return (strategy_name, result)
        
    except Exception as e:
        print(f"✗ {strategy_name} 路径测试失败: {e}")
        import traceback
        traceback.print_exc()
        
        # For W4A16/W4A8 strategies, provide more detailed error information
        if needs_special_cleanup and 'shape' in str(e).lower():
            strategy_type = "W4A16/W4A8"
            print(f"\n提示: {strategy_type} 策略失败可能是由于资源清理不彻底导致的。")
            print("      建议:")
            print("      1. 单独运行测试脚本")
            print("      2. 或者增加策略之间的清理延迟时间")
        
        return (strategy_name, None)
    finally:
        # Use force_cleanup=True for W4A16/W4A8 strategies to ensure complete cleanup
        _cleanup_llm(llm, force_cleanup=needs_special_cleanup)
        llm = None
        # Additional cleanup delay for W4A16/W4A8 to ensure resources are fully released
        if needs_special_cleanup:
            import time
            time.sleep(2.0)  # Increased delay for W4A16/W4A8


def _run_strategy_in_subprocess(
    strategy_key: str,
    *,
    model_path: str,
    max_tokens: int,
    gpu_memory_utilization: float,
) -> Tuple[str, Optional[Dict[str, float]]]:
    """Run a single strategy in a fresh subprocess to avoid cross-strategy state (CUDA/NCCL/cache/fragmentation)."""
    cmd = [
        sys.executable,
        "-u",  # unbuffered stdout/stderr so parent can stream logs in real time
        str(Path(__file__).resolve()),
        "--strategies",
        strategy_key,
        "--max-tokens",
        str(max_tokens),
        "--model-path",
        model_path,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--_emit-json",
    ]
    # NOTE: don't use capture_output=True here, otherwise the parent appears to "hang"
    # during long model init/compilation because no logs are printed until the subprocess exits.
    print(f"\n[INFO] 启动子进程运行策略: {strategy_key}")
    # Ensure CUDA env is present *before Python starts* in the subprocess.
    # This matters because TileLang caches CUDA_HOME at import time (and can be imported very early).
    child_env = os.environ.copy()
    if _CUDA_12_2_PATH.exists():
        child_env["CUDA_HOME"] = str(_CUDA_12_2_PATH)
        child_env["CUDA_PATH"] = str(_CUDA_12_2_PATH)
        child_env["PATH"] = f"{_CUDA_12_2_PATH}/bin:{child_env.get('PATH', '')}"
        child_env["LD_LIBRARY_PATH"] = f"{_CUDA_12_2_PATH}/lib64:{child_env.get('LD_LIBRARY_PATH', '')}"
        child_env["LIBRARY_PATH"] = f"{_CUDA_12_2_PATH}/lib64:{child_env.get('LIBRARY_PATH', '')}"
        child_env["CPATH"] = f"{_CUDA_12_2_PATH}/include:{child_env.get('CPATH', '')}"
        child_env["CUDACXX"] = str(_CUDA_12_2_PATH / "bin" / "nvcc")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=child_env,
    )

    marker = "__RESULT_JSON__:"
    captured_lines: List[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # Stream logs live so the user can see progress.
            print(line, end="")
            captured_lines.append(line.rstrip("\n"))
    finally:
        # Ensure process termination is observed.
        returncode = proc.wait()

    # Parse the result marker from captured stdout.
    for line in reversed(captured_lines):
        if line.startswith(marker):
            payload = json.loads(line[len(marker):])
            return payload["strategy_name"], payload["result"]

    # If we can't find the marker, treat as failure.
    print(f"✗ 子进程未返回结果标记（strategy={strategy_key}, returncode={returncode}）")
    return STRATEGY_CONFIGS.get(strategy_key, {}).get("name", strategy_key), None


def print_summary(results: Dict[str, Dict[str, float]]):
    """打印汇总结果表格"""
    if not results:
        print("\n⚠ 没有成功完成的测试")
        return
    
    print("\n" + "=" * 90)
    print("性能汇总（第二轮，kernel 已编译）")
    print("=" * 90)
    print(f"{'策略':<25} {'总时间 (秒)':<15} {'总 Token 数':<15} {'平均 TPS (tok/s)':<20}")
    print("-" * 90)
    
    # 按策略名称排序
    sorted_results = sorted(results.items())
    for name, result in sorted_results:
        print(f"{name:<25} {result['total_time']:<15.2f} {result['total_tokens']:<15} {result['avg_tps']:<20.2f}")
    
    # 计算性能对比（如果有多个结果）
    if len(results) > 1:
        print("\n" + "-" * 90)
        print("性能对比（相对于第一个策略）:")
        print("-" * 90)
        
        baseline_name = sorted_results[0][0]
        baseline_result = sorted_results[0][1]
        baseline_tps = baseline_result['avg_tps']
        
        for name, result in sorted_results[1:]:
            tps_diff = ((result['avg_tps'] - baseline_tps) / baseline_tps) * 100
            time_diff = ((result['total_time'] - baseline_result['total_time']) / baseline_result['total_time']) * 100
            
            tps_indicator = "↑" if tps_diff > 0 else "↓" if tps_diff < 0 else "≈"
            time_indicator = "↓" if time_diff < 0 else "↑" if time_diff > 0 else "≈"
            
            print(f"  {name:<25} TPS: {tps_diff:+.1f}% {tps_indicator}  时间: {time_diff:+.1f}% {time_indicator}")


def parse_strategies(args) -> List[str]:
    """解析命令行参数，返回要运行的策略列表"""
    strategies = []
    
    if args.all:
        strategies = STRATEGY_GROUPS['all']
    elif args.bf16:
        strategies = STRATEGY_GROUPS['bf16']
    elif args.w8a16:
        strategies = STRATEGY_GROUPS['w8a16']
    elif args.w4a16:
        strategies = STRATEGY_GROUPS['w4a16']
    elif args.w8a8:
        strategies = STRATEGY_GROUPS['w8a8']
    elif args.w4a8:
        strategies = STRATEGY_GROUPS['w4a8']
    elif args.fp8_w8a16:
        strategies = STRATEGY_GROUPS['fp8_w8a16']
    elif args.fp8_w8a8:
        strategies = STRATEGY_GROUPS['fp8_w8a8']
    elif args.gptq:
        strategies = STRATEGY_GROUPS['gptq']
    elif args.awq:
        strategies = STRATEGY_GROUPS['awq']
    elif args.strategies:
        # 手动指定策略，支持逗号分隔
        strategies = [s.strip() for s in args.strategies.split(',')]
        # 验证策略是否有效
        invalid = [s for s in strategies if s not in STRATEGY_CONFIGS]
        if invalid:
            print(f"✗ 无效的策略: {invalid}")
            print(f"  支持的策略: {', '.join(STRATEGY_CONFIGS.keys())}")
            sys.exit(1)
    else:
        # 默认运行所有策略
        print("未指定策略，默认运行所有策略（使用 --all 显式指定）")
        strategies = STRATEGY_GROUPS['all']
    
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description='Diffulex 量化策略文本生成测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --all                    # 运行所有策略
  %(prog)s --bf16                   # 只运行 BF16 相关策略
  %(prog)s --w8a16                  # 只运行 W8A16 相关策略
  %(prog)s --w4a16                  # 只运行 W4A16 相关策略
  %(prog)s --w8a8                   # 只运行 W8A8 相关策略
  %(prog)s --w4a8                   # 只运行 W4A8 相关策略
  %(prog)s --fp8_w8a16              # 只运行 FP8 W8A16 相关策略
  %(prog)s --fp8_w8a8               # 只运行 FP8 W8A8 相关策略
  %(prog)s --gptq                   # 只运行 GPTQ W4A16 相关策略（需要先运行量化脚本）
  %(prog)s --awq                    # 只运行 AWQ W4A16 相关策略（需要先运行量化脚本）
  %(prog)s --strategies bf16_bf16kv,w8a16_bf16kv  # 自定义选择
  %(prog)s --strategies w4a16_fp8kv --max-tokens 50  # 指定策略和参数
  %(prog)s --gptq --model-path /path/to/quantized/model  # 使用量化后的模型路径
        """
    )
    
    # 策略选择选项（互斥）
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument('--all', action='store_true', help='运行所有策略')
    strategy_group.add_argument('--bf16', action='store_true', help='只运行 BF16 相关策略')
    strategy_group.add_argument('--w8a16', action='store_true', help='只运行 W8A16 相关策略')
    strategy_group.add_argument('--w4a16', action='store_true', help='只运行 W4A16 相关策略')
    strategy_group.add_argument('--w8a8', action='store_true', help='只运行 W8A8 相关策略')
    strategy_group.add_argument('--w4a8', action='store_true', help='只运行 W4A8 相关策略')
    strategy_group.add_argument('--fp8_w8a16', action='store_true', help='只运行 FP8 W8A16 相关策略')
    strategy_group.add_argument('--fp8_w8a8', action='store_true', help='只运行 FP8 W8A8 相关策略')
    strategy_group.add_argument('--gptq', action='store_true', help='只运行 GPTQ W4A16 相关策略（需要先运行量化脚本生成离线权重）')
    strategy_group.add_argument('--awq', action='store_true', help='只运行 AWQ W4A16 相关策略（需要先运行量化脚本生成离线权重）')
    strategy_group.add_argument('--strategies', type=str, help='手动指定策略（逗号分隔），例如: bf16_bf16kv,w8a16_fp8kv')
    
    # 其他选项
    parser.add_argument('--max-tokens', type=int, default=30, help='最大生成 token 数（默认: 30）')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径（必填）')
    parser.add_argument('--lora-path', type=str, default="", help='LoRA 路径（可选）')
    parser.add_argument('--use-lora', action='store_true', help='启用 LoRA（需同时提供 --lora-path）')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.3, help='GPU 内存利用率（默认: 0.3）')
    parser.add_argument('--no-isolate', action='store_true', help='多策略运行时不使用子进程隔离（调试用，可能导致状态串扰/性能波动）')
    # Internal: emit a single JSON result line for parent process parsing.
    parser.add_argument('--_emit-json', action='store_true', help=argparse.SUPPRESS)
    
    args = parser.parse_args()
    
    # 确定模型路径
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请使用 --model-path 指向有效的模型路径")
        return
    
    # 解析要运行的策略
    strategies = parse_strategies(args)
    
    print("=" * 90)
    print("Diffulex 量化策略文本生成测试")
    print("=" * 90)
    print(f"模型路径: {model_path}")
    print(f"要测试的策略 ({len(strategies)} 个): {', '.join(STRATEGY_CONFIGS[s]['name'] for s in strategies)}")
    print(f"最大生成 token 数: {args.max_tokens}")
    print("=" * 90)
    
    # 测试 prompts (10个样例)
    test_prompts = [
        "The capital of France is",
        "Python is a programming language",
        "The largest planet in our solar system is",
        "Machine learning is a subset of",
        "The speed of light is approximately",
        "Artificial intelligence has applications in",
        "The Great Wall of China was built",
        "Quantum computing uses principles from",
        "The human brain contains approximately",
        "Climate change is caused by",
    ]
    
    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        return
    
    # 通用 Diffulex 配置
    common_kwargs = {
        'lora_path': args.lora_path,
        'use_lora': bool(args.use_lora and args.lora_path),
        'model_name': 'dream',
        'enforce_eager': True,
        'data_parallel_size': 1,
        'tensor_parallel_size': 1,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_num_batched_tokens': 1024,
        'max_num_seqs': 4,
        'max_model_len': 1024,
        'decoding_strategy': 'd2f',
        'decode_mode': 'varlen',  # 统一设置为 varlen 模式
    }
    
    # 运行所有选定的策略
    # 对于 W4A16/W4A8/GPTQ/AWQ 策略，调整运行顺序：先运行其他策略，再运行这些策略
    # 这样可以避免在运行其他策略后资源状态不一致导致的问题
    w4a16_strategies = [s for s in strategies if 'w4a16' in s.lower() and 'gptq' not in s.lower() and 'awq' not in s.lower()]
    w4a8_strategies = [s for s in strategies if 'w4a8' in s.lower()]
    gptq_strategies = [s for s in strategies if 'gptq' in s.lower()]
    awq_strategies = [s for s in strategies if 'awq' in s.lower()]
    other_strategies = [s for s in strategies if 'w4a16' not in s.lower() and 'w4a8' not in s.lower() and 'gptq' not in s.lower() and 'awq' not in s.lower()]
    # 先运行其他策略，再运行 W4A16 策略，然后 W4A8，最后 GPTQ/AWQ 策略（如果存在）
    ordered_strategies = other_strategies + w4a16_strategies + w4a8_strategies + gptq_strategies + awq_strategies
    
    results = {}
    isolate = (len(ordered_strategies) > 1) and (not args.no_isolate) and (not args._emit_json)
    for strategy_key in ordered_strategies:
        if isolate:
            strategy_name, result = _run_strategy_in_subprocess(
                strategy_key,
                model_path=model_path,
                max_tokens=args.max_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
        else:
            strategy_name, result = run_strategy(
                strategy_key,
                model_path,
                tokenizer,
                test_prompts,
                common_kwargs,
                max_tokens=args.max_tokens,
            )
        if result:
            results[strategy_name] = result
    
    # 打印汇总结果
    if args._emit_json:
        # In emit-json mode we should have exactly one strategy; return it as a single machine-readable line.
        # If multiple are present for any reason, pick the first.
        if results:
            name, result = next(iter(results.items()))
            print("__RESULT_JSON__:" + json.dumps({"strategy_name": name, "result": result}, ensure_ascii=False))
        else:
            # Fallback: map key to display name if possible
            only_key = ordered_strategies[0] if ordered_strategies else "unknown"
            only_name = STRATEGY_CONFIGS.get(only_key, {}).get("name", only_key)
            print("__RESULT_JSON__:" + json.dumps({"strategy_name": only_name, "result": None}, ensure_ascii=False))
        return

    print_summary(results)
    
    print("\n" + "=" * 90)
    print("测试完成")
    print("=" * 90)


if __name__ == "__main__":
    main()

