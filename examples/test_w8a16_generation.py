#!/usr/bin/env python3
"""测试 W8A16 Linear 量化策略的文本生成"""
import os
import sys
import time
from pathlib import Path

# 确保从当前仓库导入
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer
from diffulex import Diffulex, SamplingParams


def test_generation(llm, tokenizer, test_name: str, prompts: list[str], warmup: bool = False):
    """运行文本生成测试
    
    Args:
        llm: Diffulex 模型实例
        tokenizer: Tokenizer 实例
        test_name: 测试名称
        prompts: 输入 prompts 列表
        warmup: 如果为 True，只运行 warmup，不报告详细结果
    
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
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    
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


def main():
    # 检查模型路径
    model_path = os.getenv("DIFFULEX_TEST_MODEL", "/data1/ckpts/Dream-org/Dream-v0-Base-7B")
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请设置环境变量 DIFFULEX_TEST_MODEL 指向有效的模型路径")
        return
    
    print("=" * 70)
    print("Diffulex W8A16 Linear 量化文本生成测试")
    print("=" * 70)
    print(f"模型路径: {model_path}")
    
    # 测试 prompts
    test_prompts = [
        "The capital of France is",
        "Python is a programming language",
    ]
    
    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        return
    
    # 存储性能结果用于对比
    results = {}
    
    # 测试 1: W8A16 Linear + BF16 KV
    print("\n" + "=" * 70)
    print("测试 1: W8A16 Linear + BF16 KV Cache")
    print("=" * 70)
    
    try:
        llm_w8a16_bf16kv = Diffulex(
            model_path,
            lora_path=os.getenv("DIFFULEX_TEST_LORA", ""),
            use_lora=bool(os.getenv("DIFFULEX_TEST_LORA", "")),
            model_name="dream",
            enforce_eager=True,
            data_parallel_size=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            max_model_len=1024,
            kv_cache_dtype="bf16",
            kv_cache_layout="unified",
            decoding_strategy="d2f",
            # W8A16 配置
            linear_attn_weight_dtype="int8",
            linear_mlp_weight_dtype="int8",
            linear_attn_act_dtype="bf16",
            linear_mlp_act_dtype="bf16",
        )
        print("✓ W8A16 + BF16 KV 模型初始化成功")
        
        # 第一轮：Warmup（排除 kernel 编译影响）
        test_generation(llm_w8a16_bf16kv, tokenizer, "W8A16 Linear + BF16 KV", test_prompts, warmup=True)
        
        # 第二轮：实际测试（kernel 已编译，看稳态性能）
        result = test_generation(llm_w8a16_bf16kv, tokenizer, "W8A16 Linear + BF16 KV", test_prompts, warmup=False)
        if result:
            results['W8A16+BF16KV'] = result
        
        # 清理
        llm_w8a16_bf16kv.exit()
        del llm_w8a16_bf16kv
        import torch
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ W8A16 + BF16 KV 路径测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 2: W8A16 Linear + FP8 KV
    print("\n" + "=" * 70)
    print("测试 2: W8A16 Linear + FP8 KV Cache")
    print("=" * 70)
    
    try:
        llm_w8a16_fp8kv = Diffulex(
            model_path,
            lora_path=os.getenv("DIFFULEX_TEST_LORA", ""),
            use_lora=bool(os.getenv("DIFFULEX_TEST_LORA", "")),
            model_name="dream",
            enforce_eager=True,
            data_parallel_size=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            max_model_len=1024,
            kv_cache_dtype="fp8",  # FP8 KV cache
            kv_cache_layout="unified",  # FP8 kernel 只支持 unified layout
            decoding_strategy="d2f",
            # W8A16 配置
            linear_attn_weight_dtype="int8",
            linear_mlp_weight_dtype="int8",
            linear_attn_act_dtype="bf16",
            linear_mlp_act_dtype="bf16",
        )
        print("✓ W8A16 + FP8 KV 模型初始化成功")
        
        # 第一轮：Warmup（排除 kernel 编译影响）
        test_generation(llm_w8a16_fp8kv, tokenizer, "W8A16 Linear + FP8 KV", test_prompts, warmup=True)
        
        # 第二轮：实际测试（kernel 已编译，看稳态性能）
        result = test_generation(llm_w8a16_fp8kv, tokenizer, "W8A16 Linear + FP8 KV", test_prompts, warmup=False)
        if result:
            results['W8A16+FP8KV'] = result
        
        # 清理
        llm_w8a16_fp8kv.exit()
        del llm_w8a16_fp8kv
        import torch
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ W8A16 + FP8 KV 路径测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 性能对比
    if len(results) == 2:
        print("\n" + "=" * 70)
        print("性能对比（第二轮，kernel 已编译）")
        print("=" * 70)
        print(f"{'配置':<20} {'总时间 (秒)':<15} {'总 Token 数':<15} {'平均 TPS (tok/s)':<20}")
        print("-" * 70)
        for name, result in results.items():
            print(f"{name:<20} {result['total_time']:<15.2f} {result['total_tokens']:<15} {result['avg_tps']:<20.2f}")
        
        # 计算性能差异
        bf16kv_result = results.get('W8A16+BF16KV')
        fp8kv_result = results.get('W8A16+FP8KV')
        if bf16kv_result and fp8kv_result:
            tps_diff = ((fp8kv_result['avg_tps'] - bf16kv_result['avg_tps']) / bf16kv_result['avg_tps']) * 100
            time_diff = ((fp8kv_result['total_time'] - bf16kv_result['total_time']) / bf16kv_result['total_time']) * 100
            
            print("\n性能差异:")
            if tps_diff > 0:
                print(f"  ✓ FP8 KV 路径更快: TPS 提升 {tps_diff:.1f}%, 时间减少 {abs(time_diff):.1f}%")
            elif tps_diff < 0:
                print(f"  ⚠ BF16 KV 路径更快: TPS 高 {abs(tps_diff):.1f}%, 时间少 {abs(time_diff):.1f}%")
            else:
                print(f"  ≈ 两种路径性能相近")
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

