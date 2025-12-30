#!/usr/bin/env python3
"""简单的端到端测试脚本，验证 BF16 kernel 功能"""
import os
import time

from transformers import AutoTokenizer
from diffulex import Diffulex, SamplingParams


def main():
    # 模型配置
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    
    print("=" * 60)
    print("初始化 Diffulex 模型...")
    print("=" * 60)
    
    llm = Diffulex(
        model,
        lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream",
        enforce_eager=True,
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=2048,
        max_num_seqs=10,
        max_model_len=2048,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
        decoding_strategy="d2f"
    )
    
    print("✓ 模型初始化完成\n")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)
    
    # 定义几个测试 prompt
    test_prompts = [
        "The capital of France is",
        "1 + 1 equals",
        "Python is a programming language that",
    ]
    
    # 添加 BOS token
    prompts = [tokenizer.bos_token + p for p in test_prompts]
    
    print("=" * 60)
    print(f"运行生成测试 ({len(prompts)} 个 prompt)...")
    print("=" * 60)
    
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    
    print("\n" + "=" * 60)
    print("生成结果:")
    print("=" * 60)
    
    total_tokens = sum(len(o['token_ids']) for o in outputs)
    total_time = end_time - start_time
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    avg_diff_steps = sum(o['n_diff_steps'] for o in outputs) / len(outputs) if outputs else 0
    
    print(f"\n总计:")
    print(f"  - 生成输出数: {len(outputs)}")
    print(f"  - 总 token 数: {total_tokens}")
    print(f"  - 总时间: {total_time:.2f} 秒")
    print(f"  - 平均 TPS: {avg_tps:.2f} tok/s")
    print(f"  - 平均扩散步数: {avg_diff_steps:.2f}")
    
    print("\n" + "=" * 60)
    print("详细输出:")
    print("=" * 60)
    
    for idx, (prompt, output) in enumerate(zip(test_prompts, outputs)):
        print(f"\n[Prompt {idx + 1}]")
        print(f"输入: {prompt}")
        print(f"输出: {output['text']}")
        print(f"Token IDs 长度: {len(output['token_ids'])}")
        print(f"扩散步数: {output['n_diff_steps']}")
        print("-" * 60)
    
    print("\n✓ BF16 kernel 端到端测试完成！")


if __name__ == "__main__":
    main()

