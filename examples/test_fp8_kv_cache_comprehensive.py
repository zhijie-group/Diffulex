#!/usr/bin/env python3
"""
FP8 KV Cache 综合测试脚本

该脚本整合了所有 FP8 KV Cache 相关的测试，可以通过命令行参数选择运行哪些测试。

测试类别：
1. kernel - Kernel 层 FP8 支持测试（roundtrip tests）
2. integration - Attention layer FP8 集成测试
3. pipeline - 完整 pipeline 测试（需要模型）
4. memory - 内存使用验证测试（需要模型）
5. speed - 速度对比测试（需要模型）
6. quality - 质量和速度对比测试（需要模型，较耗时）
7. attention_kernel - FP8 attention kernel 单元测试
8. attention_e2e - FP8 attention kernel 端到端测试（需要模型）
9. attention_numerics - FP8 attention kernel 数值验证测试
10. all - 运行所有测试（不包括 quality 和 root_cause，因为需要较长时间）

用法示例：
  # 运行所有测试（除了 quality）
  python test_fp8_kv_cache_comprehensive.py --tests all

  # 运行特定测试
  python test_fp8_kv_cache_comprehensive.py --tests kernel integration

  # 运行 speed 和 quality 测试
  python test_fp8_kv_cache_comprehensive.py --tests speed quality
"""

import os
import sys
import argparse
import traceback
import torch
import time
import gc
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from einops import rearrange

# 添加项目路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# 导入必要的模块
from vllm.platforms import current_platform
from diffulex_legacy.layers.attention.ops import (
    store_kvcache_unified_layout, 
    store_kvcache_distinct_layout, 
    load_kvcache
)
from diffulex_legacy.layers.attention.attention_v4 import Attention
from diffulex_legacy.utils.context import (
    set_context_diffusion_lm, 
    get_context_diffusion_lm, 
    ContextForDiffusionLM,
    ContextForCausalLM,
    set_context_causal_lm,
    get_context_causal_lm
)
from diffulex_legacy.config import Config
from diffulex_legacy import LLM, SamplingParams
from diffulex.utils.kv_cache_dtype import parse_kv_cache_dtype
from diffulex_legacy.layers.attention.ops.triton_flash_attention import triton_flash_attention
from transformers import AutoTokenizer


# ============================================================================
# 测试辅助函数和类
# ============================================================================

@dataclass
class _Seq:
    diffusion_block_size: int = 32


@dataclass
class _Ctx:
    seq_lens_ts: torch.Tensor
    context_lens: torch.Tensor
    total_lens: torch.Tensor
    block_tables: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: List[int] = None
    seqs: List[_Seq] = None

    def __post_init__(self):
        self.seq_lens = self.seq_lens_ts.tolist()
        self.seqs = [_Seq()]


def _build_cu_seqlens(x: torch.Tensor) -> torch.Tensor:
    return torch.tensor(
        [0] + list(torch.cumsum(x, dim=0).cpu().numpy()),
        dtype=torch.int32,
        device="cuda",
    )


def get_gpu_memory_info():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
        }
    return None


# ============================================================================
# 测试函数 - Kernel 层测试
# ============================================================================

def test_kv_cache_fp8_unified_roundtrip():
    """测试 FP8 unified layout KV cache 的存储和加载往返"""
    torch.random.manual_seed(114514)

    num_seqs = 4
    blk_sz = 256
    H = 4
    head_dim = 128

    seq_lens = torch.tensor([64, 32, 64, 32], dtype=torch.int32, device="cuda")
    ctx_lens = torch.tensor([119, 110, 81, 114], dtype=torch.int32, device="cuda")
    assert seq_lens.numel() == num_seqs and ctx_lens.numel() == num_seqs
    total_lens = seq_lens + ctx_lens

    kv_shape = (int(total_lens.sum().item()), H, head_dim)
    k_all = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    v_all = torch.randn_like(k_all)

    slot_mapping: list[int] = []
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        slot_mapping.extend(list(range(seq_idx * blk_sz, seq_idx * blk_sz + ctx)))
        slot_mapping.extend([-1] * new)
        start += ctx + new
    slot_mapping_ts = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")
    assert slot_mapping_ts.numel() == kv_shape[0]

    kv_cache_shape = (num_seqs, blk_sz, H, head_dim)
    k_cache_u8 = torch.zeros(kv_cache_shape, device="cuda", dtype=torch.uint8)
    v_cache_u8 = torch.zeros_like(k_cache_u8)

    fp8 = current_platform.fp8_dtype()
    fp8_max = float(torch.finfo(fp8).max)
    eps = 1e-6
    k_absmax = k_all.to(torch.float32).abs().amax(dim=(0, 2))
    v_absmax = v_all.to(torch.float32).abs().amax(dim=(0, 2))
    k_scale = (k_absmax / fp8_max).clamp_min(eps)
    v_scale = (v_absmax / fp8_max).clamp_min(eps)

    store_kvcache_unified_layout(
        k_all, v_all, k_cache_u8, v_cache_u8, slot_mapping_ts,
        model_type="diffusion_lm",
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    k_cache_fp8 = k_cache_u8.view(fp8).to(torch.float32) * k_scale[None, None, :, None]
    v_cache_fp8 = v_cache_u8.view(fp8).to(torch.float32) * v_scale[None, None, :, None]
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        k_ctx_ref = k_all[start : start + ctx].to(torch.float32)
        v_ctx_ref = v_all[start : start + ctx].to(torch.float32)
        k_ctx_got = k_cache_fp8[seq_idx, :ctx]
        v_ctx_got = v_cache_fp8[seq_idx, :ctx]
        assert torch.allclose(k_ctx_got, k_ctx_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(v_ctx_got, v_ctx_ref, atol=1e-1, rtol=1e-1)
        start += ctx + new

    k_new_list = []
    v_new_list = []
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        k_new_list.append(k_all[start + ctx : start + ctx + new])
        v_new_list.append(v_all[start + ctx : start + ctx + new])
        start += ctx + new
    k_new = torch.cat(k_new_list, dim=0).contiguous()
    v_new = torch.cat(v_new_list, dim=0).contiguous()

    block_tables = torch.arange(num_seqs, dtype=torch.int32, device="cuda").view(num_seqs, 1)
    cu_seqlens_q = _build_cu_seqlens(seq_lens)
    cu_seqlens_k = _build_cu_seqlens(total_lens)
    ctx = _Ctx(
        seq_lens_ts=seq_lens,
        context_lens=ctx_lens,
        total_lens=total_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
    )

    k_out, v_out = load_kvcache(
        k_cache_u8, v_cache_u8, ctx, k_new, v_new,
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    out_splits = torch.split(k_out, total_lens.tolist(), dim=0)
    new_splits = torch.split(k_new, seq_lens.tolist(), dim=0)
    start = 0
    for seq_idx in range(num_seqs):
        ctx_len = int(ctx_lens[seq_idx].item())
        new_len = int(seq_lens[seq_idx].item())
        k_ref_ctx = k_all[start : start + ctx_len].to(k_out.dtype)
        k_got_ctx = out_splits[seq_idx][:ctx_len]
        assert torch.allclose(k_got_ctx, k_ref_ctx, atol=1e-1, rtol=1e-1)
        assert torch.equal(out_splits[seq_idx][ctx_len : ctx_len + new_len], new_splits[seq_idx])
        start += ctx_len + new_len

    print("FP8 unified KV cache store/load roundtrip: OK")


def test_kv_cache_fp8_distinct_roundtrip():
    """测试 FP8 distinct layout KV cache 的存储"""
    torch.random.manual_seed(114514)

    num_seqs = 4
    blk_sz = 256
    H = 4
    head_dim = 128
    x = 8

    seq_lens = torch.tensor([64, 32, 64, 32], dtype=torch.int32, device="cuda")
    ctx_lens = torch.tensor([119, 110, 81, 114], dtype=torch.int32, device="cuda")
    total_lens = seq_lens + ctx_lens

    kv_shape = (int(total_lens.sum().item()), H, head_dim)
    k_all = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
    v_all = torch.randn_like(k_all)

    slot_mapping: list[int] = []
    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        slot_mapping.extend(list(range(seq_idx * blk_sz, seq_idx * blk_sz + ctx)))
        slot_mapping.extend([-1] * new)
        start += ctx + new
    slot_mapping_ts = torch.tensor(slot_mapping, dtype=torch.int64, device="cuda")

    k_cache_u8 = torch.zeros((num_seqs, H, head_dim // x, blk_sz, x), device="cuda", dtype=torch.uint8)
    v_cache_u8 = torch.zeros((num_seqs, H, head_dim, blk_sz), device="cuda", dtype=torch.uint8)

    fp8 = current_platform.fp8_dtype()
    fp8_max = float(torch.finfo(fp8).max)
    eps = 1e-6
    k_absmax = k_all.to(torch.float32).abs().amax(dim=(0, 2))
    v_absmax = v_all.to(torch.float32).abs().amax(dim=(0, 2))
    k_scale = (k_absmax / fp8_max).clamp_min(eps)
    v_scale = (v_absmax / fp8_max).clamp_min(eps)

    store_kvcache_distinct_layout(
        k_all, v_all, k_cache_u8, v_cache_u8, slot_mapping_ts,
        model_type="diffusion_lm",
        kv_cache_dtype="fp8_e4m3",
        k_scale=k_scale,
        v_scale=v_scale,
    )

    k_cache_fp8 = k_cache_u8.view(fp8).to(torch.float32)
    v_cache_fp8 = v_cache_u8.view(fp8).to(torch.float32)
    k_cache_deq = k_cache_fp8 * k_scale[None, :, None, None, None]
    v_cache_deq = v_cache_fp8 * v_scale[None, :, None, None]
    k_cache_unified = rearrange(k_cache_deq, "b h n s x -> b s h (n x)").contiguous()
    v_cache_unified = rearrange(v_cache_deq, "b h d s -> b s h d").contiguous()

    start = 0
    for seq_idx in range(num_seqs):
        ctx = int(ctx_lens[seq_idx].item())
        new = int(seq_lens[seq_idx].item())
        k_ctx_ref = k_all[start : start + ctx].to(torch.float32)
        v_ctx_ref = v_all[start : start + ctx].to(torch.float32)
        assert torch.allclose(k_cache_unified[seq_idx, :ctx], k_ctx_ref, atol=1e-1, rtol=1e-1)
        assert torch.allclose(v_cache_unified[seq_idx, :ctx], v_ctx_ref, atol=1e-1, rtol=1e-1)
        start += ctx + new

    print("FP8 distinct KV cache store roundtrip (ctx portion): OK")


# ============================================================================
# 测试函数 - Integration 测试
# ============================================================================

def test_running_max_update(attn: Attention):
    """Test running max update in FP8 scale computation."""
    num_heads = 8
    num_kv_heads = 4
    head_dim = 128
    seq_len = 64
    
    device = 'cuda'
    k1 = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 0.5
    v1 = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 0.5
    
    kv_cache_dtype = "fp8_e4m3"
    k_scale1, v_scale1 = attn._update_and_compute_fp8_scales(k1, v1, kv_cache_dtype, device)
    
    assert k_scale1 is not None and v_scale1 is not None
    assert attn.k_max_abs is not None and attn.v_max_abs is not None
    
    k2 = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 1.5
    v2 = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * 1.5
    
    k_max_abs_before = attn.k_max_abs.clone()
    v_max_abs_before = attn.v_max_abs.clone()
    
    k_scale2, v_scale2 = attn._update_and_compute_fp8_scales(k2, v2, kv_cache_dtype, device)
    
    assert torch.all(attn.k_max_abs >= k_max_abs_before)
    assert torch.all(attn.v_max_abs >= v_max_abs_before)
    
    k_scale3, v_scale3 = attn._update_and_compute_fp8_scales(k1, v1, "bf16", device)
    assert k_scale3 is None and v_scale3 is None
    
    k_scale4, v_scale4 = attn._update_and_compute_fp8_scales(k1, v1, "fp8_e5m2", device)
    assert attn.kv_cache_dtype_cache == "fp8_e5m2"


def test_scale_computation(attn: Attention):
    """Test scale computation from running max."""
    device = 'cuda'
    seq_len = 64
    num_kv_heads = 4
    head_dim = 128
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device)
    v = torch.randn(seq_len, num_kv_heads, head_dim, device=device)
    
    kv_cache_dtype = "fp8_e4m3"
    k_scale, v_scale = attn._update_and_compute_fp8_scales(k, v, kv_cache_dtype, device)
    
    assert k_scale.shape == (num_kv_heads,)
    assert v_scale.shape == (num_kv_heads,)
    assert torch.all(k_scale > 0)
    assert torch.all(v_scale > 0)
    
    k_scale2, v_scale2 = attn._get_fp8_scales_from_max(kv_cache_dtype)
    assert k_scale2 is not None and v_scale2 is not None
    assert torch.allclose(k_scale, k_scale2)
    assert torch.allclose(v_scale, v_scale2)
    
    k_scale3, v_scale3 = attn._get_fp8_scales_from_max("bf16")
    assert k_scale3 is None and v_scale3 is None


def test_context_kv_cache_dtype():
    """Test context kv_cache_dtype access."""
    ctx_causal = ContextForCausalLM()
    assert ctx_causal.kv_cache_dtype == "bf16"
    
    set_context_causal_lm(True, kv_cache_dtype="fp8_e4m3")
    ctx_causal2 = get_context_causal_lm()
    assert ctx_causal2.kv_cache_dtype == "fp8_e4m3"
    
    from diffulex_legacy.layers.attention.attention_v4 import _get_kv_cache_dtype
    
    class MockConfig:
        kv_cache_dtype = "fp8_e4m3"
    
    class MockSeq:
        def __init__(self):
            self.config = MockConfig()
    
    ctx_diff = ContextForDiffusionLM.__new__(ContextForDiffusionLM)
    ctx_diff.seqs = [MockSeq()]
    ctx_diff.seq_lens = None
    ctx_diff.seq_lens_ts = None
    ctx_diff.kv_cache_layout = "unified"
    ctx_diff.need_kv_cache_store = True
    ctx_diff.d2f_pp = False
    ctx_diff.block_mask = None
    ctx_diff.is_prefill = False
    ctx_diff.cu_seqlens_q = None
    ctx_diff.cu_seqlens_k = None
    ctx_diff.max_seqlen_q = 0
    ctx_diff.max_seqlen_k = 0
    ctx_diff.slot_mapping = None
    ctx_diff.context_lens = None
    ctx_diff.block_tables = None
    
    dtype1 = _get_kv_cache_dtype(ctx_diff, "diffusion_lm")
    assert dtype1 == "fp8_e4m3"
    
    dtype2 = _get_kv_cache_dtype(ctx_causal2, "causal_lm")
    assert dtype2 == "fp8_e4m3"


# ============================================================================
# 测试函数 - Pipeline 测试
# ============================================================================

def test_fp8_kv_cache_pipeline():
    """Test FP8 KV cache in a complete inference pipeline."""
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    
    llm = LLM(
        model,
        lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream", 
        model_type="diffusion_lm",
        enforce_eager=True, 
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=2048,
        max_num_seqs=20,
        max_model_len=2048,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
        kv_cache_dtype="fp8_e4m3",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    test_prompts = [
        tokenizer.bos_token + "Hello, how are you?",
        tokenizer.bos_token + "The capital of France is",
        tokenizer.bos_token + "Python is a programming language that",
    ]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    outputs = llm.generate(test_prompts, sampling_params)
    
    for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
        generated_text = output.get("text", "")
        token_ids = output.get("token_ids", [])
        
        if not generated_text.strip():
            raise ValueError(f"Generated text is empty for prompt {i+1}")
        if len(token_ids) == 0:
            raise ValueError(f"No tokens generated for prompt {i+1}")


# ============================================================================
# 测试函数 - Memory 测试
# ============================================================================

def test_kv_cache_memory(kv_cache_dtype="bf16"):
    """Test KV cache memory usage with specified dtype."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    
    llm = LLM(
        model,
        lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream", 
        model_type="diffusion_lm",
        enforce_eager=True, 
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=2048,
        max_num_seqs=20,
        max_model_len=2048,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
        kv_cache_dtype=kv_cache_dtype,
    )
    
    model_runner = llm.model_runner
    if hasattr(model_runner, 'kv_cache') and model_runner.kv_cache is not None:
        kv_cache = model_runner.kv_cache
        kv_cache_size_mb = kv_cache.element_size() * kv_cache.numel() / 1024**2
        
        config = model_runner.config
        if hasattr(config, 'num_kvcache_blocks') and config.num_kvcache_blocks > 0:
            hf_config = config.hf_config
            num_layers = hf_config.num_hidden_layers
            block_size = config.kvcache_block_size
            num_blocks = config.num_kvcache_blocks
            
            if hasattr(hf_config, 'head_dim'):
                head_dim = hf_config.head_dim
            elif hasattr(hf_config, 'hidden_size') and hasattr(hf_config, 'num_attention_heads'):
                head_dim = hf_config.hidden_size // hf_config.num_attention_heads
            else:
                head_dim = 128
            
            num_kv_heads = getattr(hf_config, 'num_key_value_heads', getattr(hf_config, 'num_attention_heads', 32))
            
            spec = parse_kv_cache_dtype(kv_cache_dtype)
            itemsize = 1 if spec.is_fp8 else (2 if kv_cache_dtype in ['bf16', 'fp16'] else 4)
            
            elements_per_block = 2 * num_layers * block_size * num_kv_heads * head_dim
            size_per_block_mb = elements_per_block * itemsize / 1024**2
            
            print(f"  num_blocks: {num_blocks}")
            print(f"  Size per block: {size_per_block_mb:.2f} MB")
            print(f"  Total size: {kv_cache_size_mb:.2f} MB")


# ============================================================================
# 测试函数 - Speed 测试
# ============================================================================

def test_kv_cache_speed(kv_cache_dtype="bf16", num_prompts=3):
    """Test generation speed with specified KV cache dtype."""
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    
    llm = LLM(
        model,
        lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream", 
        model_type="diffusion_lm",
        enforce_eager=True, 
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=2048,
        max_num_seqs=20,
        max_model_len=2048,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
        kv_cache_dtype=kv_cache_dtype,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    test_prompts = [
        tokenizer.bos_token + "Hello, how are you?",
        tokenizer.bos_token + "The capital of France is",
        tokenizer.bos_token + "Python is a programming language that",
    ][:num_prompts]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    start_gen = time.time()
    outputs = llm.generate(test_prompts, sampling_params)
    gen_time = time.time() - start_gen
    
    total_tokens = sum(len(o.get("token_ids", [])) for o in outputs)
    throughput = total_tokens / gen_time
    
    print(f"  - Generation time: {gen_time:.2f}s")
    print(f"  - Total tokens: {total_tokens}")
    print(f"  - Throughput: {throughput:.2f} tok/s")
    
    return {
        "kv_cache_dtype": kv_cache_dtype,
        "gen_time": gen_time,
        "total_tokens": total_tokens,
        "throughput": throughput,
    }


# ============================================================================
# 测试函数 - Quality 测试
# ============================================================================

TEST_PROMPTS = [
    "The capital of France is",
    "In a world where technology",
    "The importance of education",
    "Climate change is one of",
    "Artificial intelligence has the potential",
]

def run_inference(llm: LLM, prompts: List[str], sampling_params: SamplingParams, num_runs: int = 3) -> Dict:
    """运行推理并收集性能和质量指标"""
    results = {
        'total_time': 0.0,
        'total_tokens': 0,
        'outputs': [],
    }
    
    for run in range(num_runs):
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed_time = time.time() - start_time
        
        total_tokens = sum(len(output.get("token_ids", [])) for output in outputs)
        
        results['total_time'] += elapsed_time
        results['total_tokens'] += total_tokens
        results['outputs'].append(outputs)
    
    results['avg_time'] = results['total_time'] / num_runs
    results['avg_tokens'] = results['total_tokens'] / num_runs
    results['avg_throughput'] = results['avg_tokens'] / results['avg_time']
    
    return results

def compare_outputs(bf16_outputs: List, fp8_outputs: List, prompts: List[str]) -> Dict:
    """比较两种配置的输出"""
    comparison = {
        'text_similarity': [],
        'texts_bf16': [],
        'texts_fp8': [],
    }
    
    for bf16_out, fp8_out, prompt in zip(bf16_outputs, fp8_outputs, prompts):
        bf16_text = bf16_out.get("text", "")
        fp8_text = fp8_out.get("text", "")
        
        comparison['texts_bf16'].append(bf16_text)
        comparison['texts_fp8'].append(fp8_text)
        
        if bf16_text and fp8_text:
            min_len = min(len(bf16_text), len(fp8_text))
            if min_len > 0:
                matches = sum(1 for a, b in zip(bf16_text[:min_len], fp8_text[:min_len]) if a == b)
                similarity = matches / min_len
                comparison['text_similarity'].append(similarity)
            else:
                comparison['text_similarity'].append(0.0)
        else:
            comparison['text_similarity'].append(0.0)
    
    comparison['avg_similarity'] = np.mean(comparison['text_similarity']) if comparison['text_similarity'] else 0.0
    
    return comparison


# ============================================================================
# 测试函数 - Attention Kernel 测试
# ============================================================================

def test_q_scale_computation(attn: Attention):
    """Test Q scale computation and running max update."""
    device = 'cuda'
    seq_len = 64
    num_heads = 8
    head_dim = 128
    kv_cache_dtype = "fp8_e4m3"
    
    q1 = torch.randn(seq_len, num_heads, head_dim, device=device) * 0.5
    
    q_scale1 = attn._update_and_compute_q_fp8_scale(q1, kv_cache_dtype, device)
    
    assert q_scale1 is not None
    assert attn.q_max_abs is not None
    assert q_scale1.shape == (num_heads,)
    assert torch.all(q_scale1 > 0)
    
    q2 = torch.randn(seq_len, num_heads, head_dim, device=device) * 1.5
    q_max_abs_before = attn.q_max_abs.clone()
    
    q_scale2 = attn._update_and_compute_q_fp8_scale(q2, kv_cache_dtype, device)
    
    assert torch.all(attn.q_max_abs >= q_max_abs_before)
    
    q_scale3 = attn._get_q_fp8_scale_from_max(kv_cache_dtype)
    assert q_scale3 is not None
    assert torch.allclose(q_scale2, q_scale3)
    
    q_scale4 = attn._update_and_compute_q_fp8_scale(q1, "bf16", device)
    assert q_scale4 is None
    
    q_scale5 = attn._update_and_compute_q_fp8_scale(q1, "fp8_e5m2", device)
    assert attn.kv_cache_dtype_cache == "fp8_e5m2"


def test_q_kv_scale_consistency(attn: Attention):
    """Test that Q, K, V scales are computed consistently."""
    device = 'cuda'
    seq_len = 64
    num_heads = 8
    num_kv_heads = 4
    head_dim = 128
    kv_cache_dtype = "fp8_e4m3"
    
    scale_factor = 1.0
    q = torch.randn(seq_len, num_heads, head_dim, device=device) * scale_factor
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * scale_factor
    v = torch.randn(seq_len, num_kv_heads, head_dim, device=device) * scale_factor
    
    q_scale = attn._update_and_compute_q_fp8_scale(q, kv_cache_dtype, device)
    k_scale, v_scale = attn._update_and_compute_fp8_scales(k, v, kv_cache_dtype, device)
    
    assert q_scale is not None
    assert k_scale is not None and v_scale is not None
    
    assert q_scale.shape == (num_heads,)
    assert k_scale.shape == (num_kv_heads,)
    assert v_scale.shape == (num_kv_heads,)
    
    assert torch.all(q_scale > 0)
    assert torch.all(k_scale > 0)
    assert torch.all(v_scale > 0)
    
    q_scale_retrieved = attn._get_q_fp8_scale_from_max(kv_cache_dtype)
    k_scale_retrieved, v_scale_retrieved = attn._get_fp8_scales_from_max(kv_cache_dtype)
    
    assert torch.allclose(q_scale, q_scale_retrieved)
    assert torch.allclose(k_scale, k_scale_retrieved)
    assert torch.allclose(v_scale, v_scale_retrieved)


def test_fp8_attention_kernel_integration(attn: Attention):
    """Test FP8 attention kernel integration in decode path."""
    device = 'cuda'
    seq_len = 32
    num_heads = 8
    num_kv_heads = 4
    head_dim = 128
    kv_cache_dtype = "fp8_e4m3"
    
    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    
    q_scale = attn._update_and_compute_q_fp8_scale(q, kv_cache_dtype, device)
    assert q_scale is not None
    assert q_scale.shape == (num_heads,)
    
    k_scale, v_scale = attn._update_and_compute_fp8_scales(k, v, kv_cache_dtype, device)
    assert k_scale is not None and v_scale is not None
    
    q_scale_retrieved = attn._get_q_fp8_scale_from_max(kv_cache_dtype)
    k_scale_retrieved, v_scale_retrieved = attn._get_fp8_scales_from_max(kv_cache_dtype)
    
    assert q_scale_retrieved is not None
    assert k_scale_retrieved is not None and v_scale_retrieved is not None


def test_fp8_attention_pipeline():
    """Test FP8 attention kernel in full pipeline."""
    model = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
    
    llm = LLM(
        model,
        lora_path="/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora",
        use_lora=True,
        model_name="dream", 
        model_type="diffusion_lm",
        enforce_eager=True, 
        data_parallel_size=1,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        max_num_batched_tokens=2048,
        max_num_seqs=20,
        max_model_len=2048,
        accept_threshold=0.95,
        complete_threshold=0.9,
        add_new_block_threshold=0.1,
        kv_cache_layout="unified",
        kv_cache_dtype="fp8_e4m3",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    test_prompts = [
        tokenizer.bos_token + "Hello, how are you?",
        tokenizer.bos_token + "The capital of France is",
    ]
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    outputs = llm.generate(test_prompts, sampling_params)
    
    for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
        generated_text = output.get("text", "")
        token_ids = output.get("token_ids", [])
        
        if not generated_text.strip():
            raise ValueError(f"Generated text is empty for prompt {i+1}")
        if len(token_ids) == 0:
            raise ValueError(f"No tokens generated for prompt {i+1}")


@torch.no_grad()
def test_fp8_attention_kernel_numerics():
    """数值对齐测试：验证 triton_flash_attention 的 FP8 路径在 scale 约定下是否正确。"""
    torch.manual_seed(0)
    device = "cuda"

    def run_case(seqlen_q: int, seqlen_k: int, nheads_q: int, nheads_kv: int, head_dim: int):
        assert head_dim in (32, 64, 128, 256)
        assert nheads_q % nheads_kv == 0
        group = nheads_q // nheads_kv

        q = (torch.randn(seqlen_q, nheads_q, head_dim, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
        k = (torch.randn(seqlen_k, nheads_kv, head_dim, device=device, dtype=torch.bfloat16) * 0.2).contiguous()
        v = (torch.randn(seqlen_k, nheads_kv, head_dim, device=device, dtype=torch.bfloat16) * 0.2).contiguous()

        spec = parse_kv_cache_dtype("fp8_e4m3")
        fp8_dtype = spec.fp8_view_dtype
        assert fp8_dtype is not None and spec.fp8_max is not None and spec.fp8_min is not None

        fp8_max = float(spec.fp8_max)
        eps = 1e-8
        q_max = q.float().abs().amax(dim=(0, 2))
        k_max = k.float().abs().amax(dim=(0, 2))
        v_max = v.float().abs().amax(dim=(0, 2))
        q_scale = (q_max / fp8_max).clamp_min(eps).float()
        k_scale = (k_max / fp8_max).clamp_min(eps).float()
        v_scale = (v_max / fp8_max).clamp_min(eps).float()
        p_scale = torch.ones(1, device=device, dtype=torch.float32)

        cu_seqlens_q = torch.tensor([0, seqlen_q], device=device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, seqlen_k], device=device, dtype=torch.int32)

        o = torch.empty_like(q)
        out = triton_flash_attention(
            q, k, v, o,
            cu_seqlens_q, cu_seqlens_k,
            seqlen_q, seqlen_k,
            causal=False,
            softmax_scale=(head_dim ** -0.5),
            bias=None,
            fp8_scales=(q_scale, k_scale, v_scale, p_scale),
            fp8_out_scale=None,
            block_table=None,
        )

        def quantize_to_fp8(t_bf16: torch.Tensor, scale: torch.Tensor, heads: int) -> torch.Tensor:
            descale = (1.0 / scale).view(1, heads, 1)
            t_q = (t_bf16.float() * descale).clamp(min=float(spec.fp8_min), max=float(spec.fp8_max))
            return t_q.to(fp8_dtype).float()

        q_q = quantize_to_fp8(q, q_scale, nheads_q)
        k_q = quantize_to_fp8(k, k_scale, nheads_kv)
        v_q = quantize_to_fp8(v, v_scale, nheads_kv)

        sm_scale = head_dim ** -0.5
        kv_for_q = torch.arange(nheads_q, device=device) // group
        k_q_mapped = k_q[:, kv_for_q, :]
        v_q_mapped = v_q[:, kv_for_q, :]
        k_scale_mapped = k_scale[kv_for_q]
        v_scale_mapped = v_scale[kv_for_q]

        scores = torch.einsum("qhd,khd->hqk", q_q, k_q_mapped)
        restore = (q_scale * k_scale_mapped) * sm_scale
        scores = scores * restore.view(-1, 1, 1)
        p = torch.softmax(scores, dim=-1)
        out_ref = torch.einsum("hqk,khd->qhd", p, v_q_mapped)
        out_ref = out_ref * v_scale_mapped.view(1, -1, 1)

        out_f = out.float()
        diff = (out_f - out_ref).abs()
        rel = diff / (out_ref.abs() + 1e-6)
        print("=" * 80)
        print(f"FP8 attention kernel numerics check (Q={seqlen_q}, K={seqlen_k}, Hq={nheads_q}, Hkv={nheads_kv}, D={head_dim})")
        print(f"abs diff: mean={diff.mean().item():.6f} max={diff.max().item():.6f}")
        print(f"rel diff: mean={rel.mean().item():.6f} max={rel.max().item():.6f}")

    run_case(seqlen_q=32, seqlen_k=32, nheads_q=4, nheads_kv=4, head_dim=64)
    run_case(seqlen_q=32, seqlen_k=64, nheads_q=32, nheads_kv=4, head_dim=64)


# ============================================================================
# 测试运行函数
# ============================================================================

def run_kernel_tests() -> Dict:
    """运行 Kernel 层 FP8 支持测试"""
    print("\n" + "=" * 80)
    print("测试类别 1: Kernel 层 FP8 支持测试")
    print("=" * 80)
    
    results = {
        'unified_roundtrip': False,
        'distinct_roundtrip': False,
    }
    
    print("\n[1.1] Unified Layout Roundtrip Test")
    print("-" * 80)
    try:
        test_kv_cache_fp8_unified_roundtrip()
        print("✅ Unified layout roundtrip test PASSED")
        results['unified_roundtrip'] = True
    except Exception as e:
        print(f"❌ Unified layout roundtrip test FAILED: {e}")
        traceback.print_exc()
    
    print("\n[1.2] Distinct Layout Roundtrip Test")
    print("-" * 80)
    try:
        test_kv_cache_fp8_distinct_roundtrip()
        print("✅ Distinct layout roundtrip test PASSED")
        results['distinct_roundtrip'] = True
    except Exception as e:
        print(f"❌ Distinct layout roundtrip test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_integration_tests() -> Dict:
    """运行 Attention layer FP8 集成测试"""
    print("\n" + "=" * 80)
    print("测试类别 2: Attention Layer FP8 集成测试")
    print("=" * 80)
    
    results = {'integration': False}
    
    print("\n[2.1] Attention Layer FP8 Integration Test")
    print("-" * 80)
    try:
        num_heads = 8
        num_kv_heads = 4
        head_dim = 128
        
        attn = Attention(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=1.0 / (head_dim ** 0.5),
            num_kv_heads=num_kv_heads,
            model_type='diffusion_lm'
        )
        
        test_context_kv_cache_dtype()
        test_running_max_update(attn)
        test_scale_computation(attn)
        
        print("✅ Attention layer FP8 integration test PASSED")
        results['integration'] = True
    except Exception as e:
        print(f"❌ Attention layer FP8 integration test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_pipeline_tests() -> Dict:
    """运行完整 Pipeline 测试"""
    print("\n" + "=" * 80)
    print("测试类别 3: 完整 Pipeline 测试")
    print("=" * 80)
    
    results = {'pipeline': False}
    
    print("\n[3.1] FP8 KV Cache Pipeline Test")
    print("-" * 80)
    print("注意：此测试需要模型 checkpoint，可能需要较长时间...")
    try:
        test_fp8_kv_cache_pipeline()
        print("✅ FP8 KV cache pipeline test PASSED")
        results['pipeline'] = True
    except Exception as e:
        print(f"❌ FP8 KV cache pipeline test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_memory_tests() -> Dict:
    """运行内存使用验证测试"""
    print("\n" + "=" * 80)
    print("测试类别 4: 内存使用验证测试")
    print("=" * 80)
    
    results = {
        'memory_bf16': False,
        'memory_fp8': False,
    }
    
    print("\n[4.1] BF16 Memory Usage Test")
    print("-" * 80)
    try:
        test_kv_cache_memory("bf16")
        print("✅ BF16 memory usage test PASSED")
        results['memory_bf16'] = True
    except Exception as e:
        print(f"❌ BF16 memory usage test FAILED: {e}")
        traceback.print_exc()
    
    print("\n[4.2] FP8 Memory Usage Test")
    print("-" * 80)
    try:
        test_kv_cache_memory("fp8_e4m3")
        print("✅ FP8 memory usage test PASSED")
        results['memory_fp8'] = True
    except Exception as e:
        print(f"❌ FP8 memory usage test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_speed_tests() -> Dict:
    """运行速度对比测试"""
    print("\n" + "=" * 80)
    print("测试类别 5: 速度对比测试")
    print("=" * 80)
    
    results = {
        'speed_bf16': False,
        'speed_fp8': False,
    }
    
    print("\n[5.1] BF16 Speed Test")
    print("-" * 80)
    try:
        test_kv_cache_speed("bf16", num_prompts=3)
        print("✅ BF16 speed test PASSED")
        results['speed_bf16'] = True
    except Exception as e:
        print(f"❌ BF16 speed test FAILED: {e}")
        traceback.print_exc()
    
    print("\n[5.2] FP8 Speed Test")
    print("-" * 80)
    try:
        test_kv_cache_speed("fp8_e4m3", num_prompts=3)
        print("✅ FP8 speed test PASSED")
        results['speed_fp8'] = True
    except Exception as e:
        print(f"❌ FP8 speed test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_quality_tests() -> Dict:
    """运行质量和速度对比测试"""
    print("\n" + "=" * 80)
    print("测试类别 6: 质量和速度对比测试")
    print("=" * 80)
    print("注意：此测试需要较长时间（可能需要 10-20 分钟）...")
    
    results = {'quality': False}
    
    print("\n[6.1] FP8 vs BF16 Quality and Speed Comparison")
    print("-" * 80)
    try:
        import torch.distributed as dist
        
        model_path = "/data1/ckpts/Dream-org/Dream-v0-Base-7B"
        lora_path = "/data1/ckpts/SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        test_prompts = [tokenizer.bos_token + prompt for prompt in TEST_PROMPTS]
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        num_runs = 3
        
        llm_bf16 = LLM(
            model_path,
            lora_path=lora_path,
            use_lora=True,
            model_name="dream",
            model_type="diffusion_lm",
            enforce_eager=True,
            data_parallel_size=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
            max_num_batched_tokens=2048,
            max_num_seqs=20,
            max_model_len=2048,
            accept_threshold=0.95,
            complete_threshold=0.9,
            add_new_block_threshold=0.1,
            kv_cache_layout="unified",
            kv_cache_dtype="bf16",
        )
        
        bf16_results = run_inference(llm_bf16, test_prompts, sampling_params, num_runs)
        print(f"\n[BF16 结果汇总]")
        print(f"  平均吞吐量: {bf16_results['avg_throughput']:.2f} tok/s")
        
        del llm_bf16
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        
        llm_fp8 = LLM(
            model_path,
            lora_path=lora_path,
            use_lora=True,
            model_name="dream",
            model_type="diffusion_lm",
            enforce_eager=True,
            data_parallel_size=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.4,
            max_num_batched_tokens=2048,
            max_num_seqs=20,
            max_model_len=2048,
            accept_threshold=0.95,
            complete_threshold=0.9,
            add_new_block_threshold=0.1,
            kv_cache_layout="unified",
            kv_cache_dtype="fp8_e4m3",
        )
        
        fp8_results = run_inference(llm_fp8, test_prompts, sampling_params, num_runs)
        print(f"\n[FP8 结果汇总]")
        print(f"  平均吞吐量: {fp8_results['avg_throughput']:.2f} tok/s")
        
        speedup = fp8_results['avg_throughput'] / bf16_results['avg_throughput']
        print(f"\n  速度比: {speedup:.2f}x")
        
        bf16_outputs_last = bf16_results['outputs'][-1]
        fp8_outputs_last = fp8_results['outputs'][-1]
        
        comparison = compare_outputs(bf16_outputs_last, fp8_outputs_last, test_prompts)
        print(f"\n平均文本相似度: {comparison['avg_similarity']:.4f}")
        
        del llm_fp8
        torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        
        print("✅ Quality and speed comparison test PASSED")
        results['quality'] = True
    except Exception as e:
        print(f"❌ Quality and speed comparison test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_attention_kernel_tests() -> Dict:
    """运行 FP8 Attention Kernel 单元测试"""
    print("\n" + "=" * 80)
    print("测试类别 7: FP8 Attention Kernel 单元测试")
    print("=" * 80)
    
    results = {'attention_kernel': False}
    
    print("\n[7.1] FP8 Attention Kernel Unit Test")
    print("-" * 80)
    try:
        num_heads = 8
        num_kv_heads = 4
        head_dim = 128
        
        attn = Attention(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=1.0 / (head_dim ** 0.5),
            num_kv_heads=num_kv_heads,
            model_type='diffusion_lm'
        )
        
        test_q_scale_computation(attn)
        test_q_kv_scale_consistency(attn)
        test_fp8_attention_kernel_integration(attn)
        
        print("✅ FP8 attention kernel unit test PASSED")
        results['attention_kernel'] = True
    except Exception as e:
        print(f"❌ FP8 attention kernel unit test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_attention_e2e_tests() -> Dict:
    """运行 FP8 Attention Kernel 端到端测试"""
    print("\n" + "=" * 80)
    print("测试类别 8: FP8 Attention Kernel 端到端测试")
    print("=" * 80)
    
    results = {'attention_e2e': False}
    
    print("\n[8.1] FP8 Attention Kernel End-to-End Test")
    print("-" * 80)
    print("注意：此测试需要模型 checkpoint，可能需要较长时间...")
    try:
        test_fp8_attention_pipeline()
        print("✅ FP8 attention kernel end-to-end test PASSED")
        results['attention_e2e'] = True
    except Exception as e:
        print(f"❌ FP8 attention kernel end-to-end test FAILED: {e}")
        traceback.print_exc()
    
    return results

def run_attention_numerics_tests() -> Dict:
    """运行 FP8 Attention Kernel 数值验证测试"""
    print("\n" + "=" * 80)
    print("测试类别 9: FP8 Attention Kernel 数值验证测试")
    print("=" * 80)
    
    results = {'attention_numerics': False}
    
    print("\n[9.1] FP8 Attention Kernel Numerics Test")
    print("-" * 80)
    try:
        test_fp8_attention_kernel_numerics()
        print("✅ FP8 attention kernel numerics test PASSED")
        results['attention_numerics'] = True
    except Exception as e:
        print(f"❌ FP8 attention kernel numerics test FAILED: {e}")
        traceback.print_exc()
    
    return results

def print_summary(all_results: Dict):
    """打印测试结果摘要"""
    print("\n" + "=" * 80)
    print("测试结果摘要")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        for test_name, passed in results.items():
            total_tests += 1
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")
            if passed:
                passed_tests += 1
    
    print(f"\n总计: {passed_tests}/{total_tests} 测试通过 ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description='FP8 KV Cache 综合测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试类别说明：
  kernel          - Kernel 层 FP8 支持测试（roundtrip tests）
  integration     - Attention layer FP8 集成测试
  pipeline        - 完整 pipeline 测试（需要模型）
  memory          - 内存使用验证测试（需要模型）
  speed           - 速度对比测试（需要模型）
  quality         - 质量和速度对比测试（需要模型，较耗时）
  attention_kernel - FP8 attention kernel 单元测试
  attention_e2e   - FP8 attention kernel 端到端测试（需要模型）
  attention_numerics - FP8 attention kernel 数值验证测试
  all             - 运行所有测试（除了 quality，因为需要较长时间）

示例：
  # 运行所有测试（除了 quality）
  python test_fp8_kv_cache_comprehensive.py --tests all

  # 运行特定测试
  python test_fp8_kv_cache_comprehensive.py --tests kernel integration

  # 运行 speed 和 quality 测试
  python test_fp8_kv_cache_comprehensive.py --tests speed quality
        """
    )
    parser.add_argument(
        '--tests',
        nargs='+',
        default=['all'],
        choices=['kernel', 'integration', 'pipeline', 'memory', 'speed', 
                 'quality', 'attention_kernel', 'attention_e2e', 'attention_numerics', 'all'],
        help='要运行的测试类别（默认: all）'
    )
    
    args = parser.parse_args()
    
    if 'all' in args.tests:
        test_categories = ['kernel', 'integration', 'pipeline', 'memory', 'speed',
                          'attention_kernel', 'attention_e2e', 'attention_numerics']
    else:
        test_categories = args.tests
    
    print("=" * 80)
    print("FP8 KV Cache 综合测试")
    print("=" * 80)
    print(f"测试类别: {', '.join(test_categories)}")
    print(f"工作目录: {PROJECT_ROOT}")
    
    all_results = {}
    
    if 'kernel' in test_categories:
        all_results['kernel'] = run_kernel_tests()
    
    if 'integration' in test_categories:
        all_results['integration'] = run_integration_tests()
    
    if 'pipeline' in test_categories:
        all_results['pipeline'] = run_pipeline_tests()
    
    if 'memory' in test_categories:
        all_results['memory'] = run_memory_tests()
    
    if 'speed' in test_categories:
        all_results['speed'] = run_speed_tests()
    
    if 'quality' in test_categories:
        all_results['quality'] = run_quality_tests()
    
    if 'attention_kernel' in test_categories:
        all_results['attention_kernel'] = run_attention_kernel_tests()
    
    if 'attention_e2e' in test_categories:
        all_results['attention_e2e'] = run_attention_e2e_tests()
    
    if 'attention_numerics' in test_categories:
        all_results['attention_numerics'] = run_attention_numerics_tests()
    
    exit_code = print_summary(all_results)
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
