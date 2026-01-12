import os
from pathlib import Path

import torch
import tilelang
import tilelang.testing
import torch.nn.functional as F
from einops import rearrange

from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_decode_kernel


def naive_sdpa_with_kvcache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    num_groups: int,
    page_block_size: int,
) -> torch.Tensor:
    """
    Naive attention reference implementation with KV cache support.
    
    Args:
        q: [Q_LEN, NUM_HEADS, HEAD_DIM]
        k: [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        v: [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        k_cache: [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
        v_cache: [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
        block_tables: [NUM_SEQS, MAX_SEQ_NUM_BLOCKS]
        context_lens: [NUM_SEQS]
        cu_seqlens_q: [NUM_SEQS + 1]
        cu_seqlens_k: [NUM_SEQS + 1]
        scale: attention scale
        num_groups: number of GQA groups
        page_block_size: page block size
    
    Returns:
        output: [Q_LEN, NUM_HEADS, HEAD_DIM]
    """
    num_seqs = len(cu_seqlens_q) - 1
    
    output = torch.zeros_like(q)
    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        kv_start = cu_seqlens_k[seq_idx].item()
        kv_end = cu_seqlens_k[seq_idx + 1].item()
        
        q_seq = q[q_start:q_end]  # [seq_q_len, num_heads, head_dim]
        k_seq = k[kv_start:kv_end]  # [seq_kv_len, num_kv_heads, head_dim]
        v_seq = v[kv_start:kv_end]  # [seq_kv_len, num_kv_heads, head_dim]
        
        context_len = context_lens[seq_idx].item()
        
        # Load KV cache for this sequence
        k_cache_seq_list = []
        v_cache_seq_list = []
        
        for block_idx in range(block_tables.shape[1]):
            page_block_idx = block_tables[seq_idx, block_idx].item()
            if page_block_idx >= 0:
                # Calculate how many tokens to take from this block
                block_start = block_idx * page_block_size
                if block_start < context_len:
                    block_end = min(block_start + page_block_size, context_len)
                    num_tokens = block_end - block_start
                    k_cache_seq_list.append(k_cache[page_block_idx, :num_tokens])
                    v_cache_seq_list.append(v_cache[page_block_idx, :num_tokens])
        
        if k_cache_seq_list:
            k_cache_seq = torch.cat(k_cache_seq_list, dim=0)  # [context_len, num_kv_heads, head_dim]
            v_cache_seq = torch.cat(v_cache_seq_list, dim=0)  # [context_len, num_kv_heads, head_dim]
            
            # Combine KV cache and current KV
            k_combined = torch.cat([k_cache_seq, k_seq], dim=0)
            v_combined = torch.cat([v_cache_seq, v_seq], dim=0)
        else:
            k_combined = k_seq
            v_combined = v_seq

        q_sdpa = rearrange(q_seq, 's h d -> 1 h s d') # [1, num_heads, seq_q_len, head_dim]
        k_sdpa = rearrange(k_combined, 's h d -> 1 h s d') # [1, num_heads, total_kv_len, head_dim]
        v_sdpa = rearrange(v_combined, 's h d -> 1 h s d') # [1, num_heads, total_kv_len, head_dim]

        attn_out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=True,
        )  # [1, num_heads, seq_q_len, head_dim]

        output[q_start:q_end] = rearrange(attn_out, '1 h s d -> s h d').to(output.dtype)
    
    return output


def run_dllm_flash_attn_decode(
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_len: int,
    context_len: int,
    page_block_size: int,
    diffusion_block_size: int,
    is_block_attn: bool,
    dtype: str = "bfloat16",
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 1,
    num_threads: int = 128,
):
    """
    Run DLLM flash attention decode kernel test.
    """
    torch_dtype = getattr(torch, dtype)
    device = "cuda"
    
    num_groups = num_heads // num_kv_heads
    
    # Decode phase: each sequence decodes exactly one block; length equals block size
    total_q_len = num_seqs * diffusion_block_size
    total_kv_len = num_seqs * diffusion_block_size
    
    # Calculate number of page blocks needed
    num_blocks_per_seq = (context_len + page_block_size - 1) // page_block_size
    max_seq_num_blocks = num_blocks_per_seq
    num_page_blocks = num_seqs * num_blocks_per_seq
    
    # Generate input tensors
    q = torch.randn(total_q_len, num_heads, head_dim, dtype=torch_dtype, device=device)
    k = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=torch_dtype, device=device)
    v = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=torch_dtype, device=device)
    
    # KV cache
    k_cache = torch.randn(num_page_blocks, page_block_size, num_kv_heads, head_dim, dtype=torch_dtype, device=device)
    v_cache = torch.randn(num_page_blocks, page_block_size, num_kv_heads, head_dim, dtype=torch_dtype, device=device)
    
    # Block tables - assign page blocks sequentially for each sequence
    block_tables = torch.zeros(num_seqs, max_seq_num_blocks, dtype=torch.int32, device=device)
    for seq_idx in range(num_seqs):
        for block_idx in range(num_blocks_per_seq):
            block_tables[seq_idx, block_idx] = seq_idx * num_blocks_per_seq + block_idx
    
    # Context lengths
    context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32, device=device)
    
    # Cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, (num_seqs + 1) * diffusion_block_size, diffusion_block_size, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (num_seqs + 1) * diffusion_block_size, diffusion_block_size, dtype=torch.int32, device=device)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    # Run kernel
    decode_kernel = dllm_flash_attn_decode_kernel(
        num_seqs,
        num_groups,
        num_page_blocks,
        total_q_len,
        total_kv_len,
        num_heads,
        head_dim,
        is_block_attn,
        diffusion_block_size,
        max_seq_num_blocks,
        page_block_size,
        block_m,
        block_n,
        num_stages,
        num_threads,
    )
    
    kernel_source = decode_kernel.get_kernel_source()

    cuda_cache_dir = os.getenv("CUDA_CACHE_DIR", "./cuda_cache")
    cache_root = Path(cuda_cache_dir) / "test_dllm_flash_attn_decode_kernel"
    case_dir = cache_root / (
        f"seq{num_seqs}_heads{num_heads}_kv{num_kv_heads}_hd{head_dim}_"
        f"ctx{context_len}_pbs{page_block_size}_dbs{diffusion_block_size}_"
        f"block{int(is_block_attn)}_dtype{dtype}_bm{block_m}_bn{block_n}_"
        f"stg{num_stages}_thr{num_threads}_mq{max_q_len}_mk{max_kv_len}"
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = case_dir / "kernel.cu"
    kernel_path.write_text(kernel_source)
    print(f"Kernel source saved to {kernel_path}")
    
    output = decode_kernel(
        q, k, v, k_cache, v_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        max_q_len,
    )
    
    # Compute reference output
    ref_output = naive_sdpa_with_kvcache(
        q, k, v, k_cache, v_cache,
        block_tables, context_lens,
        cu_seqlens_q, cu_seqlens_k,
        scale, num_groups, page_block_size,
    )
    
    # Compare outputs
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
    print(f"Test passed! Shape: {output.shape}")


# ==================== Kernel Tests ====================
def test_decode_bf16_single_seq():
    """Test with single sequence, bfloat16."""
    run_dllm_flash_attn_decode(
        num_seqs=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=128,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_multi_seq():
    """Test with multiple sequences, bfloat16."""
    run_dllm_flash_attn_decode(
        num_seqs=4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=256,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_multi_seq_long_context():
    """Test with multiple sequences, bfloat16."""
    run_dllm_flash_attn_decode(
        num_seqs=4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=1024,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_block_attn():
    """Test with block attention enabled."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=128,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=True,
        dtype="bfloat16",
    )


def test_decode_bf16_gqa_4():
    """Test with GQA ratio 4."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=128,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_gqa_8():
    """Test with GQA ratio 8."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=128,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_head_dim_64():
    """Test with head dimension 64."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        max_q_len=64,
        max_kv_len=64,
        context_len=128,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_large_context():
    """Test with larger context length."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=512,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_page_block_64():
    """Test with page block size 64."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=256,
        page_block_size=64,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
    )


def test_decode_bf16_diffusion_block_64():
    """Test with diffusion block size 64."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=128,
        page_block_size=32,
        diffusion_block_size=64,
        is_block_attn=True,
        dtype="bfloat16",
    )


def test_decode_bf16_varied_stages():
    """Test with different pipeline stages."""
    run_dllm_flash_attn_decode(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=64,
        max_kv_len=64,
        context_len=128,
        page_block_size=32,
        diffusion_block_size=32,
        is_block_attn=False,
        dtype="bfloat16",
        num_stages=2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
