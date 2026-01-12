import os
from pathlib import Path

import torch
import tilelang
import tilelang.testing
import torch.nn.functional as F
from einops import rearrange

from diffulex_kernel.python.dllm_flash_attn_kernels import dllm_flash_attn_prefill_kernel


def naive_sdpa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    scale: float,
    diffusion_block_size: int,
    is_block_attn: bool,
) -> torch.Tensor:
    """
    Naive prefill attention reference to verify TileLang kernel.
    """
    num_seqs = len(cu_seqlens_q) - 1

    output = torch.zeros_like(q)
    for seq_idx in range(num_seqs):
        q_start = cu_seqlens_q[seq_idx].item()
        q_end = cu_seqlens_q[seq_idx + 1].item()
        kv_start = cu_seqlens_k[seq_idx].item()
        kv_end = cu_seqlens_k[seq_idx + 1].item()

        q_seq = q[q_start:q_end]
        k_seq = k[kv_start:kv_end]
        v_seq = v[kv_start:kv_end]

        q_len = q_seq.shape[0]
        kv_len = k_seq.shape[0]

        q_sdpa = rearrange(q_seq, 's h d -> 1 h s d') # [1, num_heads, q_len, head_dim]
        k_sdpa = rearrange(k_seq, 's h d -> 1 h s d') # [1, num_heads, kv_len, head_dim]
        v_sdpa = rearrange(v_seq, 's h d -> 1 h s d') # [1, num_heads, kv_len, head_dim]

        if not is_block_attn:
            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )
        else:
            block_mask = torch.zeros((1, 1, q_len, kv_len), dtype=q.dtype, device=q.device).bool()
            num_diffusion_blocks = (kv_len + diffusion_block_size - 1) // diffusion_block_size
            for block_idx in range(num_diffusion_blocks):
                block_start = block_idx * diffusion_block_size
                block_end = min(block_start + diffusion_block_size, kv_len)
                block_mask[..., block_start:block_end, :block_end] = True

            attn_out = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=block_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                enable_gqa=True,
            )

        output[q_start:q_end] = rearrange(attn_out, '1 h s d -> s h d').to(output.dtype)

    return output


def run_dllm_flash_attn_prefill(
    num_seqs: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_q_len: int,
    max_kv_len: int,
    is_block_attn: bool,
    diffusion_block_size: int,
    dtype: str = "bfloat16",
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 1,
    num_threads: int = 128,
):
    """Run prefill kernel and compare with naive reference."""
    torch_dtype = getattr(torch, dtype)
    device = "cuda"
    num_groups = num_heads // num_kv_heads

    # Use uniform seq length per sequence to cover block mask branches
    cu_seqlens_q = torch.arange(0, (num_seqs + 1) * max_q_len, max_q_len, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (num_seqs + 1) * max_kv_len, max_kv_len, dtype=torch.int32, device=device)

    total_q_len = cu_seqlens_q[-1].item()
    total_kv_len = cu_seqlens_k[-1].item()

    q = torch.randn(total_q_len, num_heads, head_dim, dtype=torch_dtype, device=device)
    k = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=torch_dtype, device=device)
    v = torch.randn_like(k)

    prefill_kernel = dllm_flash_attn_prefill_kernel(
        num_seqs,
        num_groups,
        total_q_len,
        total_kv_len,
        num_heads,
        head_dim,
        is_block_attn,
        diffusion_block_size,
        block_m,
        block_n,
        num_stages,
        num_threads,
    )

    kernel_source = prefill_kernel.get_kernel_source()
    cuda_cache_dir = os.getenv("CUDA_CACHE_DIR", "./cuda_cache")
    cache_root = Path(cuda_cache_dir) / "test_dllm_flash_attn_prefill_kernel"
    case_dir = cache_root / (
        f"seq{num_seqs}_heads{num_heads}_kv{num_kv_heads}_hd{head_dim}_"
        f"mq{max_q_len}_mk{max_kv_len}_block{int(is_block_attn)}_"
        f"dbs{diffusion_block_size}_dtype{dtype}_bm{block_m}_bn{block_n}_"
        f"stg{num_stages}_thr{num_threads}"
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    kernel_path = case_dir / "kernel.cu"
    kernel_path.write_text(kernel_source)
    print(f"Kernel source saved to {kernel_path}")

    output = prefill_kernel(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_q_len,
    )

    scale = 1.0 / (head_dim ** 0.5)
    ref_output = naive_sdpa_prefill(
        q, k, v,
        cu_seqlens_q,
        cu_seqlens_k,
        scale,
        diffusion_block_size,
        is_block_attn,
    )

    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
    print(f"Test passed! Shape: {output.shape}")


# ==================== Kernel Tests ====================
def test_prefill_bf16_single_seq():
    """Single sequence, bfloat16."""
    run_dllm_flash_attn_prefill(
        num_seqs=1,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=192,
        max_kv_len=192,
        is_block_attn=False,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_multi_seq():
    """Multiple sequences, bfloat16."""
    run_dllm_flash_attn_prefill(
        num_seqs=4,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=256,
        max_kv_len=256,
        is_block_attn=False,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_block_attn():
    """Block attention, bfloat16."""
    run_dllm_flash_attn_prefill(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=256,
        max_kv_len=256,
        is_block_attn=True,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_block_attn_multi_seq_long_ctx():
    """Block attention, more sequences and longer context."""
    run_dllm_flash_attn_prefill(
        num_seqs=3,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=320,
        max_kv_len=320,
        is_block_attn=True,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_block_attn_diffusion_64():
    """Block attention, diffusion block 64 to hit mask branch."""
    run_dllm_flash_attn_prefill(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=256,
        max_kv_len=256,
        is_block_attn=True,
        diffusion_block_size=64,
        dtype="bfloat16",
    )


def test_prefill_bf16_gqa_4():
    """GQA ratio = 4."""
    run_dllm_flash_attn_prefill(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=192,
        max_kv_len=192,
        is_block_attn=False,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_gqa_8():
    """GQA ratio = 8."""
    run_dllm_flash_attn_prefill(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        max_q_len=192,
        max_kv_len=192,
        is_block_attn=False,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_block_attn_gqa_8():
    """Block attention with GQA ratio = 8."""
    run_dllm_flash_attn_prefill(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        max_q_len=256,
        max_kv_len=256,
        is_block_attn=True,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_head_dim_64():
    """Head dim = 64."""
    run_dllm_flash_attn_prefill(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        max_q_len=192,
        max_kv_len=192,
        is_block_attn=False,
        diffusion_block_size=32,
        dtype="bfloat16",
    )


def test_prefill_bf16_varied_stages():
    """Multiple pipeline stages."""
    run_dllm_flash_attn_prefill(
        num_seqs=2,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        max_q_len=192,
        max_kv_len=192,
        is_block_attn=False,
        diffusion_block_size=32,
        dtype="bfloat16",
        num_stages=2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
