# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: D2F

# Organization: SJTU DENG Lab
# Author: Drew Jin (JIN. Yijie, @drewjin)
# Date: 2025-08-07
# Email: drewjin0827@gmail.com
# All rights reserved.

import torch
import triton

import triton.language as tl

from diffulex.legacy.utils.context import ContextForDiffusionLM


def CHECK_ATTENTION(o: torch.Tensor, q: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                    k_cache: torch.Tensor, v_cache: torch.Tensor, context: ContextForDiffusionLM):
    """
    Check the attention output against the input tensors.
    """
    from einops import rearrange
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    from torch.nn.attention import SDPBackend, sdpa_kernel
    
    from diffulex.legacy.layers.attention.ops import load_kvcache
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    h_dim = v_cache.shape[-2]
    x = k_cache.shape[-1]
    k_cache_unified = rearrange(k_cache, "b h n s x -> b s h (n x)", n=h_dim // x, x=x).contiguous()
    v_cache_unified = rearrange(v_cache, "b h d s -> b s h d").contiguous()
    
    transpose_fn = lambda x: rearrange(x, 's h d -> 1 h s d').contiguous()
    k, v = load_kvcache(k_cache_unified, v_cache_unified, context, k_new, v_new)
    q, k, v = map(transpose_fn, (q, k, v))
    mask = context.block_mask_for_checking
    with sdpa_kernel(SDPBackend.MATH):
        ref_o = sdpa(q, k, v, attn_mask=mask, enable_gqa=True)
    
    ref_o = rearrange(ref_o, '1 h s d -> s h d')
    assert torch.allclose(o, ref_o, atol=1e-3, rtol=1e-3), "Attention output does not match reference!"


@triton.jit
def dlm_flash_decoding_kernel(q_ptr, k_ptr, v_ptr, o_ptr, mask_ptr, softmax_scale,
                              k_cache_ptr, v_cache_ptr, block_tables_ptr,
                              cu_seqlens_q_ptr, total_lens_ptr, ctx_lens_ptr,
                              q_stride_m, q_stride_nh, q_stride_d,
                              k_stride_n, k_stride_nh, k_stride_d,
                              v_stride_n, v_stride_nh, v_stride_d,
                              o_stride_m, o_stride_nh, o_stride_d,
                              mask_stride_m, mask_stride_n,
                              k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
                              v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
                              block_tables_stride_nseqs, block_tables_stride_nblks,
                              cu_seqlens_q_ptr_stride, total_lens_ptr_stride, ctx_lens_ptr_stride,
                              NUM_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, KV_HEAD_GROUP_SIZE: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, x: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                              NUM_UNROLL_CACHE: tl.constexpr = 4, NUM_UNROLL_Q: tl.constexpr = 1):
    pass


def diffusion_lm_flash_decoding(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, 
                                k_cache: torch.Tensor, v_cache: torch.Tensor, block_tables: torch.Tensor, 
                                cu_seqlens_q: torch.Tensor, seq_lens: torch.Tensor, total_lens: torch.Tensor, ctx_lens: torch.Tensor,
                                max_total_len: int | None = None, max_seq_len: int | None = None, 
                                diffusion_block_size: int = 32):
    '''
        FIXME
        q: [TotalInputLength, NumHeads, HeadDim]
        k: [TotalInputLength, NumHeads, HeadDim]
        v: [TotalInputLength, NumHeads, HeadDim]
        mask: [TotalInputLength, TotalInputLength]
        k_cache: [NumBlocks, NumHeads, HeadDim // x, BlockSize, x]
        v_cache: [NumBlocks, NumHeads, HeadDim, BlockSize]
        block_tables: [NumSeqs, MaxSeqNumBlocks] # NumSeqs == BatchSize
        ...
    '''
    is_pow_of_2 = lambda x: (x & (x - 1)) == 0 and x > 0
    assert k_cache.shape[-2] == v_cache.shape[-1], "BLOCK_SIZE between k_cache and v_cache must match"
    assert k.shape == v.shape, "k, v must have the same shape"
    assert k.shape[1] == k_cache.shape[1] == v_cache.shape[1], "Number of heads must match"
    assert q.shape[1] % k.shape[1] == 0, "Number of heads in q must be a multiple of the number of heads in k and v"
    assert k_cache.shape[-3] * k_cache.shape[-1] == v_cache.shape[-2] == q.shape[-1], "Head dimension must match"
    assert is_pow_of_2(q.shape[-1]) and is_pow_of_2(k_cache.shape[-3] * k_cache.shape[-1]), \
        "Head dimension must be a multiple of 2 for triton kernel compatibility"
    assert len(seq_lens) == len(ctx_lens) == len(total_lens) == len(cu_seqlens_q) - 1 == len(block_tables), \
        "Number of sequences must match across all inputs"

    BLOCK_SIZE = k_cache.shape[-2]  # BLOCK_SIZE or PAGE_SIZE of paged kv cache
    NUM_SEQS = len(ctx_lens)
    NUM_HEADS = q.shape[1]
    o = torch.empty_like(q).to(q.device).to(q.dtype)
    x = k_cache.shape[-1]
    max_seq_len = max_seq_len if max_seq_len is not None else max(seq_lens)
    max_total_len = max_total_len if max_total_len is not None else max(total_lens)
    softmax_scale = 1.0 / (k.shape[-1] ** 0.5)
    
    KV_HEAD_GROUP_SIZE = q.shape[1] // k.shape[1]
    HEAD_DIM = q.shape[-1]
    BLOCK_M = BLOCK_N = diffusion_block_size * 2
    GRID = (NUM_SEQS, NUM_HEADS, triton.cdiv(max_seq_len, BLOCK_M))
    
    dlm_flash_decoding_kernel[GRID](
        q, k, v, o, mask, softmax_scale, k_cache, v_cache, block_tables,
        cu_seqlens_q, total_lens, ctx_lens,
        *q.stride(), *k.stride(), *v.stride(), *o.stride(), *mask.stride(),
        *k_cache.stride(), *v_cache.stride(), *block_tables.stride(),
        cu_seqlens_q.stride(0), total_lens.stride(0), ctx_lens.stride(0),
        NUM_HEADS=NUM_HEADS, HEAD_DIM=HEAD_DIM, 
        KV_HEAD_GROUP_SIZE=KV_HEAD_GROUP_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, x=x,
        BLOCK_SIZE=BLOCK_SIZE, 
        NUM_UNROLL_CACHE=4, NUM_UNROLL_Q=1
    )
    return o