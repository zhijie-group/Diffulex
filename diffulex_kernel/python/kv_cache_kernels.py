import torch
import triton 

import triton.language as tl

from typing import Tuple
import os

from diffulex.attention.metadata import AttnMetaDataBase
    

@triton.jit
def dllm_store_kvcache_kernel_unified_bf16(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    """BF16 unified layout store kernel - no quantization, direct storage."""
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    key_offsets = token_idx * key_stride + tl.arange(0, D)
    value_offsets = token_idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def dllm_store_kvcache_kernel_distinct_bf16(
    k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    k_stride, v_stride,  
    k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
    v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
    nheads, hdim, blk_sz,
    x: tl.constexpr, D: tl.constexpr
):  
    """BF16 distinct layout store kernel - no quantization, direct storage."""
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-FileCopyrightText: D2F

    # Translated from vLLM's CUDA kernel 
    # Referencing https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu#L212
    # and https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu#L415
    
    # Organization: SJTU DENG Lab
    # Author: Drew Jin (JIN. Yijie, @drewjin)
    # Date: 2025-08-03
    # Email: drewjin0827@gmail.com
    # All rights reserved.
    
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return
    
    blk_idx = slot_idx // blk_sz
    off_blk = slot_idx % blk_sz
    
    offs_d = tl.arange(0, D)
    offs_k = token_idx * k_stride + offs_d
    offs_v = token_idx * v_stride + offs_d
    k = tl.load(k_ptr + offs_k)
    v = tl.load(v_ptr + offs_v)

    h_ids = offs_d // hdim
    h_offs = offs_d % hdim
    x_ids = h_offs // x
    x_offs = h_offs % x
    
    k_cache_offs = (blk_idx * k_cache_stride_nblks + h_ids * k_cache_stride_h +
                    x_ids * k_cache_stride_dx + off_blk * k_cache_stride_blk_sz + 
                    x_offs * k_cache_stride_x)
    v_cache_offs = (blk_idx * v_cache_stride_nblks + h_ids * v_cache_stride_h +
                    h_offs * v_cache_stride_d + off_blk * v_cache_stride_blk_sz)
    
    tl.store(k_cache_ptr + k_cache_offs, k)
    tl.store(v_cache_ptr + v_cache_offs, v)
    

@triton.jit
def dllm_store_kvcache_kernel_distinct_fp8(
    k_quantized_ptr, v_quantized_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    k_quantized_stride, v_quantized_stride,
    k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
    v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
    nheads, hdim, blk_sz,
    x: tl.constexpr, D: tl.constexpr
):
    """FP8 distinct layout store kernel - stores already quantized uint8 key/value to cache."""
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-FileCopyrightText: D2F
    
    # Organization: SJTU DENG Lab
    # Author: Drew Jin (JIN. Yijie, @drewjin)
    # Date: 2025-12-29
    # Email: drewjin0827@gmail.com
    # All rights reserved.
    
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return
    
    blk_idx = slot_idx // blk_sz
    off_blk = slot_idx % blk_sz
    
    offs_d = tl.arange(0, D)
    offs_k = token_idx * k_quantized_stride + offs_d
    offs_v = token_idx * v_quantized_stride + offs_d
    k_uint8 = tl.load(k_quantized_ptr + offs_k)
    v_uint8 = tl.load(v_quantized_ptr + offs_v)

    h_ids = offs_d // hdim
    h_offs = offs_d % hdim
    x_ids = h_offs // x
    x_offs = h_offs % x
    
    k_cache_offs = (blk_idx * k_cache_stride_nblks + h_ids * k_cache_stride_h +
                    x_ids * k_cache_stride_dx + off_blk * k_cache_stride_blk_sz + 
                    x_offs * k_cache_stride_x)
    v_cache_offs = (blk_idx * v_cache_stride_nblks + h_ids * v_cache_stride_h +
                    h_offs * v_cache_stride_d + off_blk * v_cache_stride_blk_sz)
    
    tl.store(k_cache_ptr + k_cache_offs, k_uint8)
    tl.store(v_cache_ptr + v_cache_offs, v_uint8)


def _store_kvcache_distinct_fp8(key: torch.Tensor, value: torch.Tensor,
                                k_cache: torch.Tensor, v_cache: torch.Tensor,
                                slot_mapping: torch.Tensor,
                                k_scale: torch.Tensor, v_scale: torch.Tensor,
                                *, strategy) -> None:
    """Helper function for FP8 distinct layout store.
    
    Quantizes BF16 key/value to FP8 (uint8 storage) using strategy, then stores to cache.
    """
    # k_cache: [num_blks, h, hdim // x, blk_sz, x]
    # v_cache: [num_blks, h, hdim, blk_sz]
    NBlks, NHeads, HDim_x, Blk_sz, x = k_cache.shape
    HDim = HDim_x * x
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    
    assert HDim == head_dim and NHeads == num_kv_heads
    assert N == slot_mapping.numel()
    
    # Vectorized quantization: [N, H, D] -> [N, H, D] uint8, then flatten to [N, H*D]
    k_q, v_q = strategy.quantize_kv_for_store(key, value, k_scale=k_scale, v_scale=v_scale)
    key_quantized = k_q.reshape(N, D).contiguous()
    value_quantized = v_q.reshape(N, D).contiguous()
    
    assert key_quantized.dtype == torch.uint8, f"Expected uint8, got {key_quantized.dtype}"
    assert value_quantized.dtype == torch.uint8, f"Expected uint8, got {value_quantized.dtype}"
    
    GRID = (N, )
    dllm_store_kvcache_kernel_distinct_fp8[GRID](
        key_quantized, value_quantized,
        k_cache, v_cache,
        slot_mapping,
        key_quantized.stride(0), value_quantized.stride(0),
        *k_cache.stride(), *v_cache.stride(),
        NHeads, HDim, Blk_sz,
        x, D
    )


def _store_kvcache_distinct_bf16(key: torch.Tensor, value: torch.Tensor, 
                                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                  slot_mapping: torch.Tensor) -> None:
    """Helper function for BF16 distinct layout store."""
    # k_cache: [num_blks, h, hdim // x, blk_sz, x]
    # v_cache: [num_blks, h, hdim, blk_sz]
    NBlks, NHeads, HDim_x, Blk_sz, x = k_cache.shape
    HDim = HDim_x * x
    N = key.shape[0]
    assert HDim == key.shape[-1] and NHeads == key.shape[1]
    assert N == slot_mapping.numel()
    
    GRID = (N, )
    dllm_store_kvcache_kernel_distinct_bf16[GRID](
        key, value,
        k_cache, v_cache,
        slot_mapping,
        key.stride(0), value.stride(0), 
        *k_cache.stride(), *v_cache.stride(),
        NHeads, HDim, Blk_sz,
        x, HDim * NHeads
    )


@triton.jit
def dllm_store_kvcache_kernel_unified_fp8(
    key_quantized_ptr,
    key_quantized_stride,
    value_quantized_ptr,
    value_quantized_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    """FP8 unified layout store kernel - stores already quantized uint8 key/value to cache.
    
    For unified layout cache shape [num_blocks, block_size, num_kv_heads, head_dim],
    we assume stride(1) == D (where D = num_kv_heads * head_dim), so offset is slot * D.
    This matches the BF16 kernel's behavior.
    """
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    key_offsets = token_idx * key_quantized_stride + tl.arange(0, D)
    value_offsets = token_idx * value_quantized_stride + tl.arange(0, D)
    key_uint8 = tl.load(key_quantized_ptr + key_offsets)
    value_uint8 = tl.load(value_quantized_ptr + value_offsets)
    # For unified layout with stride(1) == D, offset is slot * D
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key_uint8)
    tl.store(v_cache_ptr + cache_offsets, value_uint8)


def _store_kvcache_unified_fp8(key: torch.Tensor, value: torch.Tensor,
                                k_cache: torch.Tensor, v_cache: torch.Tensor,
                                slot_mapping: torch.Tensor,
                                k_scale: torch.Tensor, v_scale: torch.Tensor,
                                *, strategy) -> None:
    """Helper function for FP8 unified layout store.
    
    Quantizes BF16 key/value to FP8 (uint8 storage) using strategy, then stores to cache.
    """
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    
    # Vectorized quantization: [N, H, D] -> [N, H, D] uint8, then flatten to [N, H*D]
    k_q, v_q = strategy.quantize_kv_for_store(key, value, k_scale=k_scale, v_scale=v_scale)
    key_quantized = k_q.reshape(N, D).contiguous()
    value_quantized = v_q.reshape(N, D).contiguous()
    
    assert key_quantized.dtype == torch.uint8, f"Expected uint8, got {key_quantized.dtype}"
    assert value_quantized.dtype == torch.uint8, f"Expected uint8, got {value_quantized.dtype}"
    assert N == slot_mapping.numel(), f"`N`: {N}, `slot_mapping.numel()`: {slot_mapping.numel()}"
    
    # For unified layout, cache shape is [num_blocks, block_size, num_kv_heads, head_dim]
    # BF16 kernel uses cache directly (no view) and assumes stride(1) == D
    # For FP8, we should do the same to match BF16 behavior
    assert k_cache.stride(1) == D and v_cache.stride(1) == D, \
        f"Expected stride(1) == D ({D}), got k_cache.stride(1)={k_cache.stride(1)}, v_cache.stride(1)={v_cache.stride(1)}"
    
    # Use cache directly, matching BF16 kernel behavior
    # Kernel uses slot * D as offset, which works with stride(1) == D
    # Pass cache directly to kernel, matching BF16 kernel behavior
    # The kernel expects cache to have stride(1) == D, which we've already verified
    dllm_store_kvcache_kernel_unified_fp8[(N,)](
        key_quantized, key_quantized.stride(0),
        value_quantized, value_quantized.stride(0),
        k_cache, v_cache, slot_mapping, D
    )


def _store_kvcache_unified_bf16(key: torch.Tensor, value: torch.Tensor, 
                                 k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                 slot_mapping: torch.Tensor) -> None:
    """Helper function for BF16 unified layout store."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert N == slot_mapping.numel(), f"`N`: {N}, `slot_mapping.numel()`: {slot_mapping.numel()}"
    
    dllm_store_kvcache_kernel_unified_bf16[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )
        

@triton.jit
def load_kvcache_kernel_bf16(k_cache_ptr, v_cache_ptr,
                        k_new_ptr, v_new_ptr,
                        block_table_ptr,
                        k_out_ptr, v_out_ptr, 
                        seqlens_ptr, ctxlens_ptr,
                        cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                        kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d,
                        kv_new_stride_s, kv_new_stride_h, kv_new_stride_d,
                        block_table_stride_nseqs, block_table_stride_maxblks,
                        kv_out_stride_s, kv_out_stride_h, kv_out_stride_d,
                        ctxlens_stride, seqlens_stride,
                        cu_seqlens_q_stride, cu_seqlens_k_stride,
                        LAST_BLK_ID: tl.constexpr,
                        HEAD_DIM: tl.constexpr,
                        PAGE_SIZE: tl.constexpr,
                        DIFFUSION_BLOCK_SIZE: tl.constexpr,
                        KV_LOAD_UNROLL_FACTOR: tl.constexpr):
    # BUG FIX
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-FileCopyrightText: D2F
    
    # Organization: SJTU DENG Lab
    # Author: Drew Jin (JIN. Yijie, @drewjin)
    # Date: 2025-08-01
    # Email: drewjin0827@gmail.com
    # All rights reserved.
    
    seq_idx = tl.program_id(0)
    local_blk_idx = tl.program_id(1)
    kv_head_idx = tl.program_id(2)

    off_local_blk = seq_idx * block_table_stride_nseqs + local_blk_idx * block_table_stride_maxblks
    global_blk_idx = tl.load(block_table_ptr + off_local_blk)
    
    if global_blk_idx != -1:
        off_ctxlen = seq_idx * ctxlens_stride
        global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
        cur_window_sz = (local_blk_idx + 1) * PAGE_SIZE
        prev_window_sz = local_blk_idx * PAGE_SIZE
        local_ctxlen = tl.where(global_ctxlen > cur_window_sz, PAGE_SIZE, global_ctxlen % PAGE_SIZE)
        if global_ctxlen > prev_window_sz:
            # Load KV cache
            offs_kv_cache_seq = tl.arange(0, PAGE_SIZE)
            offs_kv_cache_hdim = tl.arange(0, HEAD_DIM)
            offs_kv_cache = ( # [NBlks, BlkSz, Hkv, Hdim]
                global_blk_idx[None, :] * kv_cache_stride_nblks + # NBlks: BlkId
                offs_kv_cache_seq[None, :] * kv_cache_stride_blk + # BlkSz: TokenIds
                kv_head_idx * kv_cache_stride_h + # Hkv: HeadId
                offs_kv_cache_hdim[:, None] * kv_cache_stride_d # Hdim: HeadDim Elems
            )
            kv_cache_mask = offs_kv_cache_seq[None, :] < local_ctxlen
            k_cache = tl.load(k_cache_ptr + offs_kv_cache, mask=kv_cache_mask, other=0.0)
            v_cache = tl.load(v_cache_ptr + offs_kv_cache, mask=kv_cache_mask, other=0.0)
            
            # Store KV cache into output KV tensors
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_cache_to_out_start_idx = kv_out_start_idx + prev_window_sz
            offs_kv_cache_to_out = ( # [Seq, Hkv, Hdim]
                (cur_kv_cache_to_out_start_idx + offs_kv_cache_seq[None, :]) * kv_out_stride_s + # Seq: TokenIds over Offset
                kv_head_idx * kv_out_stride_h + # Hkv: HeadId
                offs_kv_cache_hdim[:, None] * kv_out_stride_d # Hdim: HeadDim Elems
            )
            tl.store(k_out_ptr + offs_kv_cache_to_out, k_cache, mask=kv_cache_mask)
            tl.store(v_out_ptr + offs_kv_cache_to_out, v_cache, mask=kv_cache_mask)

    # Load and store active KV only once when first meet
    if local_blk_idx == LAST_BLK_ID: 
        # Load KV new
        off_cu_seqlens_q = seq_idx * cu_seqlens_q_stride
        off_seqlens = seq_idx * seqlens_stride
        kv_new_start_idx = tl.load(cu_seqlens_q_ptr + off_cu_seqlens_q)
        active_seqlen = tl.load(seqlens_ptr + off_seqlens)
        offs_kv_new_seq = tl.arange(0, DIFFUSION_BLOCK_SIZE)
        offs_kv_new_hdim = tl.arange(0, HEAD_DIM)
        
        for diff_blk_idx in tl.range(active_seqlen // DIFFUSION_BLOCK_SIZE, loop_unroll_factor=KV_LOAD_UNROLL_FACTOR):
            off_diff_blk = diff_blk_idx * DIFFUSION_BLOCK_SIZE
            cur_kv_new_start_idx = kv_new_start_idx + off_diff_blk
            offs_cur_kv_new_seq = ( # [Seq, Hkv, Hdim]
                (cur_kv_new_start_idx + offs_kv_new_seq[None, :]) * kv_new_stride_s + # Seq: TokenIds over Offset
                kv_head_idx * kv_new_stride_h + # Hkv: HeadId
                offs_kv_new_hdim[:, None] * kv_new_stride_d # Hdim: HeadDim Elems
            )
            k_new = tl.load(k_new_ptr + offs_cur_kv_new_seq)
            v_new = tl.load(v_new_ptr + offs_cur_kv_new_seq)

            # Store KV new into output KV tensors
            off_ctxlen = seq_idx * ctxlens_stride
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_new_to_out_start_idx = global_ctxlen + kv_out_start_idx + off_diff_blk
            offs_cur_kv_new_to_out = ( # [Seq, Hkv, Hdim]
                (cur_kv_new_to_out_start_idx + offs_kv_new_seq[None, :]) * kv_out_stride_s + # Seq: TokenIds over Offset
                kv_head_idx * kv_out_stride_h + # Hkv: HeadId
                offs_kv_new_hdim[:, None] * kv_out_stride_d # Hdim: HeadDim Elems
            )
            tl.store(k_out_ptr + offs_cur_kv_new_to_out, k_new)
            tl.store(v_out_ptr + offs_cur_kv_new_to_out, v_new)


@triton.jit
def load_kvcache_kernel_bf16_distinct(
    k_cache_ptr,
    v_cache_ptr,
    k_new_ptr,
    v_new_ptr,
    block_table_ptr,
    k_out_ptr,
    v_out_ptr,
    seqlens_ptr,
    ctxlens_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    # distinct cache strides
    k_cache_stride_nblks,
    k_cache_stride_h,
    k_cache_stride_dx,
    k_cache_stride_blk_sz,
    k_cache_stride_x,
    v_cache_stride_nblks,
    v_cache_stride_h,
    v_cache_stride_d,
    v_cache_stride_blk_sz,
    # new / out / block_table strides
    kv_new_stride_s,
    kv_new_stride_h,
    kv_new_stride_d,
    block_table_stride_nseqs,
    block_table_stride_maxblks,
    kv_out_stride_s,
    kv_out_stride_h,
    kv_out_stride_d,
    ctxlens_stride,
    seqlens_stride,
    cu_seqlens_q_stride,
    cu_seqlens_k_stride,
    LAST_BLK_ID: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIFFUSION_BLOCK_SIZE: tl.constexpr,
    KV_LOAD_UNROLL_FACTOR: tl.constexpr,
    X: tl.constexpr,
):
    """
    Distinct layout BF16 load kernel.

    Layouts:
    - k_cache: [NBlks, Hkv, HEAD_DIM//X, PAGE_SIZE, X]
    - v_cache: [NBlks, Hkv, HEAD_DIM, PAGE_SIZE]
    """
    seq_idx = tl.program_id(0)
    local_blk_idx = tl.program_id(1)
    kv_head_idx = tl.program_id(2)

    off_local_blk = seq_idx * block_table_stride_nseqs + local_blk_idx * block_table_stride_maxblks
    global_blk_idx = tl.load(block_table_ptr + off_local_blk)

    if global_blk_idx != -1:
        off_ctxlen = seq_idx * ctxlens_stride
        global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
        cur_window_sz = (local_blk_idx + 1) * PAGE_SIZE
        prev_window_sz = local_blk_idx * PAGE_SIZE
        local_ctxlen = tl.where(global_ctxlen > cur_window_sz, PAGE_SIZE, global_ctxlen % PAGE_SIZE)
        if global_ctxlen > prev_window_sz:
            offs_kv_cache_seq = tl.arange(0, PAGE_SIZE)
            offs_kv_cache_hdim = tl.arange(0, HEAD_DIM)

            x_ids = offs_kv_cache_hdim // X
            x_offs = offs_kv_cache_hdim % X

            offs_k = (
                global_blk_idx * k_cache_stride_nblks
                + kv_head_idx * k_cache_stride_h
                + x_ids[:, None] * k_cache_stride_dx
                + offs_kv_cache_seq[None, :] * k_cache_stride_blk_sz
                + x_offs[:, None] * k_cache_stride_x
            )
            offs_v = (
                global_blk_idx * v_cache_stride_nblks
                + kv_head_idx * v_cache_stride_h
                + offs_kv_cache_hdim[:, None] * v_cache_stride_d
                + offs_kv_cache_seq[None, :] * v_cache_stride_blk_sz
            )

            kv_cache_mask = offs_kv_cache_seq[None, :] < local_ctxlen
            k_cache = tl.load(k_cache_ptr + offs_k, mask=kv_cache_mask, other=0.0)
            v_cache = tl.load(v_cache_ptr + offs_v, mask=kv_cache_mask, other=0.0)

            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_cache_to_out_start_idx = kv_out_start_idx + prev_window_sz
            offs_kv_cache_to_out = (
                (cur_kv_cache_to_out_start_idx + offs_kv_cache_seq[None, :]) * kv_out_stride_s
                + kv_head_idx * kv_out_stride_h
                + offs_kv_cache_hdim[:, None] * kv_out_stride_d
            )
            tl.store(k_out_ptr + offs_kv_cache_to_out, k_cache, mask=kv_cache_mask)
            tl.store(v_out_ptr + offs_kv_cache_to_out, v_cache, mask=kv_cache_mask)

    if local_blk_idx == LAST_BLK_ID:
        off_cu_seqlens_q = seq_idx * cu_seqlens_q_stride
        off_seqlens = seq_idx * seqlens_stride
        kv_new_start_idx = tl.load(cu_seqlens_q_ptr + off_cu_seqlens_q)
        active_seqlen = tl.load(seqlens_ptr + off_seqlens)

        offs_kv_new_seq = tl.arange(0, DIFFUSION_BLOCK_SIZE)
        offs_kv_new_hdim = tl.arange(0, HEAD_DIM)

        for diff_blk_idx in tl.range(active_seqlen // DIFFUSION_BLOCK_SIZE, loop_unroll_factor=KV_LOAD_UNROLL_FACTOR):
            off_diff_blk = diff_blk_idx * DIFFUSION_BLOCK_SIZE
            cur_kv_new_start_idx = kv_new_start_idx + off_diff_blk
            offs_cur_kv_new_seq = (
                (cur_kv_new_start_idx + offs_kv_new_seq[None, :]) * kv_new_stride_s
                + kv_head_idx * kv_new_stride_h
                + offs_kv_new_hdim[:, None] * kv_new_stride_d
            )
            k_new = tl.load(k_new_ptr + offs_cur_kv_new_seq)
            v_new = tl.load(v_new_ptr + offs_cur_kv_new_seq)

            off_ctxlen = seq_idx * ctxlens_stride
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_new_to_out_start_idx = global_ctxlen + kv_out_start_idx + off_diff_blk
            offs_cur_kv_new_to_out = (
                (cur_kv_new_to_out_start_idx + offs_kv_new_seq[None, :]) * kv_out_stride_s
                + kv_head_idx * kv_out_stride_h
                + offs_kv_new_hdim[:, None] * kv_out_stride_d
            )
            tl.store(k_out_ptr + offs_cur_kv_new_to_out, k_new)
            tl.store(v_out_ptr + offs_cur_kv_new_to_out, v_new)


@triton.jit
def load_kvcache_kernel_fp8_distinct(
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    k_new_ptr,
    v_new_ptr,
    block_table_ptr,
    k_out_ptr,
    v_out_ptr,
    seqlens_ptr,
    ctxlens_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    # distinct cache strides
    k_cache_stride_nblks,
    k_cache_stride_h,
    k_cache_stride_dx,
    k_cache_stride_blk_sz,
    k_cache_stride_x,
    v_cache_stride_nblks,
    v_cache_stride_h,
    v_cache_stride_d,
    v_cache_stride_blk_sz,
    # new / out / block_table strides
    kv_new_stride_s,
    kv_new_stride_h,
    kv_new_stride_d,
    block_table_stride_nseqs,
    block_table_stride_maxblks,
    kv_out_stride_s,
    kv_out_stride_h,
    kv_out_stride_d,
    ctxlens_stride,
    seqlens_stride,
    cu_seqlens_q_stride,
    cu_seqlens_k_stride,
    LAST_BLK_ID: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIFFUSION_BLOCK_SIZE: tl.constexpr,
    KV_LOAD_UNROLL_FACTOR: tl.constexpr,
    X: tl.constexpr,
):
    """
    Distinct layout FP8 load kernel:
    - Gather paged KV cache blocks from distinct K/V layouts.
    - Dequantize FP8 -> BF16 and apply per-head scale inside kernel.

    Layouts:
    - k_cache: [NBlks, Hkv, HEAD_DIM//X, PAGE_SIZE, X] (float8 view)
    - v_cache: [NBlks, Hkv, HEAD_DIM, PAGE_SIZE] (float8 view)
    """
    seq_idx = tl.program_id(0)
    local_blk_idx = tl.program_id(1)
    kv_head_idx = tl.program_id(2)

    off_local_blk = seq_idx * block_table_stride_nseqs + local_blk_idx * block_table_stride_maxblks
    global_blk_idx = tl.load(block_table_ptr + off_local_blk)

    k_scale = tl.load(k_scale_ptr + kv_head_idx).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + kv_head_idx).to(tl.float32)

    if global_blk_idx != -1:
        off_ctxlen = seq_idx * ctxlens_stride
        global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
        cur_window_sz = (local_blk_idx + 1) * PAGE_SIZE
        prev_window_sz = local_blk_idx * PAGE_SIZE
        local_ctxlen = tl.where(global_ctxlen > cur_window_sz, PAGE_SIZE, global_ctxlen % PAGE_SIZE)
        if global_ctxlen > prev_window_sz:
            offs_kv_cache_seq = tl.arange(0, PAGE_SIZE)
            offs_kv_cache_hdim = tl.arange(0, HEAD_DIM)

            x_ids = offs_kv_cache_hdim // X
            x_offs = offs_kv_cache_hdim % X

            offs_k = (
                global_blk_idx * k_cache_stride_nblks
                + kv_head_idx * k_cache_stride_h
                + x_ids[:, None] * k_cache_stride_dx
                + offs_kv_cache_seq[None, :] * k_cache_stride_blk_sz
                + x_offs[:, None] * k_cache_stride_x
            )
            offs_v = (
                global_blk_idx * v_cache_stride_nblks
                + kv_head_idx * v_cache_stride_h
                + offs_kv_cache_hdim[:, None] * v_cache_stride_d
                + offs_kv_cache_seq[None, :] * v_cache_stride_blk_sz
            )

            kv_cache_mask = offs_kv_cache_seq[None, :] < local_ctxlen
            k_cache = tl.load(k_cache_ptr + offs_k, mask=kv_cache_mask, other=0.0).to(tl.float32) * k_scale
            v_cache = tl.load(v_cache_ptr + offs_v, mask=kv_cache_mask, other=0.0).to(tl.float32) * v_scale
            k_cache_bf16 = k_cache.to(tl.bfloat16)
            v_cache_bf16 = v_cache.to(tl.bfloat16)

            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_cache_to_out_start_idx = kv_out_start_idx + prev_window_sz
            offs_kv_cache_to_out = (
                (cur_kv_cache_to_out_start_idx + offs_kv_cache_seq[None, :]) * kv_out_stride_s
                + kv_head_idx * kv_out_stride_h
                + offs_kv_cache_hdim[:, None] * kv_out_stride_d
            )
            tl.store(k_out_ptr + offs_kv_cache_to_out, k_cache_bf16, mask=kv_cache_mask)
            tl.store(v_out_ptr + offs_kv_cache_to_out, v_cache_bf16, mask=kv_cache_mask)

    if local_blk_idx == LAST_BLK_ID:
        off_cu_seqlens_q = seq_idx * cu_seqlens_q_stride
        off_seqlens = seq_idx * seqlens_stride
        kv_new_start_idx = tl.load(cu_seqlens_q_ptr + off_cu_seqlens_q)
        active_seqlen = tl.load(seqlens_ptr + off_seqlens)

        offs_kv_new_seq = tl.arange(0, DIFFUSION_BLOCK_SIZE)
        offs_kv_new_hdim = tl.arange(0, HEAD_DIM)

        for diff_blk_idx in tl.range(active_seqlen // DIFFUSION_BLOCK_SIZE, loop_unroll_factor=KV_LOAD_UNROLL_FACTOR):
            off_diff_blk = diff_blk_idx * DIFFUSION_BLOCK_SIZE
            cur_kv_new_start_idx = kv_new_start_idx + off_diff_blk
            offs_cur_kv_new_seq = (
                (cur_kv_new_start_idx + offs_kv_new_seq[None, :]) * kv_new_stride_s
                + kv_head_idx * kv_new_stride_h
                + offs_kv_new_hdim[:, None] * kv_new_stride_d
            )
            k_new = tl.load(k_new_ptr + offs_cur_kv_new_seq)
            v_new = tl.load(v_new_ptr + offs_cur_kv_new_seq)

            off_ctxlen = seq_idx * ctxlens_stride
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_new_to_out_start_idx = global_ctxlen + kv_out_start_idx + off_diff_blk
            offs_cur_kv_new_to_out = (
                (cur_kv_new_to_out_start_idx + offs_kv_new_seq[None, :]) * kv_out_stride_s
                + kv_head_idx * kv_out_stride_h
                + offs_kv_new_hdim[:, None] * kv_out_stride_d
            )
            tl.store(k_out_ptr + offs_cur_kv_new_to_out, k_new)
            tl.store(v_out_ptr + offs_cur_kv_new_to_out, v_new)

@triton.jit
def load_kvcache_kernel_fp8_unified(
    k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    k_new_ptr, v_new_ptr,
    block_table_ptr,
    k_out_ptr, v_out_ptr,
    seqlens_ptr, ctxlens_ptr,
    cu_seqlens_q_ptr, cu_seqlens_k_ptr,
    kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d,
    kv_new_stride_s, kv_new_stride_h, kv_new_stride_d,
    block_table_stride_nseqs, block_table_stride_maxblks,
    kv_out_stride_s, kv_out_stride_h, kv_out_stride_d,
    ctxlens_stride, seqlens_stride,
    cu_seqlens_q_stride, cu_seqlens_k_stride,
    LAST_BLK_ID: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    DIFFUSION_BLOCK_SIZE: tl.constexpr,
    KV_LOAD_UNROLL_FACTOR: tl.constexpr,
):
    """
    Unified layout FP8 load kernel:
    - Gather paged KV cache blocks using block_tables/context_lens (same as BF16 kernel)
    - Dequantize FP8 -> BF16 and apply per-head scale inside kernel
    - Also appends active KV (k_new/v_new) once at LAST_BLK_ID
    """
    seq_idx = tl.program_id(0)
    local_blk_idx = tl.program_id(1)
    kv_head_idx = tl.program_id(2)

    off_local_blk = seq_idx * block_table_stride_nseqs + local_blk_idx * block_table_stride_maxblks
    global_blk_idx = tl.load(block_table_ptr + off_local_blk)

    # Per-head scales (float32)
    k_scale = tl.load(k_scale_ptr + kv_head_idx).to(tl.float32)
    v_scale = tl.load(v_scale_ptr + kv_head_idx).to(tl.float32)

    if global_blk_idx != -1:
        off_ctxlen = seq_idx * ctxlens_stride
        global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
        cur_window_sz = (local_blk_idx + 1) * PAGE_SIZE
        prev_window_sz = local_blk_idx * PAGE_SIZE
        local_ctxlen = tl.where(global_ctxlen > cur_window_sz, PAGE_SIZE, global_ctxlen % PAGE_SIZE)
        if global_ctxlen > prev_window_sz:
            offs_kv_cache_seq = tl.arange(0, PAGE_SIZE)
            offs_kv_cache_hdim = tl.arange(0, HEAD_DIM)
            offs_kv_cache = (
                global_blk_idx[None, :] * kv_cache_stride_nblks +
                offs_kv_cache_seq[None, :] * kv_cache_stride_blk +
                kv_head_idx * kv_cache_stride_h +
                offs_kv_cache_hdim[:, None] * kv_cache_stride_d
            )
            kv_cache_mask = offs_kv_cache_seq[None, :] < local_ctxlen

            # Load FP8 -> fp32, apply scale, store BF16
            k_cache = tl.load(k_cache_ptr + offs_kv_cache, mask=kv_cache_mask, other=0.0).to(tl.float32) * k_scale
            v_cache = tl.load(v_cache_ptr + offs_kv_cache, mask=kv_cache_mask, other=0.0).to(tl.float32) * v_scale
            k_cache_bf16 = k_cache.to(tl.bfloat16)
            v_cache_bf16 = v_cache.to(tl.bfloat16)

            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_cache_to_out_start_idx = kv_out_start_idx + prev_window_sz
            offs_kv_cache_to_out = (
                (cur_kv_cache_to_out_start_idx + offs_kv_cache_seq[None, :]) * kv_out_stride_s +
                kv_head_idx * kv_out_stride_h +
                offs_kv_cache_hdim[:, None] * kv_out_stride_d
            )
            tl.store(k_out_ptr + offs_kv_cache_to_out, k_cache_bf16, mask=kv_cache_mask)
            tl.store(v_out_ptr + offs_kv_cache_to_out, v_cache_bf16, mask=kv_cache_mask)

    # Load and store active KV only once when first meet
    if local_blk_idx == LAST_BLK_ID:
        off_cu_seqlens_q = seq_idx * cu_seqlens_q_stride
        off_seqlens = seq_idx * seqlens_stride
        kv_new_start_idx = tl.load(cu_seqlens_q_ptr + off_cu_seqlens_q)
        active_seqlen = tl.load(seqlens_ptr + off_seqlens)

        offs_kv_new_seq = tl.arange(0, DIFFUSION_BLOCK_SIZE)
        offs_kv_new_hdim = tl.arange(0, HEAD_DIM)

        for diff_blk_idx in tl.range(active_seqlen // DIFFUSION_BLOCK_SIZE, loop_unroll_factor=KV_LOAD_UNROLL_FACTOR):
            off_diff_blk = diff_blk_idx * DIFFUSION_BLOCK_SIZE
            cur_kv_new_start_idx = kv_new_start_idx + off_diff_blk
            offs_cur_kv_new_seq = (
                (cur_kv_new_start_idx + offs_kv_new_seq[None, :]) * kv_new_stride_s +
                kv_head_idx * kv_new_stride_h +
                offs_kv_new_hdim[:, None] * kv_new_stride_d
            )
            k_new = tl.load(k_new_ptr + offs_cur_kv_new_seq)
            v_new = tl.load(v_new_ptr + offs_cur_kv_new_seq)

            off_ctxlen = seq_idx * ctxlens_stride
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            global_ctxlen = tl.load(ctxlens_ptr + off_ctxlen)
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_new_to_out_start_idx = global_ctxlen + kv_out_start_idx + off_diff_blk
            offs_cur_kv_new_to_out = (
                (cur_kv_new_to_out_start_idx + offs_kv_new_seq[None, :]) * kv_out_stride_s +
                kv_head_idx * kv_out_stride_h +
                offs_kv_new_hdim[:, None] * kv_out_stride_d
            )
            tl.store(k_out_ptr + offs_cur_kv_new_to_out, k_new)
            tl.store(v_out_ptr + offs_cur_kv_new_to_out, v_new)


def _load_kvcache_bf16(k_cache: torch.Tensor, v_cache: torch.Tensor,
                 attn_metadata: AttnMetaDataBase,
                 k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function for BF16 load.
    
    Supports both unified and distinct layouts:
    - Unified: k_cache.shape == v_cache.shape == [num_blocks, page_size, num_kv_heads, head_dim]
    - Distinct: k_cache.shape = [num_blks, h, hdim // x, blk_sz, x], v_cache.shape = [num_blks, h, hdim, blk_sz]
    """
    assert k_new.shape == v_new.shape
    
    # Determine layout from cache shape
    is_unified = k_cache.shape == v_cache.shape and len(k_cache.shape) == 4
    
    if is_unified:
        # Unified layout: [num_blocks, page_size, num_kv_heads, head_dim]
        N_BLOCKS, PAGE_SIZE, H_KV, HEAD_DIM = k_cache.shape
    else:
        # Distinct layout: k_cache [num_blks, h, hdim // x, blk_sz, x], v_cache [num_blks, h, hdim, blk_sz]
        # For load kernel, we need PAGE_SIZE and HEAD_DIM
        # PAGE_SIZE is typically the block size (blk_sz)
        # HEAD_DIM is the head dimension
        N_BLOCKS = k_cache.shape[0]
        H_KV = k_cache.shape[1]
        PAGE_SIZE = k_cache.shape[3]  # blk_sz
        # For distinct layout, HEAD_DIM is the total head dimension
        # k_cache: [num_blks, h, hdim // x, blk_sz, x] -> HEAD_DIM = (hdim // x) * x
        # v_cache: [num_blks, h, hdim, blk_sz] -> HEAD_DIM = hdim
        HEAD_DIM = v_cache.shape[2]  # hdim
    NUM_SEQS, MAX_SEQ_BLOCKS = attn_metadata.block_tables.shape
    
    ctxlens = attn_metadata.context_lens
    seqlens = attn_metadata.seq_lens_ts
    assert sum(seqlens) == k_new.shape[0]
    DIFFUSION_BLOCK_SIZE = attn_metadata.seqs[0].diffusion_block_size
    MAX_DIFFUSION_BLOCK_SIZE = max(seqlens)
    assert MAX_DIFFUSION_BLOCK_SIZE % DIFFUSION_BLOCK_SIZE == 0
    
    total_lens = ctxlens + seqlens
    cu_seqlens_q = attn_metadata.cu_seqlens_q
    cu_seqlens_k = attn_metadata.cu_seqlens_k
    assert sum(total_lens) == cu_seqlens_k[-1]
    assert cu_seqlens_q.shape == cu_seqlens_k.shape
    assert cu_seqlens_q.shape[0] == NUM_SEQS + 1
    
    kv_output_shape = (sum(total_lens).item(), H_KV, HEAD_DIM)
    k_output = torch.empty(kv_output_shape, device=k_cache.device, dtype=k_cache.dtype)
    v_output = torch.empty_like(k_output)
    
    GRID = (NUM_SEQS, MAX_SEQ_BLOCKS, H_KV)

    if is_unified:
        # Unified cache: [NBlks, BlkSz, Hkv, Hdim]
        kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d = k_cache.stride()
        load_kvcache_kernel_bf16[GRID](
            k_cache, v_cache,
            k_new, v_new,
            attn_metadata.block_tables,
            k_output, v_output,
            seqlens, ctxlens,
            cu_seqlens_q, cu_seqlens_k,
            kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d,
            *k_new.stride(),
            *attn_metadata.block_tables.stride(),
            *k_output.stride(),
            ctxlens.stride(0),
            seqlens.stride(0),
            cu_seqlens_q.stride(0),
            cu_seqlens_k.stride(0),
            LAST_BLK_ID=attn_metadata.block_tables.shape[-1] - 1,
            HEAD_DIM=HEAD_DIM,
            PAGE_SIZE=PAGE_SIZE,
            DIFFUSION_BLOCK_SIZE=DIFFUSION_BLOCK_SIZE,
            KV_LOAD_UNROLL_FACTOR=2,
        )
    else:
        # Distinct cache needs a dedicated gather kernel due to K split layout.
        x = int(k_cache.shape[-1])
        load_kvcache_kernel_bf16_distinct[GRID](
            k_cache, v_cache,
            k_new, v_new,
            attn_metadata.block_tables,
            k_output, v_output,
            seqlens, ctxlens,
            cu_seqlens_q, cu_seqlens_k,
            *k_cache.stride(),
            *v_cache.stride(),
            *k_new.stride(),
            *attn_metadata.block_tables.stride(),
            *k_output.stride(),
            ctxlens.stride(0),
            seqlens.stride(0),
            cu_seqlens_q.stride(0),
            cu_seqlens_k.stride(0),
            LAST_BLK_ID=attn_metadata.block_tables.shape[-1] - 1,
            HEAD_DIM=HEAD_DIM,
            PAGE_SIZE=PAGE_SIZE,
            DIFFUSION_BLOCK_SIZE=DIFFUSION_BLOCK_SIZE,
            KV_LOAD_UNROLL_FACTOR=2,
            X=x,
        )
    
    return k_output, v_output


def store_kvcache_unified_layout(key: torch.Tensor, value: torch.Tensor, 
                                 k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                 slot_mapping: torch.Tensor, attn_metadata: AttnMetaDataBase) -> None:
    """
    Store KV cache (unified layout).
    Dynamically selects the appropriate kernel based on quantization strategy from context.
    """
    from diffulex.utils.quantization.context import get_kv_cache_strategy
    strategy = get_kv_cache_strategy()
    if strategy is None:
        _store_kvcache_unified_bf16(key, value, k_cache, v_cache, slot_mapping)
        return
    
    fmt = getattr(strategy, "kv_cache_format", "bf16")
    if fmt == "bf16":
        _store_kvcache_unified_bf16(key, value, k_cache, v_cache, slot_mapping)
        return
    if fmt == "fp8":
        if attn_metadata.k_scale is None or attn_metadata.v_scale is None:
            raise ValueError("FP8 quantization requires k_scale and v_scale in metadata")
        _store_kvcache_unified_fp8(
            key, value, k_cache, v_cache, slot_mapping,
            attn_metadata.k_scale, attn_metadata.v_scale,
            strategy=strategy,
        )
        return
    raise ValueError(f"Unsupported kv_cache_format={fmt!r} for unified layout (strategy={type(strategy)})")


def store_kvcache_distinct_layout(key: torch.Tensor, value: torch.Tensor, 
                                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                  slot_mapping: torch.Tensor, attn_metadata: AttnMetaDataBase) -> None:
    """
    Store KV cache (distinct layout).
    Dynamically selects the appropriate kernel based on quantization strategy from context.
    """
    from diffulex.utils.quantization.context import get_kv_cache_strategy
    strategy = get_kv_cache_strategy()
    if strategy is None:
        _store_kvcache_distinct_bf16(key, value, k_cache, v_cache, slot_mapping)
        return
    
    fmt = getattr(strategy, "kv_cache_format", "bf16")
    if fmt == "bf16":
        _store_kvcache_distinct_bf16(key, value, k_cache, v_cache, slot_mapping)
        return
    if fmt == "fp8":
        if attn_metadata.k_scale is None or attn_metadata.v_scale is None:
            raise ValueError("FP8 quantization requires k_scale and v_scale in metadata")
        _store_kvcache_distinct_fp8(
            key, value, k_cache, v_cache, slot_mapping,
            attn_metadata.k_scale, attn_metadata.v_scale,
            strategy=strategy,
        )
        return
    raise ValueError(f"Unsupported kv_cache_format={fmt!r} for distinct layout (strategy={type(strategy)})")


def _load_kvcache_fp8(k_cache: torch.Tensor, v_cache: torch.Tensor,
                      attn_metadata: AttnMetaDataBase,
                      k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Helper function for FP8 load.

    Unified layout uses a Triton fused kernel to gather+dequantize+apply-scale on-the-fly.
    Distinct layout also uses a fused kernel (no Python full-cache dequant fallback).
    
    Supports both unified and distinct layouts:
    - Unified: [num_blocks, page_size, num_kv_heads, head_dim]
    - Distinct: k_cache [num_blks, h, hdim // x, blk_sz, x], v_cache [num_blks, h, hdim, blk_sz]
    """
    from diffulex.utils.quantization.context import get_kv_cache_strategy
    strategy = get_kv_cache_strategy()
    if strategy is None or getattr(strategy, "kv_cache_format", "bf16") != "fp8":
        raise ValueError(f"Expected kv_cache_format='fp8', got strategy={type(strategy)}")
    
    # Get scales from metadata
    if attn_metadata.k_scale is None or attn_metadata.v_scale is None:
        raise ValueError("FP8 dequantization requires k_scale and v_scale in metadata")
    
    k_scale = attn_metadata.k_scale  # [num_kv_heads]
    v_scale = attn_metadata.v_scale  # [num_kv_heads]
    
    # Determine layout from cache shape
    # Unified: k_cache.shape == v_cache.shape == [num_blocks, page_size, num_kv_heads, head_dim]
    # Distinct: k_cache.shape = [num_blks, h, hdim // x, blk_sz, x], v_cache.shape = [num_blks, h, hdim, blk_sz]
    is_unified = k_cache.shape == v_cache.shape and len(k_cache.shape) == 4
    
    if is_unified:
        # Unified layout: [num_blocks, page_size, num_kv_heads, head_dim]
        N_BLOCKS, PAGE_SIZE, H_KV, HEAD_DIM = k_cache.shape

        # Ensure Triton sees float8 element type (storage is uint8 view)
        k_cache_fp8 = strategy.view_kv_cache_for_kernels(k_cache)
        v_cache_fp8 = strategy.view_kv_cache_for_kernels(v_cache)

        NUM_SEQS, MAX_SEQ_BLOCKS = attn_metadata.block_tables.shape
        ctxlens = attn_metadata.context_lens
        seqlens = attn_metadata.seq_lens_ts
        assert sum(seqlens) == k_new.shape[0]
        DIFFUSION_BLOCK_SIZE = attn_metadata.seqs[0].diffusion_block_size
        MAX_DIFFUSION_BLOCK_SIZE = max(seqlens)
        assert MAX_DIFFUSION_BLOCK_SIZE % DIFFUSION_BLOCK_SIZE == 0

        total_lens = ctxlens + seqlens
        cu_seqlens_q = attn_metadata.cu_seqlens_q
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        assert sum(total_lens) == cu_seqlens_k[-1]
        assert cu_seqlens_q.shape == cu_seqlens_k.shape
        assert cu_seqlens_q.shape[0] == NUM_SEQS + 1

        kv_output_shape = (sum(total_lens).item(), H_KV, HEAD_DIM)
        k_output = torch.empty(kv_output_shape, device=k_cache.device, dtype=torch.bfloat16)
        v_output = torch.empty_like(k_output)

        # Strides for unified cache: [stride(0), stride(1), stride(2), stride(3)]
        kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d = k_cache_fp8.stride()

        GRID = (NUM_SEQS, MAX_SEQ_BLOCKS, H_KV)
        load_kvcache_kernel_fp8_unified[GRID](
            k_cache_fp8, v_cache_fp8,
            k_scale, v_scale,
            k_new, v_new,
            attn_metadata.block_tables,
            k_output, v_output,
            seqlens, ctxlens,
            cu_seqlens_q, cu_seqlens_k,
            kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d,
            *k_new.stride(),
            *attn_metadata.block_tables.stride(),
            *k_output.stride(),
            ctxlens.stride(0),
            seqlens.stride(0),
            cu_seqlens_q.stride(0),
            cu_seqlens_k.stride(0),
            LAST_BLK_ID=attn_metadata.block_tables.shape[-1] - 1,
            HEAD_DIM=HEAD_DIM,
            PAGE_SIZE=PAGE_SIZE,
            DIFFUSION_BLOCK_SIZE=DIFFUSION_BLOCK_SIZE,
            KV_LOAD_UNROLL_FACTOR=2,
        )

        # Optional correctness check: compare with the old Python dequant+BF16-gather reference
        if os.getenv("DIFFULEX_DEBUG_FP8_LOAD_REF", "0") == "1":
            # Avoid huge overhead accidentally
            try:
                total_tokens = int(sum(total_lens).item())
            except Exception:
                total_tokens = -1
            if 0 <= total_tokens <= 4096:
                # Reference dequantization (slow): full cache dequant in Python
                k_cache_fp32 = k_cache_fp8.float()
                v_cache_fp32 = v_cache_fp8.float()
                k_scale_broadcast = k_scale.view(1, 1, -1, 1)
                v_scale_broadcast = v_scale.view(1, 1, -1, 1)
                k_cache_bf16_ref = (k_cache_fp32 * k_scale_broadcast).to(torch.bfloat16)
                v_cache_bf16_ref = (v_cache_fp32 * v_scale_broadcast).to(torch.bfloat16)
                k_ref, v_ref = _load_kvcache_bf16(k_cache_bf16_ref, v_cache_bf16_ref, attn_metadata, k_new, v_new)
                max_diff_k = (k_ref - k_output).abs().max().item()
                max_diff_v = (v_ref - v_output).abs().max().item()
                print(f"[DIFFULEX_DEBUG_FP8_LOAD_REF] max_abs_diff k={max_diff_k:.6g} v={max_diff_v:.6g} (total_tokens={total_tokens})")
                # Be strict: any mismatch likely indicates indexing/mask/scale bug.
                if max_diff_k > 0 or max_diff_v > 0:
                    raise RuntimeError(
                        f"FP8 fused load mismatch: max_abs_diff k={max_diff_k} v={max_diff_v}. "
                        "Set DIFFULEX_DEBUG_FP8_LOAD_REF=0 to disable."
                    )

        return k_output, v_output
    else:
        # Distinct layout: fused gather + dequant + scale in kernel.
        k_cache_fp8 = strategy.view_kv_cache_for_kernels(k_cache)
        v_cache_fp8 = strategy.view_kv_cache_for_kernels(v_cache)

        NUM_SEQS, MAX_SEQ_BLOCKS = attn_metadata.block_tables.shape
        ctxlens = attn_metadata.context_lens
        seqlens = attn_metadata.seq_lens_ts
        assert sum(seqlens) == k_new.shape[0]
        DIFFUSION_BLOCK_SIZE = attn_metadata.seqs[0].diffusion_block_size
        MAX_DIFFUSION_BLOCK_SIZE = max(seqlens)
        assert MAX_DIFFUSION_BLOCK_SIZE % DIFFUSION_BLOCK_SIZE == 0

        total_lens = ctxlens + seqlens
        cu_seqlens_q = attn_metadata.cu_seqlens_q
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        assert sum(total_lens) == cu_seqlens_k[-1]
        assert cu_seqlens_q.shape == cu_seqlens_k.shape
        assert cu_seqlens_q.shape[0] == NUM_SEQS + 1

        # Distinct cache shapes:
        # k_cache: [NBlks, Hkv, HEAD_DIM//x, PAGE_SIZE, x]
        # v_cache: [NBlks, Hkv, HEAD_DIM, PAGE_SIZE]
        PAGE_SIZE = int(k_cache.shape[3])
        HEAD_DIM = int(v_cache.shape[2])
        H_KV = int(v_cache.shape[1])
        x = int(k_cache.shape[-1])

        kv_output_shape = (sum(total_lens).item(), H_KV, HEAD_DIM)
        k_output = torch.empty(kv_output_shape, device=k_cache.device, dtype=torch.bfloat16)
        v_output = torch.empty_like(k_output)

        GRID = (NUM_SEQS, MAX_SEQ_BLOCKS, H_KV)
        load_kvcache_kernel_fp8_distinct[GRID](
            k_cache_fp8, v_cache_fp8,
            k_scale, v_scale,
            k_new, v_new,
            attn_metadata.block_tables,
            k_output, v_output,
            seqlens, ctxlens,
            cu_seqlens_q, cu_seqlens_k,
            *k_cache_fp8.stride(),
            *v_cache_fp8.stride(),
            *k_new.stride(),
            *attn_metadata.block_tables.stride(),
            *k_output.stride(),
            ctxlens.stride(0),
            seqlens.stride(0),
            cu_seqlens_q.stride(0),
            cu_seqlens_k.stride(0),
            LAST_BLK_ID=attn_metadata.block_tables.shape[-1] - 1,
            HEAD_DIM=HEAD_DIM,
            PAGE_SIZE=PAGE_SIZE,
            DIFFUSION_BLOCK_SIZE=DIFFUSION_BLOCK_SIZE,
            KV_LOAD_UNROLL_FACTOR=2,
            X=x,
        )

        return k_output, v_output


def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                 attn_metadata: AttnMetaDataBase,
                 k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load KV cache.
    Dynamically selects the appropriate kernel based on quantization strategy from context.
    """
    from diffulex.utils.quantization.context import get_kv_cache_strategy
    strategy = get_kv_cache_strategy()
    if strategy is None:
        return _load_kvcache_bf16(k_cache, v_cache, attn_metadata, k_new, v_new)
    
    fmt = getattr(strategy, "kv_cache_format", "bf16")
    if fmt == "bf16":
        return _load_kvcache_bf16(k_cache, v_cache, attn_metadata, k_new, v_new)
    if fmt == "fp8":
        return _load_kvcache_fp8(k_cache, v_cache, attn_metadata, k_new, v_new)
    raise ValueError(f"Unsupported kv_cache_format={fmt!r} for load (strategy={type(strategy)})")