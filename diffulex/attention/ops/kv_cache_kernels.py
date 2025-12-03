import torch
import triton 

import triton.language as tl

from typing import Any


@triton.jit
def store_kvcache_kernel_causal_lm(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
    

@triton.jit
def store_kvcache_kernel_diffusion_lm(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr
):
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
def store_kvcache_kernel_diffusion_lm_distinct(
    k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    k_stride, v_stride,  
    k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
    v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
    nheads, hdim, blk_sz,
    x: tl.constexpr, D: tl.constexpr
):  
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
    

def store_kvcache_distinct_layout(key: torch.Tensor, value: torch.Tensor, 
                                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                  slot_mapping: torch.Tensor,
                                  context = None) -> None:
    # k_cache: [num_blks, h, hdim // x, blk_sz, x]
    # v_cache: [num_blks, h, hdim, blk_sz]
    NBlks, NHeads, HDim_x, Blk_sz, x = k_cache.shape
    HDim = HDim_x * x
    N = key.shape[0]
    assert HDim == key.shape[-1] and NHeads == key.shape[1]
    assert N == slot_mapping.numel()
    
    GRID = (N, )
    store_kvcache_kernel_diffusion_lm_distinct[GRID](
        key, value,
        k_cache, v_cache,
        slot_mapping,
        key.stride(0), value.stride(0), 
        *k_cache.stride(), *v_cache.stride(),
        NHeads, HDim, Blk_sz,
        x, HDim * NHeads
    )


def store_kvcache_unified_layout(key: torch.Tensor, value: torch.Tensor, 
                                 k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                 slot_mapping: torch.Tensor,
                                 context: Any = None) -> None:
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert N == slot_mapping.numel(), f"`N`: {N}, `slot_mapping.numel()`: {slot_mapping.numel()}"

    store_kvcache_kernel_diffusion_lm[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )
        

@triton.jit
def load_kvcache_kernel_kv(k_cache_ptr, v_cache_ptr,
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


def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                 context: Any,
                 k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert k_cache.shape == v_cache.shape
    assert k_new.shape == v_new.shape
    N_BLOCKS, PAGE_SIZE, H_KV, HEAD_DIM = k_cache.shape
    NUM_SEQS, MAX_SEQ_BLOCKS = context.block_tables.shape
    
    ctxlens = context.context_lens
    seqlens = context.seq_lens_ts
    assert sum(seqlens) == k_new.shape[0]
    DIFFUSION_BLOCK_SIZE = context.seqs[0].diffusion_block_size
    MAX_DIFFUSION_BLOCK_SIZE = max(seqlens)
    assert MAX_DIFFUSION_BLOCK_SIZE % DIFFUSION_BLOCK_SIZE == 0
    
    total_lens = ctxlens + seqlens
    cu_seqlens_q = context.cu_seqlens_q
    cu_seqlens_k = context.cu_seqlens_k
    assert sum(total_lens) == cu_seqlens_k[-1]
    assert cu_seqlens_q.shape == cu_seqlens_k.shape
    assert cu_seqlens_q.shape[0] == NUM_SEQS + 1
    
    kv_output_shape = (sum(total_lens).item(), H_KV, HEAD_DIM)
    k_output = torch.empty(kv_output_shape, device=k_cache.device, dtype=k_cache.dtype)
    v_output = torch.empty_like(k_output)
    
    GRID = (NUM_SEQS, MAX_SEQ_BLOCKS, H_KV)
    load_kvcache_kernel_kv[GRID](
        k_cache, v_cache,
        k_new, v_new,
        context.block_tables,
        k_output, v_output,
        seqlens, ctxlens,
        cu_seqlens_q, cu_seqlens_k,
        *k_cache.stride(),
        *k_new.stride(),
        *context.block_tables.stride(),
        *k_output.stride(),
        ctxlens.stride(0),
        seqlens.stride(0),
        cu_seqlens_q.stride(0),
        cu_seqlens_k.stride(0),
        LAST_BLK_ID=context.block_tables.shape[-1] - 1,
        HEAD_DIM=HEAD_DIM,
        PAGE_SIZE=PAGE_SIZE,
        DIFFUSION_BLOCK_SIZE=DIFFUSION_BLOCK_SIZE,
        KV_LOAD_UNROLL_FACTOR=2
    )
    
    return k_output, v_output


def CHECK_STORING(k_cache: torch.Tensor, v_cache: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor,
                  context) -> None:
    k_list, v_list = [torch.split(tensor, context.seq_lens, dim=0) for tensor in (k, v)]
    for seq_idx, seq in enumerate(context.seqs):
        cached_num_tokens = seq.cached_num_tokens
        caching_num_tokens = seq.caching_num_tokens
        block_size = seq.block_size
        if caching_num_tokens == 0:
            continue

        k_cache_list, v_cache_list = [], []
        for local_mem_blk_idx, global_mem_blk_idx in enumerate(context.block_tables[seq_idx]):
            if caching_num_tokens == 0:
                break
            
            if global_mem_blk_idx.item() == -1:
                continue
            
            if cached_num_tokens > block_size:
                cached_num_tokens -= block_size
                continue
            
            cur_start_idx = cached_num_tokens % block_size
            remain_num_tokens = min(block_size - cur_start_idx, caching_num_tokens)
            k_cache_list.append(k_cache[global_mem_blk_idx, cur_start_idx:cur_start_idx + remain_num_tokens])
            v_cache_list.append(v_cache[global_mem_blk_idx, cur_start_idx:cur_start_idx + remain_num_tokens])
            cached_num_tokens += remain_num_tokens
            caching_num_tokens -= remain_num_tokens
        k_cache_temp = torch.cat(k_cache_list, dim=0)
        v_cache_temp = torch.cat(v_cache_list, dim=0)
        assert torch.allclose(k_cache_temp, k_list[seq_idx][:seq.caching_num_tokens], atol=1e-5), f"K cache mismatch for seq {seq_idx}!"
        assert torch.allclose(v_cache_temp, v_list[seq_idx][:seq.caching_num_tokens], atol=1e-5), f"V cache mismatch for seq {seq_idx}!"
        

def CHECK_LOADING(k_comb: torch.Tensor, v_comb: torch.Tensor,
                  k_new: torch.Tensor, v_new: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  context: Any) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        k_list, v_list = [torch.split(tensor, context.seq_lens, dim=0) for tensor in (k_new, v_new)]
        cat_k_list = []
        cat_v_list = []
        for seq_idx, (k, v) in enumerate(zip(k_list, v_list)):
            cur_ctxlen = context.context_lens[seq_idx]
            k_cache_temp, v_cache_temp = None, None
            for mem_block_idx in context.block_tables[seq_idx]:
                if mem_block_idx.item() == -1:
                    continue
                k_mem_block, v_mem_block = k_cache[mem_block_idx], v_cache[mem_block_idx]
                mem_block_size = k_cache.shape[1]
                cur_window = mem_block_size if mem_block_size <= cur_ctxlen else cur_ctxlen % mem_block_size
                cur_ctxlen = cur_ctxlen - cur_window
                k_cache_temp = k_mem_block[:cur_window] if k_cache_temp is None \
                    else torch.cat((k_cache_temp, k_mem_block[:cur_window]), dim=0)
                v_cache_temp = v_mem_block[:cur_window] if v_cache_temp is None \
                    else torch.cat((v_cache_temp, v_mem_block[:cur_window]), dim=0)
            cat_k_list.extend([k_cache_temp, k])
            cat_v_list.extend([v_cache_temp, v])
        k_cache_check, v_cache_check = torch.cat(cat_k_list, dim=0), torch.cat(cat_v_list, dim=0)
        assert torch.allclose(k_comb, k_cache_check, atol=1e-5), "K cache mismatch!"
        assert torch.allclose(v_comb, v_cache_check, atol=1e-5), "V cache mismatch!"
        return k_comb, v_comb
    except AssertionError as e:
        raise ValueError(f"KV cache loading check failed: {e}")
        # return k_cache_check, v_cache_check