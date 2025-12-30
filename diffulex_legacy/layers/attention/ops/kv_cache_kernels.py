import torch
import triton 

import triton.language as tl

from typing import Tuple
from einops import rearrange

from diffulex_legacy.utils.context import ContextForDiffusionLM 
from diffulex_legacy.engine.sequence import SequenceForDiffusionLM
from diffulex.utils.kv_cache_dtype import (
    KvCacheDType,
    ensure_scale_tensor,
    parse_kv_cache_dtype,
    view_fp8_cache,
)

@triton.jit
def store_kvcache_kernel_causal_lm(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    HEAD_DIM: tl.constexpr,
    D: tl.constexpr,
    KV_CACHE_DTYPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr
):
    idx = tl.program_id(0)
    offs_d = tl.arange(0, D)
    key_offsets = idx * key_stride + offs_d
    value_offsets = idx * value_stride + offs_d
    key = tl.load(key_ptr + key_offsets).to(tl.float32)
    value = tl.load(value_ptr + value_offsets).to(tl.float32)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + offs_d

    # Triton kernels cannot reference Python globals (e.g., Enum). Use constexpr ints.
    if KV_CACHE_DTYPE == 3 or KV_CACHE_DTYPE == 4:
        head_id = offs_d // HEAD_DIM
        k_scale = tl.load(k_scale_ptr + head_id).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + head_id).to(tl.float32)
        k_scale = tl.maximum(k_scale, 1e-8)
        v_scale = tl.maximum(v_scale, 1e-8)
        key = key / k_scale
        value = value / v_scale
        key = tl.maximum(tl.minimum(key, FP8_MAX), FP8_MIN)
        value = tl.maximum(tl.minimum(value, FP8_MAX), FP8_MIN)

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
    k_scale_ptr,
    v_scale_ptr,
    HEAD_DIM: tl.constexpr,
    D: tl.constexpr,
    KV_CACHE_DTYPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr
):
    token_idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    offs_d = tl.arange(0, D)
    key_offsets = token_idx * key_stride + offs_d
    value_offsets = token_idx * value_stride + offs_d
    key = tl.load(key_ptr + key_offsets).to(tl.float32)
    value = tl.load(value_ptr + value_offsets).to(tl.float32)
    cache_offsets = slot * D + offs_d

    if KV_CACHE_DTYPE == 3 or KV_CACHE_DTYPE == 4:
        head_id = offs_d // HEAD_DIM
        k_scale = tl.load(k_scale_ptr + head_id).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + head_id).to(tl.float32)
        k_scale = tl.maximum(k_scale, 1e-8)
        v_scale = tl.maximum(v_scale, 1e-8)
        key = key / k_scale
        value = value / v_scale
        key = tl.maximum(tl.minimum(key, FP8_MAX), FP8_MIN)
        value = tl.maximum(tl.minimum(value, FP8_MAX), FP8_MIN)

    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def store_kvcache_kernel_diffusion_lm_distinct(
    k_ptr, v_ptr, k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
    k_stride, v_stride,  
    k_cache_stride_nblks, k_cache_stride_h, k_cache_stride_dx, k_cache_stride_blk_sz, k_cache_stride_x,
    v_cache_stride_nblks, v_cache_stride_h, v_cache_stride_d, v_cache_stride_blk_sz,
    k_scale_ptr, v_scale_ptr,
    nheads, hdim, blk_sz,
    x: tl.constexpr,
    D: tl.constexpr,
    KV_CACHE_DTYPE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr
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
    k = tl.load(k_ptr + offs_k).to(tl.float32)
    v = tl.load(v_ptr + offs_v).to(tl.float32)

    h_ids = offs_d // hdim
    h_offs = offs_d % hdim
    x_ids = h_offs // x
    x_offs = h_offs % x
    
    k_cache_offs = (blk_idx * k_cache_stride_nblks + h_ids * k_cache_stride_h +
                    x_ids * k_cache_stride_dx + off_blk * k_cache_stride_blk_sz + 
                    x_offs * k_cache_stride_x)
    v_cache_offs = (blk_idx * v_cache_stride_nblks + h_ids * v_cache_stride_h +
                    h_offs * v_cache_stride_d + off_blk * v_cache_stride_blk_sz)
    
    if KV_CACHE_DTYPE == 3 or KV_CACHE_DTYPE == 4:
        k_scale = tl.load(k_scale_ptr + h_ids).to(tl.float32)
        v_scale = tl.load(v_scale_ptr + h_ids).to(tl.float32)
        k_scale = tl.maximum(k_scale, 1e-8)
        v_scale = tl.maximum(v_scale, 1e-8)
        k = k / k_scale
        v = v / v_scale
        k = tl.maximum(tl.minimum(k, FP8_MAX), FP8_MIN)
        v = tl.maximum(tl.minimum(v, FP8_MAX), FP8_MIN)
    
    tl.store(k_cache_ptr + k_cache_offs, k)
    tl.store(v_cache_ptr + v_cache_offs, v)
    

def store_kvcache_distinct_layout(key: torch.Tensor, value: torch.Tensor, 
                                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                  slot_mapping: torch.Tensor, model_type: str = 'causal_lm',
                                  kv_cache_dtype: str = "bf16",
                                  k_scale=None,
                                  v_scale=None,
                                  context: ContextForDiffusionLM = None) -> None:
    spec = parse_kv_cache_dtype(kv_cache_dtype)
    
    if model_type == 'causal_lm':
        # k_cache: [num_blks, blk_sz, h, hdim]
        # v_cache: [num_blks, blk_sz, h, hdim]
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert N == slot_mapping.numel()
        k_cache_view = view_fp8_cache(k_cache, kv_cache_dtype)
        v_cache_view = view_fp8_cache(v_cache, kv_cache_dtype)
        k_scale_ts = ensure_scale_tensor(k_scale, num_kv_heads=num_heads, device=key.device)
        v_scale_ts = ensure_scale_tensor(v_scale, num_kv_heads=num_heads, device=key.device)
        store_kvcache_kernel_causal_lm[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache_view, v_cache_view, slot_mapping,
            k_scale_ts, v_scale_ts,
            head_dim,
            D,
            int(spec.enum),
            float(spec.fp8_min or 0.0),
            float(spec.fp8_max or 0.0),
        )
    else:
        # TODO: implement diffusion lm kv cache store
        # k_cache: [num_blks, h, hdim // x, blk_sz, x]
        # v_cache: [num_blks, h, hdim, blk_sz]
        NBlks, NHeads, HDim_x, Blk_sz, x = k_cache.shape
        HDim = HDim_x * x
        N = key.shape[0]
        assert HDim == key.shape[-1] and NHeads == key.shape[1]
        assert N == slot_mapping.numel()
        k_cache_view = view_fp8_cache(k_cache, kv_cache_dtype)
        v_cache_view = view_fp8_cache(v_cache, kv_cache_dtype)
        k_scale_ts = ensure_scale_tensor(k_scale, num_kv_heads=NHeads, device=key.device)
        v_scale_ts = ensure_scale_tensor(v_scale, num_kv_heads=NHeads, device=key.device)
        
        GRID = (N, )
        store_kvcache_kernel_diffusion_lm_distinct[GRID](
            key, value,
            k_cache_view, v_cache_view,
            slot_mapping,
            key.stride(0), value.stride(0), 
            *k_cache.stride(), *v_cache.stride(),
            k_scale_ts, v_scale_ts,
            NHeads, HDim, Blk_sz,
            x,
            HDim * NHeads,
            int(spec.enum),
            float(spec.fp8_min or 0.0),
            float(spec.fp8_max or 0.0),
        )


def store_kvcache_unified_layout(key: torch.Tensor, value: torch.Tensor, 
                                 k_cache: torch.Tensor, v_cache: torch.Tensor, 
                                 slot_mapping: torch.Tensor, model_type: str = 'causal_lm', 
                                 kv_cache_dtype: str = "bf16",
                                 k_scale=None,
                                 v_scale=None,
                                 context: ContextForDiffusionLM = None) -> None:
    spec = parse_kv_cache_dtype(kv_cache_dtype)
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert N == slot_mapping.numel(), f"`N`: {N}, `slot_mapping.numel()`: {slot_mapping.numel()}"
    k_cache_view = view_fp8_cache(k_cache, kv_cache_dtype)
    v_cache_view = view_fp8_cache(v_cache, kv_cache_dtype)
    k_scale_ts = ensure_scale_tensor(k_scale, num_kv_heads=num_heads, device=key.device)
    v_scale_ts = ensure_scale_tensor(v_scale, num_kv_heads=num_heads, device=key.device)

    if model_type == 'causal_lm':
        store_kvcache_kernel_causal_lm[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache_view, v_cache_view, slot_mapping,
            k_scale_ts, v_scale_ts,
            head_dim,
            D,
            int(spec.enum),
            float(spec.fp8_min or 0.0),
            float(spec.fp8_max or 0.0),
        )
    elif model_type == 'diffusion_lm':
        store_kvcache_kernel_diffusion_lm[(N,)](
            key, key.stride(0),
            value, value.stride(0),
            k_cache_view, v_cache_view, slot_mapping,
            k_scale_ts, v_scale_ts,
            head_dim,
            D,
            int(spec.enum),
            float(spec.fp8_min or 0.0),
            float(spec.fp8_max or 0.0),
        )
        

@triton.jit
def load_kvcache_kernel_kv(k_cache_ptr, v_cache_ptr,
                           k_new_ptr, v_new_ptr,
                           block_table_ptr,
                           k_out_ptr, v_out_ptr, 
                           seqlens_ptr, ctxlens_ptr,
                           cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                           k_scale_ptr, v_scale_ptr,
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
                           KV_CACHE_DTYPE: tl.constexpr,
                           OUT_DTYPE: tl.constexpr,
                           FP8_MIN: tl.constexpr,
                           FP8_MAX: tl.constexpr):
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

            # If output is FP8, don't dequantize (keep FP8)
            # If output is not FP8, dequantize to float32 first
            if OUT_DTYPE == 3 or OUT_DTYPE == 4:
                # FP8 output: store directly without dequantization
                k_cache_out = k_cache
                v_cache_out = v_cache
            else:
                # Non-FP8 output: dequantize if needed
                if KV_CACHE_DTYPE == 3 or KV_CACHE_DTYPE == 4:
                    k_cache = k_cache.to(tl.float32)
                    v_cache = v_cache.to(tl.float32)
                    k_scale = tl.load(k_scale_ptr + kv_head_idx).to(tl.float32)
                    v_scale = tl.load(v_scale_ptr + kv_head_idx).to(tl.float32)
                    k_cache_out = k_cache * k_scale
                    v_cache_out = v_cache * v_scale
                else:
                    k_cache_out = k_cache.to(tl.float32)
                    v_cache_out = v_cache.to(tl.float32)
            
            # Store KV cache into output KV tensors
            off_cu_seqlens_k = seq_idx * cu_seqlens_k_stride
            kv_out_start_idx = tl.load(cu_seqlens_k_ptr + off_cu_seqlens_k)
            cur_kv_cache_to_out_start_idx = kv_out_start_idx + prev_window_sz
            offs_kv_cache_to_out = ( # [Seq, Hkv, Hdim]
                (cur_kv_cache_to_out_start_idx + offs_kv_cache_seq[None, :]) * kv_out_stride_s + # Seq: TokenIds over Offset
                kv_head_idx * kv_out_stride_h + # Hkv: HeadId
                offs_kv_cache_hdim[:, None] * kv_out_stride_d # Hdim: HeadDim Elems
            )
            tl.store(k_out_ptr + offs_kv_cache_to_out, k_cache_out, mask=kv_cache_mask)
            tl.store(v_out_ptr + offs_kv_cache_to_out, v_cache_out, mask=kv_cache_mask)

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
            # IMPORTANT:
            # - When OUT_DTYPE is FP8, the output K/V are consumed by FP8 attention kernels
            #   which assume K/V are in the *quantized domain* (value / scale).
            # - Cached K/V are already stored in quantized FP8 domain.
            # - But k_new/v_new are BF16/FP16 values. If we store them directly into FP8 output,
            #   they would be cast without dividing by scale, causing a scale mismatch and large errors.
            if OUT_DTYPE == 3 or OUT_DTYPE == 4:
                if KV_CACHE_DTYPE == 3 or KV_CACHE_DTYPE == 4:
                    k_s = tl.load(k_scale_ptr + kv_head_idx).to(tl.float32)
                    v_s = tl.load(v_scale_ptr + kv_head_idx).to(tl.float32)
                    k_s = tl.maximum(k_s, 1e-8)
                    v_s = tl.maximum(v_s, 1e-8)
                    k_new_q = (k_new.to(tl.float32) / k_s)
                    v_new_q = (v_new.to(tl.float32) / v_s)
                else:
                    # If cache isn't FP8, treat scale as 1.0 for quantized output.
                    k_new_q = k_new.to(tl.float32)
                    v_new_q = v_new.to(tl.float32)
                # Clamp to FP8 representable range before storing
                k_new_q = tl.maximum(tl.minimum(k_new_q, FP8_MAX), FP8_MIN)
                v_new_q = tl.maximum(tl.minimum(v_new_q, FP8_MAX), FP8_MIN)
                tl.store(k_out_ptr + offs_cur_kv_new_to_out, k_new_q)
                tl.store(v_out_ptr + offs_cur_kv_new_to_out, v_new_q)
            else:
                tl.store(k_out_ptr + offs_cur_kv_new_to_out, k_new)
                tl.store(v_out_ptr + offs_cur_kv_new_to_out, v_new)


def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                 context: ContextForDiffusionLM,
                 k_new: torch.Tensor, v_new: torch.Tensor,
                 kv_cache_dtype: str = "bf16",
                 k_scale=None,
                 v_scale=None,
                 out_dtype: torch.dtype | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    spec = parse_kv_cache_dtype(kv_cache_dtype)
    assert k_cache.shape == v_cache.shape
    assert k_new.shape == v_new.shape
    k_cache_view = view_fp8_cache(k_cache, kv_cache_dtype)
    v_cache_view = view_fp8_cache(v_cache, kv_cache_dtype)
    N_BLOCKS, PAGE_SIZE, H_KV, HEAD_DIM = k_cache_view.shape
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
    out_dtype = k_new.dtype if out_dtype is None else out_dtype
    
    # Determine OUT_DTYPE for kernel (constexpr int)
    from diffulex.utils.kv_cache_dtype import KvCacheDType
    if out_dtype == torch.bfloat16:
        out_dtype_enum = int(KvCacheDType.BF16)  # 0
    elif out_dtype == torch.float16:
        out_dtype_enum = int(KvCacheDType.FP16)  # 1
    elif out_dtype == torch.float32:
        out_dtype_enum = int(KvCacheDType.FP32)  # 2
    elif spec.is_fp8 and out_dtype == spec.fp8_view_dtype:
        out_dtype_enum = int(spec.enum)  # 3 or 4
    else:
        # Default: use k_new.dtype
        if k_new.dtype == torch.bfloat16:
            out_dtype_enum = int(KvCacheDType.BF16)
        elif k_new.dtype == torch.float16:
            out_dtype_enum = int(KvCacheDType.FP16)
        elif k_new.dtype == torch.float32:
            out_dtype_enum = int(KvCacheDType.FP32)
        else:
            raise ValueError(f"Unsupported out_dtype: {out_dtype}")
    
    k_output = torch.empty(kv_output_shape, device=k_cache.device, dtype=out_dtype)
    v_output = torch.empty_like(k_output)
    k_scale_ts = ensure_scale_tensor(k_scale, num_kv_heads=H_KV, device=k_cache.device)
    v_scale_ts = ensure_scale_tensor(v_scale, num_kv_heads=H_KV, device=k_cache.device)
    
    GRID = (NUM_SEQS, MAX_SEQ_BLOCKS, H_KV)
    load_kvcache_kernel_kv[GRID](
        k_cache_view, v_cache_view,
        k_new, v_new,
        context.block_tables,
        k_output, v_output,
        seqlens, ctxlens,
        cu_seqlens_q, cu_seqlens_k,
        k_scale_ts, v_scale_ts,
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
        KV_LOAD_UNROLL_FACTOR=2,
        KV_CACHE_DTYPE=int(spec.enum),
        OUT_DTYPE=out_dtype_enum,
        FP8_MIN=float(spec.fp8_min) if spec.is_fp8 and spec.fp8_min is not None else 0.0,
        FP8_MAX=float(spec.fp8_max) if spec.is_fp8 and spec.fp8_max is not None else 0.0,
    )
    
    return k_output, v_output


def CHECK_STORING(k_cache: torch.Tensor, v_cache: torch.Tensor,
                  k: torch.Tensor, v: torch.Tensor,
                  context: ContextForDiffusionLM) -> None:
    # FP8 cache uses uint8 storage; exact bitwise match is not expected here.
    if k_cache.dtype == torch.uint8 or v_cache.dtype == torch.uint8:
        return
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
                  context: ContextForDiffusionLM) -> Tuple[torch.Tensor, torch.Tensor]:
    if k_cache.dtype == torch.uint8 or v_cache.dtype == torch.uint8:
        return k_comb, v_comb
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