import time
import os
import csv
import torch
import triton

import torch.nn as nn
import triton.language as tl

from typing import List, Tuple
from functools import lru_cache, partial
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

is_rtx_xx90 = lambda x: "4090" in x or "3090" in x
if is_rtx_xx90(torch.cuda.get_device_name(0)):
    # Placeholder for non-flash attention implementation
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
else:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from diffulex.legacy.utils.context import (
    ContextForCausalLM, ContextForDiffusionLM, 
    get_context_causal_lm, get_context_diffusion_lm
)


@triton.jit
def store_kvcache_kernel(key_ptr,
                         key_stride,
                         value_ptr,
                         value_stride,
                         k_cache_ptr,
                         v_cache_ptr,
                         slot_mapping_ptr,
                         D: tl.constexpr):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, 
                  k_cache: torch.Tensor, v_cache: torch.Tensor, 
                  slot_mapping: torch.Tensor, model_type: str = 'causal_lm') -> None:
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D

    N = slot_mapping.numel() if model_type == 'diffusion_lm' else N
    assert N == slot_mapping.numel()

    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )


@triton.jit
def load_kvcache_kernel_kv_both(k_cache_ptr, v_cache_ptr,
                                k_new_ptr, v_new_ptr,
                                block_table_ptr,
                                k_out_ptr, v_out_ptr, 
                                seq_lens_ptr, ctx_lens_ptr,
                                cu_seqlens_q_ptr, cu_seqlens_k_ptr,
                                kv_cache_stride_nblks, kv_cache_stride_blk, kv_cache_stride_h, kv_cache_stride_d,
                                kv_new_stride_s, kv_new_stride_h, kv_new_stride_d,
                                block_table_stride_nseqs, block_table_stride_maxblks,
                                kv_out_stride_s, kv_out_stride_h, kv_out_stride_d,
                                ctx_lens_stride, seq_lens_stride,
                                cu_seqlens_q_stride, cu_seqlens_k_stride,
                                HEAD_DIM: tl.constexpr,
                                MEM_BLOCK_SIZE: tl.constexpr,
                                DIFFUSION_BLOCK_SIZE: tl.constexpr):
    seq_idx = tl.program_id(0)
    inseq_blk_idx = tl.program_id(1)
    head_idx = tl.program_id(2)

    offset_inseq_blk = seq_idx * block_table_stride_nseqs + inseq_blk_idx * block_table_stride_maxblks
    global_blk_idx = tl.load(block_table_ptr + offset_inseq_blk)
    if global_blk_idx != -1:
        ctx_len_offset = seq_idx * ctx_lens_stride
        ctx_len = tl.load(ctx_lens_ptr + ctx_len_offset)
        included_blk_sz = MEM_BLOCK_SIZE * inseq_blk_idx
        cur_ctx_len = MEM_BLOCK_SIZE if ctx_len - included_blk_sz >= MEM_BLOCK_SIZE else ctx_len - included_blk_sz
        
        # Load KV cache
        kv_cache_seq_offsets = tl.arange(0, MEM_BLOCK_SIZE)
        kv_cache_hdim_offsets = tl.arange(0, HEAD_DIM)
        kv_cache_offsets = ( # [NBlks, BlkSz, Hkv, Hdim]
            global_blk_idx * kv_cache_stride_nblks + # NBlks: BlkId
            kv_cache_seq_offsets[:, None] * kv_cache_stride_blk + # BlkSz: TokenIds
            head_idx * kv_cache_stride_h + # Hkv: HeadId
            kv_cache_hdim_offsets[None, :] * kv_cache_stride_d # Hdim: HeadDim Elems
        )
        k_cache_ptrs = k_cache_ptr + kv_cache_offsets
        v_cache_ptrs = v_cache_ptr + kv_cache_offsets
        kv_cache_mask = kv_cache_seq_offsets[:, None] < cur_ctx_len
        k_cache = tl.load(k_cache_ptrs, mask=kv_cache_mask, other=0.0)
        v_cache = tl.load(v_cache_ptrs, mask=kv_cache_mask, other=0.0)
        
        # Store KV cache into output KV tensors
        cu_seqlens_k_offset = seq_idx * cu_seqlens_k_stride
        kv_out_start_idx = tl.load(cu_seqlens_k_ptr + cu_seqlens_k_offset)
        cur_kv_cache_to_out_start_idx = kv_out_start_idx + included_blk_sz
        kv_cache_to_out_offsets = ( # [Seq, Hkv, Hdim]
            (cur_kv_cache_to_out_start_idx + kv_cache_seq_offsets[:, None]) * kv_out_stride_s + # Seq: TokenIds over Offset
            head_idx * kv_out_stride_h + # Hkv: HeadId
            kv_cache_hdim_offsets[None, :] * kv_out_stride_d # Hdim: HeadDim Elems
        )
        k_cache_toout_ptrs = k_out_ptr + kv_cache_to_out_offsets
        v_cache_toout_ptrs = v_out_ptr + kv_cache_to_out_offsets
        tl.store(k_cache_toout_ptrs, k_cache, mask=kv_cache_mask)
        tl.store(v_cache_toout_ptrs, v_cache, mask=kv_cache_mask)

        # Load and store active KV only once when first meet
        if inseq_blk_idx == 0: 
            # Load KV new
            cu_seqlens_q_offset = seq_idx * cu_seqlens_q_stride
            seq_lens_offset = seq_idx * seq_lens_stride
            kv_new_start_idx = tl.load(cu_seqlens_q_ptr + cu_seqlens_q_offset)
            active_seq_len = tl.load(seq_lens_ptr + seq_lens_offset)
            kv_new_seq_offsets = tl.arange(0, DIFFUSION_BLOCK_SIZE)
            kv_new_hdim_offsets = kv_cache_hdim_offsets
            for diff_blk_idx in range(active_seq_len // DIFFUSION_BLOCK_SIZE):
                diff_blk_offset = diff_blk_idx * DIFFUSION_BLOCK_SIZE
                cur_kv_new_start_idx = kv_new_start_idx + diff_blk_offset
                cur_kv_new_seq_offsets = ( # [Seq, Hkv, Hdim]
                    (cur_kv_new_start_idx + kv_new_seq_offsets[:, None]) * kv_new_stride_s + # Seq: TokenIds over Offset
                    head_idx * kv_new_stride_h + # Hkv: HeadId
                    kv_new_hdim_offsets[None, :] * kv_new_stride_d # Hdim: HeadDim Elems
                )
                k_new_ptrs = k_new_ptr + cur_kv_new_seq_offsets
                v_new_ptrs = v_new_ptr + cur_kv_new_seq_offsets
                k_new = tl.load(k_new_ptrs)
                v_new = tl.load(v_new_ptrs)

                # Store KV new into output KV tensors
                cur_kv_new_to_out_start_idx = ctx_len + kv_out_start_idx + diff_blk_offset
                cur_kv_new_to_out_offsets = ( # [Seq, Hkv, Hdim]
                    (cur_kv_new_to_out_start_idx + kv_new_seq_offsets[:, None]) * kv_out_stride_s + # Seq: TokenIds over Offset
                    head_idx * kv_out_stride_h + # Hkv: HeadId
                    kv_new_hdim_offsets[None, :] * kv_out_stride_d # Hdim: HeadDim Elems
                )
                k_new_out_ptrs = k_out_ptr + cur_kv_new_to_out_offsets
                v_new_out_ptrs = v_out_ptr + cur_kv_new_to_out_offsets
                tl.store(k_new_out_ptrs, k_new)
                tl.store(v_new_out_ptrs, v_new)
            

def load_kvcache(k_cache: torch.Tensor, v_cache: torch.Tensor,
                 context: ContextForDiffusionLM,
                 k_new: torch.Tensor, v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert k_cache.shape == v_cache.shape
    assert k_new.shape == v_new.shape
    N_BLOCKS, MEM_BLOCK_SIZE, H_KV, HEAD_DIM = k_cache.shape
    NUM_SEQS, MAX_SEQ_BLOCKS = context.block_tables.shape
    
    ctx_lens = context.context_lens
    seq_lens = context.seq_lens_ts
    DIFFUSION_BLOCK_SIZE = context.seqs[0].diffusion_block_size
    MAX_DIFFUSION_BLOCK_SIZE = max(seq_lens)
    assert MAX_DIFFUSION_BLOCK_SIZE % DIFFUSION_BLOCK_SIZE == 0
    
    total_lens = ctx_lens + seq_lens
    cu_seqlens_q = context.cu_seqlens_q
    cu_seqlens_k = context.cu_seqlens_k
    kv_output_shape = (sum(total_lens), H_KV, HEAD_DIM)
    k_output = torch.empty(kv_output_shape, device=k_cache.device, dtype=k_cache.dtype)
    v_output = torch.empty_like(k_output)
    
    GRID = (NUM_SEQS, MAX_SEQ_BLOCKS, H_KV)
    load_kvcache_kernel_kv_both[GRID](
        k_cache, v_cache,
        k_new, v_new,
        context.block_tables,
        k_output, v_output,
        seq_lens, ctx_lens,
        cu_seqlens_q, cu_seqlens_k,
        *k_cache.stride(),
        *k_new.stride(),
        *context.block_tables.stride(),
        *k_output.stride(),
        ctx_lens.stride(0),
        seq_lens.stride(0),
        cu_seqlens_q.stride(0),
        cu_seqlens_k.stride(0),
        HEAD_DIM=HEAD_DIM,
        MEM_BLOCK_SIZE=MEM_BLOCK_SIZE,
        DIFFUSION_BLOCK_SIZE=DIFFUSION_BLOCK_SIZE
    )
    
    return k_output, v_output


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        model_type='causal_lm'
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.causal = model_type == 'causal_lm'
        self.model_type = model_type
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        } if is_rtx_xx90(torch.cuda.get_device_name(0)) else None
        self.flex_attention = torch.compile(
            partial(flex_attention, kernel_options=kernel_options, enable_gqa=True), dynamic=True)
        self._block_mask_cache = {}
        # CSV file path
        self.csv_path = "log/attention_profile.csv"

    @lru_cache(maxsize=32)
    def dllm_block_mask(self, block_mask: torch.Tensor, 
                        B: int, H: int, Q_LEN: int, KV_LEN: int, device: str):
        cache_key = (B, H, Q_LEN, KV_LEN, device)
        def _mask_mod(batch, head, token_q, token_kv):
            return block_mask[token_q, token_kv]
        if cache_key not in self._block_mask_cache:
            self._block_mask_cache[cache_key] = create_block_mask(
                _mask_mod, B, H, Q_LEN, KV_LEN, device=device
            )
        return self._block_mask_cache[cache_key]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: List[torch.Tensor] | None = None) -> torch.Tensor:
        timings = {
            "reshape_time": 0,
            "store_kvcache_time": 0,
            "attn_time": 0,
            "split_time": 0,
            "loadkv_time": 0,
            "prefill_transpose_time": 0,
            "loadkv_time_transpose": 0,
            "maskgen_time": 0,
            "final_reshape_time": 0,
            "overall_attention_time": 0
        }
        start_all = time.time()

        # Reshape
        start = time.time()
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        timings['reshape_time'] = time.time() - start

        context: ContextForDiffusionLM = get_context_causal_lm() if self.model_type == 'causal_lm' else get_context_diffusion_lm()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Fast Store KV cache
        if k_cache.numel() and v_cache.numel():
            store_start = time.time()
            if not (self.model_type == 'diffusion_lm' and context.slot_mapping.numel() == 0):
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, self.model_type)
            timings['store_kvcache_time'] = time.time() - store_start

        # Prefill / Decode logic
        if context.is_prefill:
            # Block PK
            if context.block_tables is not None and self.model_type == 'causal_lm':
                k, v = k_cache, v_cache

            # Attention computation
            attn_start = time.time()
            if self.model_type == 'causal_lm':
                o = flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, causal=self.causal, block_table=context.block_tables
                )
            else:
                transpose_start = time.time()
                transpose_fn = lambda x: rearrange(x, 's h d -> 1 h s d')
                q_t, k_t, v_t = [transpose_fn(t) for t in (q, k, v)]
                timings['prefill_transpose_time'] = time.time() - transpose_start

                mask_start = time.time()
                B, H, S, _ = q_t.shape
                block_mask = self.dllm_block_mask(context.block_mask, B, H, S, S, str(q.device))
                timings['maskgen_time'] = time.time() - mask_start

                o = self.flex_attention(q_t, k_t, v_t, block_mask=block_mask)
            timings['attn_time'] = time.time() - attn_start
        else:
            if self.model_type == 'causal_lm':
                decode_start = time.time()
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens, block_table=context.block_tables,
                    softmax_scale=self.scale, causal=self.causal
                )
                timings['decode_time'] = time.time() - decode_start
            else:
                split_start = time.time()
                transpose_fn = lambda x: rearrange(x, 's h d -> 1 h s d')
                q_t = transpose_fn(q)
                k_list, v_list = [torch.split(tensor, context.seq_lens, dim=0) for tensor in (k, v)]
                timings['split_time'] = time.time() - split_start

                loadkv_start = time.time()
                # Fast Load KV cache
                k_comb, v_comb = load_kvcache(self.k_cache, self.v_cache, context, k, v)
                loadkv_time = time.time() - loadkv_start
                timings['loadkv_time'] = loadkv_time

                transpose_load_start = time.time()
                k_t, v_t = transpose_fn(k_comb), transpose_fn(v_comb)
                timings['loadkv_time_transpose'] = time.time() - transpose_load_start

                mask_start = time.time()
                B, H, Sq, _ = q_t.shape
                _, _, Skv, _ = k_t.shape
                block_mask = self.dllm_block_mask(context.block_mask, B, H, Sq, Skv, str(q.device))
                timings['maskgen_time'] = time.time() - mask_start

                attn_start = time.time()
                o = self.flex_attention(q_t, k_t, v_t, block_mask=block_mask)
                timings['attn_time'] = time.time() - attn_start

        # Final reshape
        final_start = time.time()
        if self.model_type == 'causal_lm':
            o = o.view(-1, self.num_heads * self.head_dim)
        else:
            o = rearrange(o, '1 h s d -> s (h d)')
        timings['final_reshape_time'] = time.time() - final_start

        # Overall
        timings['overall_attention_time'] = time.time() - start_all

        # Write to CSV (ensure header even if file exists)
        write_header = True
        if os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 0:
            write_header = False
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(timings.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(timings)

        return o