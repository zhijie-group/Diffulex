# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: D2F
# type: ignore

# Organization: SJTU DENG Lab
# Author: Drew Jin (JIN. Yijie, @drewjin)
# Date: 2025-08-15
# Email: drewjin0827@gmail.com
# All rights reserved.

import tilus
import torch

import numpy as np

from hidet.ir import DataType
from tilus.utils import cdiv
from tilus import boolean, f32, int32, int64, void_p


tilus.option.cache_dir("./cache")


class TilusDecodeAttnForDifusionLM(tilus.Script):
    """
        Fusing kvcache loading, attention against kvcache, self-attention, 
        and self-attention custom mask applying all together
    """
    def __init__(self, dtype: DataType, num_heads: int, num_kv_heads: int, 
                 head_dim: int, num_warps: int, diffusion_block_size: int,
                 page_size: int = 256, x: int = 8):
        super().__init__()
        self.dtype: DataType = dtype
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.x = x
        self.head_dim_x = head_dim // x
        self.num_warps = num_warps
        self.block_q = diffusion_block_size * 2
        self.block_kv = diffusion_block_size * 2
        self.block_kvc = self.page_size = page_size
        self.score_scale = float(1.0 / np.sqrt(head_dim))
        self.group_size = num_heads // num_kv_heads
        
        # For attn against kvcache
        self.qkc_config = self.cuda.resolve_dot_config(
            dtype,
            f32,
            m=self.block_q,
            n=self.block_kv,
            k=self.head_dim,
            warp_m=self.num_warps,
            warp_n=1,
        )
        self.pvc_config = self.cuda.resolve_dot_config(
            dtype,
            f32,
            m=self.block_q,
            n=self.head_dim,
            k=self.block_kvc,
            warp_m=self.num_warps,
            warp_n=1,
        )
        
        # For self-attn
        self.qk_config = self.cuda.resolve_dot_config(
            dtype,
            f32,
            m=self.block_q,
            n=self.block_kv,
            k=self.head_dim,
            warp_m=self.num_warps,
            warp_n=1,
        )
        self.pv_config = self.cuda.resolve_dot_config(
            dtype,
            f32,
            m=self.block_q,
            n=self.head_dim,
            k=self.block_kv,
            warp_m=self.num_warps,
            warp_n=1,
        )
        assert self.qk_config.lc == self.pv_config.la


    def __call__(self, q_ptr: void_p, k_ptr: void_p, v_ptr: void_p, o_ptr: void_p,
                 k_cache_ptr: void_p, v_cache_ptr: void_p, page_table_ptr: void_p,
                 cu_seqlens_q_ptr: void_p, total_lens_ptr: void_p, ctxlens_ptr: void_p,
                 num_seqs: int, max_seqlen: int, q_len: int, kv_len: int, num_pages: int, max_seq_pages: int):
        # TODO
        # Setup Grid
        self.attrs.warps = self.num_warps
        self.attrs.blocks = (cdiv(max_seqlen, self.block_q), self.num_heads, num_seqs)
        
        # Get programs ids
        start_m = self.blockIdx.x
        head = self.blockIdx.y
        seq = self.blockIdx.z
        
        # build-up global_views
        global_q = self.global_view(q_ptr, dtype=self.dtype, shape=[q_len, self.num_heads, self.head_dim])
        global_k = self.global_view(k_ptr, dtype=self.dtype, shape=[kv_len, self.num_kv_heads, self.head_dim])
        global_v = self.global_view(v_ptr, dtype=self.dtype, shape=[kv_len, self.num_kv_heads, self.head_dim])
        global_o = self.global_view(o_ptr, dtype=self.dtype, shape=[q_len, self.num_heads, self.head_dim])
        global_k_cache = self.global_view(k_cache_ptr, dtype=self.dtype, shape=[num_pages, self.num_kv_heads, 
                                                                                self.head_dim_x, self.page_size, self.x])
        global_v_cache = self.global_view(v_cache_ptr, dtype=self.dtype, shape=[num_pages, self.num_kv_heads, 
                                                                                self.head_dim, self.page_size])
        global_page_table = self.global_view(page_table_ptr, dtype=int64, shape=[num_seqs, max_seq_pages])
        global_cu_seqlens_q = self.global_view(cu_seqlens_q_ptr, dtype=int32, shape=[num_seqs + 1])
        global_total_lens = self.global_view(total_lens_ptr, dtype=int32, shape=[num_seqs])
        global_ctxlens = self.global_view(ctxlens_ptr, dtype=int32, shape=[num_seqs])
        
        # Allocate registers for q_start_idx, total_len, ctxlen
        shared_q_start_idx = self.shared_tensor(dtype=int32, shape=[1])
        shared_total_len = self.shared_tensor(dtype=int32, shape=[1])
        shared_ctxlen = self.shared_tensor(dtype=int32, shape=[1])
        load_q_start_idx = self.load_global(global_cu_seqlens_q, offsets=[seq], shape=[1], dims=[0])
        load_total_len = self.load_global(global_total_lens, offsets=[seq], shape=[1], dims=[0])
        load_ctxlen = self.load_global(global_ctxlens, offsets=[seq], shape=[1], dims=[0])
        self.store_shared(shared_q_start_idx, load_q_start_idx)
        self.store_shared(shared_total_len, load_total_len)
        self.store_shared(shared_ctxlen, load_ctxlen)
        self.sync()
        q_start_idx = self.load_shared(shared_q_start_idx)
        total_len = self.load_shared(shared_total_len)
        ctxlen = self.load_shared(shared_ctxlen)
        self.sync()
        self.free_shared(shared_q_start_idx)
        self.free_shared(shared_total_len)
        self.free_shared(shared_ctxlen)

        # Load q tile into register
        off_q = start_m * self.block_q + q_start_idx
        shared_q = self.shared_tensor(dtype=self.dtype, shape=[self.block_q, self.head_dim])
        load_q = self.load_global(global_q, offsets=[off_q, head, 0], shape=[self.block_q, self.head_dim], dims=[0, 2])
        self.store_shared(shared_q, load_q)
        self.sync()
        q = self.load_shared(shared_q)
        self.sync()
        self.free_shared(shared_q)
        
        # Allocate shared memory for k, v, k_cache, and v_cache
        shared_k = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_dim])
        shared_v = self.shared_tensor(dtype=self.dtype, shape=[self.block_kv, self.head_dim])
        shared_k_cache = self.shared_tensor(dtype=self.dtype, shape=[self.page_size, self.head_dim])
        shared_v_cache = self.shared_tensor(dtype=self.dtype, shape=[self.page_size, self.head_dim])
        shared_page_table = self.shared_tensor(dtype=int64, shape=[1])

        # Init accumulators
        acc = self.register_tensor(dtype=f32, shape=[self.block_q, self.head_dim], init=0.0)
        m_i = self.register_tensor(dtype=f32, shape=[self.block_q, 1], init=-1e6) # rowmax(attn_score)
        l_i = self.register_tensor(dtype=f32, shape=[self.block_q, 1], init=0.0) # rowsum(exp(attn_score - m_i))

        # Pre-launch async copy for K Cache
        self.copy_async(global_k_cache, shared_k_cache, 
                        offsets=[seq_first_page, head // self.group_size, 0, 0, 0], dims=[2, 3, 4])
        self.copy_async_commit_group()
         