import os
import torch
import tilelang
import tilelang.language as T

from flash_attn import flash_attn_varlen_func
from tilelang.autotuner import set_autotune_inputs

from diffulex_kernel.python.auto_tuner import build_configs
from diffulex_kernel.python.kv_cache_kernels import load_kvcache
from diffulex.attention.metadata import AttnMetaDataBase, is_warming_up
from test.python.utils.checker import CHECK_FLASH_ATTN_PREFILL, CHECK_FLASH_ATTN_DECODE


# from tilelang.engine.callback import register_cuda_postproc_callback
# @register_cuda_postproc_callback
# def tilelang_callback_cuda_postproc(code, _):
#     code = "// tilelang_callback_cuda_postproc: generated CUDA code by TileLang\n" + code
#     print(code)
#     return code


kernel_config = None
kernel_config_bf16_q_fp8_kv_decode = None


@tilelang.autotune(configs=build_configs())
@tilelang.jit(
    # NOTE: Disable TMA and warp specialized for now to avoid compile error on Hopper
    out_idx=[-1], 
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def dllm_flash_attn_prefill_kernel(
    NUM_SEQS: int,
    NUM_GROUPS: int,
    Q_LEN: int,
    KV_LEN: int,
    NUM_HEADS: int,
    HEAD_DIM: int,
    IS_BLOCK_ATTN: bool,
    DIFFUSION_BLOCK_SIZE: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    NUM_STAGES: int = 1,
    NUM_THREADS: int = 128,
):
    SCALE = (1.0 / HEAD_DIM)**0.5 * 1.44269504  # log2(e)
    NUM_KV_HEADS = NUM_HEADS // NUM_GROUPS
    Q_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    KV_SHAPE = [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    O_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    DTYPE = "bfloat16"
    ACCUM_DTYPE = "float"
    
    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_SHAPE, DTYPE),
        K: T.Tensor(KV_SHAPE, DTYPE),
        V: T.Tensor(KV_SHAPE, DTYPE),
        cu_seqlens_q: T.Tensor(NUM_SEQS + 1, "int32"),
        cu_seqlens_k: T.Tensor(NUM_SEQS + 1, "int32"),
        max_seqlen_q: T.int32,
        O: T.Tensor(O_SHAPE, DTYPE),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, BLOCK_M), NUM_HEADS, NUM_SEQS, threads=NUM_THREADS) as (bx, by, bz):
            Q_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)
            K_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            V_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            O_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)
            
            acc_score = T.alloc_fragment([BLOCK_M, BLOCK_N], ACCUM_DTYPE)
            acc_score_cast = T.alloc_fragment([BLOCK_M, BLOCK_N], DTYPE)
            acc_output = T.alloc_fragment([BLOCK_M, HEAD_DIM], ACCUM_DTYPE)
            scores_max = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_max_prev = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_scale = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            log_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            })
            
            q_block_idx = bx
            seq_idx = bz
            head_idx = by
            kv_head_idx = head_idx // NUM_GROUPS
            
            q_start_idx = cu_seqlens_q[seq_idx]
            kv_start_idx = cu_seqlens_k[seq_idx]
            q_end_idx = cu_seqlens_q[seq_idx + 1]
            kv_end_idx = cu_seqlens_k[seq_idx + 1]
            
            cur_q_seqlen = q_end_idx - q_start_idx
            cur_kv_seqlen = kv_end_idx - kv_start_idx
            
            T.copy(Q[q_start_idx + q_block_idx * BLOCK_M : q_start_idx + (q_block_idx + 1) * BLOCK_M, head_idx, :], Q_shared)
            
            T.fill(acc_output, 0)
            T.fill(acc_score, 0)
            T.fill(log_sum, 0)
            T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
            
            # The same boundary condition as naive causal mask
            loop_range = (
                T.min(T.ceildiv(cur_q_seqlen + (q_block_idx + 1) * BLOCK_M, BLOCK_N), T.ceildiv(cur_kv_seqlen, BLOCK_N))
                if IS_BLOCK_ATTN else T.ceildiv(cur_kv_seqlen, BLOCK_N)
            )
            for kv_block_idx in T.Pipelined(loop_range, num_stages=NUM_STAGES):
                T.copy(K[kv_start_idx + kv_block_idx * BLOCK_N : kv_start_idx + (kv_block_idx + 1) * BLOCK_N, kv_head_idx, :], K_shared)
                
                # Initialize acc_score with mask
                if IS_BLOCK_ATTN:
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        num_diffusion_blocks = (q_block_idx * BLOCK_M + i) // DIFFUSION_BLOCK_SIZE + 1
                        acc_score[i, j] = T.if_then_else(
                            (num_diffusion_blocks * DIFFUSION_BLOCK_SIZE <= kv_block_idx * BLOCK_N + j) or
                            (q_block_idx * BLOCK_M + i >= cur_q_seqlen or 
                            kv_block_idx * BLOCK_N + j >= cur_kv_seqlen), -1e9, 0
                        )
                else:
                    for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                        acc_score[i, j] = T.if_then_else(
                            (q_block_idx * BLOCK_M + i >= cur_q_seqlen or 
                            kv_block_idx * BLOCK_N + j >= cur_kv_seqlen), -1e9, 0
                        )
                        
                # Compute attention scores
                T.gemm(Q_shared, K_shared, acc_score, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                # Compute online softmax
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
                T.reduce_max(acc_score, scores_max, dim=1, clear=False) # T.reduce_max(acc_score, scores_max, dim=1, clear=True) # TODO: check if this is correct
                for i in T.Parallel(BLOCK_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                
                for i in T.parallel(BLOCK_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * SCALE - scores_max[i] * SCALE)
                    
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    acc_score[i, j] = T.exp2(acc_score[i, j] * SCALE - scores_max[i] * SCALE)
                    
                T.reduce_sum(acc_score, scores_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    log_sum[i] = log_sum[i] * scores_scale[i] + scores_sum[i]
                
                T.copy(acc_score, acc_score_cast)
                for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                    acc_output[i, j] *= scores_scale[i]
                
                # Compute attention output
                T.copy(V[kv_start_idx + kv_block_idx * BLOCK_N : kv_start_idx + (kv_block_idx + 1) * BLOCK_N, kv_head_idx, :], V_shared)
                T.gemm(acc_score_cast, V_shared, acc_output, policy=T.GemmWarpPolicy.FullRow)
            
            for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                acc_output[i, j] /= log_sum[i]
                
            T.copy(acc_output, O_shared)
            for i, d_idx in T.Parallel(BLOCK_M, HEAD_DIM):
                if i + q_block_idx * BLOCK_M < cur_q_seqlen:
                    O[i + q_start_idx + q_block_idx * BLOCK_M, head_idx, d_idx] = O_shared[i, d_idx]
            
    return kernel


@tilelang.jit(
    # NOTE: Disable TMA and warp specialized for now to avoid compile error on Hopper
    out_idx=[-1], 
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        # tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,pdf"
    }
)
def dllm_flash_attn_decode_kernel(
    NUM_SEQS: int,
    NUM_GROUPS: int,
    NUM_PAGE_BLOCKS: int,
    Q_LEN: int,
    KV_LEN: int,
    NUM_HEADS: int,
    HEAD_DIM: int,
    IS_BLOCK_ATTN: bool,
    DIFFUSION_BLOCK_SIZE: int,
    MAX_SEQ_NUM_BLOCKS: int,
    PAGE_BLOCK_SIZE: int = 32,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    NUM_STAGES: int = 1,
    NUM_THREADS: int = 128,
):
    SCALE = (1.0 / HEAD_DIM)**0.5 * 1.44269504  # log2(e)
    NUM_KV_HEADS = NUM_HEADS // NUM_GROUPS
    Q_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    KV_SHAPE = [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    O_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    K_CACHE_SHAPE = [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    V_CACHE_SHAPE = [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    MAX_SEQ_NUM_BLOCKS = T.dynamic("MAX_SEQ_NUM_BLOCKS", 'int32')
    BLOCK_TABLES_SHAPE = [NUM_SEQS, MAX_SEQ_NUM_BLOCKS]
    DTYPE = "bfloat16"
    ACCUM_DTYPE = "float32"
   
    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_SHAPE, DTYPE),
        K: T.Tensor(KV_SHAPE, DTYPE),
        V: T.Tensor(KV_SHAPE, DTYPE),
        K_Cache: T.Tensor(K_CACHE_SHAPE, DTYPE),
        V_Cache: T.Tensor(V_CACHE_SHAPE, DTYPE),
        block_tables: T.Tensor(BLOCK_TABLES_SHAPE, "int32"),
        context_lens: T.Tensor(NUM_SEQS, "int32"),
        cu_seqlens_q: T.Tensor(NUM_SEQS + 1, "int32"),
        cu_seqlens_k: T.Tensor(NUM_SEQS + 1, "int32"),
        max_seqlen_q: T.int32, 
        O: T.Tensor(O_SHAPE, DTYPE),
    ):
        with T.Kernel(NUM_SEQS, NUM_HEADS, threads=NUM_THREADS) as (bx, by):
            Q_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)
            K_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            V_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            O_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)
            K_Cache_shared = T.alloc_shared([PAGE_BLOCK_SIZE, HEAD_DIM], DTYPE)
            V_Cache_shared = T.alloc_shared([PAGE_BLOCK_SIZE, HEAD_DIM], DTYPE)
            
            acc_score_kv = T.alloc_fragment([BLOCK_M, BLOCK_N], ACCUM_DTYPE)
            acc_score_kv_cast = T.alloc_fragment([BLOCK_M, BLOCK_N], DTYPE)
            acc_score_kvcache = T.alloc_fragment([BLOCK_M, PAGE_BLOCK_SIZE], ACCUM_DTYPE)
            acc_score_kvcache_cast = T.alloc_fragment([BLOCK_M, PAGE_BLOCK_SIZE], DTYPE)
            
            acc_output = T.alloc_fragment([BLOCK_M, HEAD_DIM], ACCUM_DTYPE)
            scores_max = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_max_prev = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_scale = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            log_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            })

            seq_idx = bx
            head_idx = by
            kv_head_idx = head_idx // NUM_GROUPS
            
            q_start_idx = cu_seqlens_q[seq_idx]
            kv_start_idx = cu_seqlens_k[seq_idx]
            q_end_idx = cu_seqlens_q[seq_idx + 1]
            kv_end_idx = cu_seqlens_k[seq_idx + 1]
            
            cur_q_seqlen = q_end_idx - q_start_idx
            cur_kv_seqlen = kv_end_idx - kv_start_idx
            
            cur_context_len = context_lens[seq_idx]
            
            T.copy(Q[q_start_idx : q_start_idx + BLOCK_M, head_idx, :], Q_shared)
            
            T.fill(acc_output, 0)
            T.fill(acc_score_kv, 0)
            T.fill(acc_score_kvcache, 0)
            T.fill(log_sum, 0)
            T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
            
            # ==========================
            # Stage 1: KV Cache Attention (Context)
            # ==========================
            for page_block_idx_local in T.Pipelined(MAX_SEQ_NUM_BLOCKS, num_stages=NUM_STAGES):
                page_block_idx_global = block_tables[seq_idx, page_block_idx_local]
                
                if page_block_idx_global >= 0:
                    T.copy(K_Cache[page_block_idx_global, :, kv_head_idx, :], K_Cache_shared)
                    
                    for i, j in T.Parallel(BLOCK_M, PAGE_BLOCK_SIZE):
                        acc_score_kvcache[i, j] = T.if_then_else(
                            (i >= cur_q_seqlen or 
                            page_block_idx_local * PAGE_BLOCK_SIZE + j >= cur_context_len), -1e9, 0
                        )
                    
                    # Compute attention scores
                    T.gemm(Q_shared, K_Cache_shared, acc_score_kvcache, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    
                    # Compute online softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
                    T.reduce_max(acc_score_kvcache, scores_max, dim=1, clear=False)
                    for i in T.Parallel(BLOCK_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    
                    for i in T.Parallel(BLOCK_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * SCALE - scores_max[i] * SCALE)
                        
                    for i, j in T.Parallel(BLOCK_M, PAGE_BLOCK_SIZE):
                        acc_score_kvcache[i, j] = T.exp2(acc_score_kvcache[i, j] * SCALE - scores_max[i] * SCALE)
                        
                    T.reduce_sum(acc_score_kvcache, scores_sum, dim=1)
                    for i in T.Parallel(BLOCK_M):
                        log_sum[i] = log_sum[i] * scores_scale[i] + scores_sum[i]
                        
                    T.copy(acc_score_kvcache, acc_score_kvcache_cast)
                    
                    # Scale previous output accumulator
                    for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                        acc_output[i, j] *= scores_scale[i]
                    
                    # Accumulate current V_cache contribution
                    T.copy(V_Cache[page_block_idx_global, :, kv_head_idx, :], V_Cache_shared)
                    T.gemm(acc_score_kvcache_cast, V_Cache_shared, acc_output, policy=T.GemmWarpPolicy.FullRow)
                
            
            # ==========================
            # Stage 2: Fresh KV Attention (Self-Attn)
            # ==========================
            for idx in T.Pipelined(T.ceildiv(DIFFUSION_BLOCK_SIZE, BLOCK_N), num_stages=NUM_STAGES):
                T.copy(K[kv_start_idx : kv_start_idx + BLOCK_N, kv_head_idx, :], K_shared)

                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    acc_score_kv[i, j] = T.if_then_else(i >= cur_q_seqlen or j >= cur_kv_seqlen, -1e9, 0)
                
                T.gemm(Q_shared, K_shared, acc_score_kv, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
                T.reduce_max(acc_score_kv, scores_max, dim=1, clear=False)
                for i in T.Parallel(BLOCK_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                
                for i in T.Parallel(BLOCK_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * SCALE - scores_max[i] * SCALE)
                
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    acc_score_kv[i, j] = T.exp2(acc_score_kv[i, j] * SCALE - scores_max[i] * SCALE)
                    
                T.reduce_sum(acc_score_kv, scores_sum, dim=1)
                for i in T.Parallel(BLOCK_M):
                    log_sum[i] = log_sum[i] * scores_scale[i] + scores_sum[i]
                    
                T.copy(acc_score_kv, acc_score_kv_cast)
                
                # Scale previous output
                for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                    acc_output[i, j] *= scores_scale[i]
                
                T.copy(V[kv_start_idx : kv_start_idx + BLOCK_N, kv_head_idx, :], V_shared)
                
                # Accumulate current V contribution
                T.gemm(acc_score_kv_cast, V_shared, acc_output, policy=T.GemmWarpPolicy.FullRow)
            
            
            # ==========================
            # Stage 3: Finalize
            # ==========================
            for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                acc_output[i, j] /= log_sum[i]

            T.copy(acc_output, O_shared)
            for i, d_idx in T.Parallel(BLOCK_M, HEAD_DIM):
                if i < cur_q_seqlen:
                    O[i + q_start_idx, head_idx, d_idx] = O_shared[i, d_idx] 
            
    return kernel


@tilelang.autotune(configs=build_configs())
@tilelang.jit(
    # NOTE: Disable TMA and warp specialized for now to avoid compile error on Hopper
    out_idx=[-1], 
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def dllm_flash_attn_decode_kernel_bf16_q_fp8_kv(
    NUM_SEQS: int,
    NUM_GROUPS: int,
    NUM_PAGE_BLOCKS: int,
    Q_LEN: int,
    KV_LEN: int,
    NUM_HEADS: int,
    HEAD_DIM: int,
    IS_BLOCK_ATTN: bool,
    DIFFUSION_BLOCK_SIZE: int,
    MAX_SEQ_NUM_BLOCKS: int,
    PAGE_BLOCK_SIZE: int = 32,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    NUM_STAGES: int = 1,
    NUM_THREADS: int = 128,
):
    SCALE = (1.0 / HEAD_DIM)**0.5 * 1.44269504  # log2(e)
    NUM_KV_HEADS = NUM_HEADS // NUM_GROUPS
    Q_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    KV_SHAPE = [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
    O_SHAPE = [Q_LEN, NUM_HEADS, HEAD_DIM]
    K_CACHE_SHAPE = [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    V_CACHE_SHAPE = [NUM_PAGE_BLOCKS, PAGE_BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM]
    MAX_SEQ_NUM_BLOCKS = T.dynamic("MAX_SEQ_NUM_BLOCKS", 'int32')
    BLOCK_TABLES_SHAPE = [NUM_SEQS, MAX_SEQ_NUM_BLOCKS]
    DTYPE = "bfloat16"
    ACCUM_DTYPE = "float32"
    FP8_DTYPE = "float8_e4m3fn"
   
    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_SHAPE, DTYPE),
        K: T.Tensor(KV_SHAPE, DTYPE),
        V: T.Tensor(KV_SHAPE, DTYPE),
        K_Cache: T.Tensor(K_CACHE_SHAPE, FP8_DTYPE),
        V_Cache: T.Tensor(V_CACHE_SHAPE, FP8_DTYPE),
        K_Scale: T.Tensor([NUM_KV_HEADS], "float32"),
        V_Scale: T.Tensor([NUM_KV_HEADS], "float32"),
        block_tables: T.Tensor(BLOCK_TABLES_SHAPE, "int32"),
        context_lens: T.Tensor(NUM_SEQS, "int32"),
        cu_seqlens_q: T.Tensor(NUM_SEQS + 1, "int32"),
        cu_seqlens_k: T.Tensor(NUM_SEQS + 1, "int32"),
        max_seqlen_q: T.int32, 
        O: T.Tensor(O_SHAPE, DTYPE),
    ):
        with T.Kernel(NUM_SEQS, NUM_HEADS, threads=NUM_THREADS) as (bx, by):
            Q_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)
            K_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            V_shared = T.alloc_shared([BLOCK_N, HEAD_DIM], DTYPE)
            O_shared = T.alloc_shared([BLOCK_M, HEAD_DIM], DTYPE)
            
            # KV cache shared staging buffers (BF16):
            # HBM(FP8) -> T.copy (implicit cast) -> shared(BF16) -> GEMM
            K_Cache_shared_bf16 = T.alloc_shared([PAGE_BLOCK_SIZE, HEAD_DIM], DTYPE)
            V_Cache_shared_bf16 = T.alloc_shared([PAGE_BLOCK_SIZE, HEAD_DIM], DTYPE)
            
            acc_score_kv = T.alloc_fragment([BLOCK_M, BLOCK_N], ACCUM_DTYPE)
            acc_score_kv_cast = T.alloc_fragment([BLOCK_M, BLOCK_N], DTYPE)
            acc_score_kvcache = T.alloc_fragment([BLOCK_M, PAGE_BLOCK_SIZE], ACCUM_DTYPE)
            acc_score_kvcache_cast = T.alloc_fragment([BLOCK_M, PAGE_BLOCK_SIZE], DTYPE)
            
            acc_output = T.alloc_fragment([BLOCK_M, HEAD_DIM], ACCUM_DTYPE)
            scores_max = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_max_prev = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_scale = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            scores_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            log_sum = T.alloc_fragment([BLOCK_M], ACCUM_DTYPE)
            
            T.annotate_layout({
                Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                O_shared: tilelang.layout.make_swizzled_layout(O_shared),
            })

            seq_idx = bx
            head_idx = by
            kv_head_idx = head_idx // NUM_GROUPS
            
            q_start_idx = cu_seqlens_q[seq_idx]
            kv_start_idx = cu_seqlens_k[seq_idx]
            q_end_idx = cu_seqlens_q[seq_idx + 1]
            kv_end_idx = cu_seqlens_k[seq_idx + 1]
            
            cur_q_seqlen = q_end_idx - q_start_idx
            cur_kv_seqlen = kv_end_idx - kv_start_idx
            
            cur_context_len = context_lens[seq_idx]
            
            T.copy(Q[q_start_idx : q_start_idx + BLOCK_M, head_idx, :], Q_shared)
            
            T.fill(acc_output, 0)
            T.fill(acc_score_kv, 0)
            T.fill(acc_score_kvcache, 0)
            T.fill(log_sum, 0)
            T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
            
            # ==========================
            # Stage 1: KV Cache Attention (Context)
            # ==========================
            for page_block_idx_local in T.Pipelined(MAX_SEQ_NUM_BLOCKS, num_stages=NUM_STAGES):
                page_block_idx_global = block_tables[seq_idx, page_block_idx_local]
                if page_block_idx_global >= 0:
                    # Step 1: Load FP8 K_Cache, implicit cast to BF16 (vectorized path).
                    # K_Scale will be applied on scores (much cheaper than scaling K elementwise).
                    T.copy(K_Cache[page_block_idx_global, :, kv_head_idx, :], K_Cache_shared_bf16)
                    
                    # Initialize scores with mask, then GEMM accumulates into it (masked entries remain ~-1e9).
                    for i, j in T.Parallel(BLOCK_M, PAGE_BLOCK_SIZE):
                        acc_score_kvcache[i, j] = T.if_then_else(
                            (i >= cur_q_seqlen or page_block_idx_local * PAGE_BLOCK_SIZE + j >= cur_context_len),
                            -1e9,
                            0,
                        )
                    
                    # Compute attention scores
                    T.gemm(Q_shared, K_Cache_shared_bf16, acc_score_kvcache, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    # Apply per-head K scale on scores: (Q·(K*ks)) == (Q·K) * ks
                    for i, j in T.Parallel(BLOCK_M, PAGE_BLOCK_SIZE):
                        acc_score_kvcache[i, j] *= K_Scale[kv_head_idx]
                    
                    # Compute online softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
                    T.reduce_max(acc_score_kvcache, scores_max, dim=1, clear=False)
                    for i in T.Parallel(BLOCK_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    
                    for i in T.Parallel(BLOCK_M):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * SCALE - scores_max[i] * SCALE)
                        
                    for i, j in T.Parallel(BLOCK_M, PAGE_BLOCK_SIZE):
                        acc_score_kvcache[i, j] = T.exp2(acc_score_kvcache[i, j] * SCALE - scores_max[i] * SCALE)
                        
                    T.reduce_sum(acc_score_kvcache, scores_sum, dim=1)
                    for i in T.Parallel(BLOCK_M):
                        log_sum[i] = log_sum[i] * scores_scale[i] + scores_sum[i]
                        
                    # Cast weights to BF16 for V GEMM, fuse per-head V scale here:
                    # (softmax * (V*vs)) == ((softmax*vs) · V)
                    # Use separate loop to avoid layout infer conflict
                    for i, j in T.Parallel(BLOCK_M, PAGE_BLOCK_SIZE):
                        acc_score_kvcache_cast[i, j] = (acc_score_kvcache[i, j] * V_Scale[kv_head_idx]).astype(T.bfloat16)
                    
                    # Scale previous output accumulator
                    for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                        acc_output[i, j] *= scores_scale[i]
                    
                    # Step 2: Load FP8 V_Cache, implicit cast to BF16 (vectorized path).
                    T.copy(V_Cache[page_block_idx_global, :, kv_head_idx, :], V_Cache_shared_bf16)
                    
                    # Accumulate current V_cache contribution using BF16 V_Cache shared buffer
                    T.gemm(acc_score_kvcache_cast, V_Cache_shared_bf16, acc_output, policy=T.GemmWarpPolicy.FullRow)
                
                if page_block_idx_local == MAX_SEQ_NUM_BLOCKS - 1:
                    # ==========================
                    # Stage 2: Fresh KV Attention (Self-Attn)
                    # ==========================
                    for idx in T.Pipelined(T.ceildiv(DIFFUSION_BLOCK_SIZE, BLOCK_N), num_stages=NUM_STAGES):
                        T.copy(K[kv_start_idx : kv_start_idx + BLOCK_N, kv_head_idx, :], K_shared)

                        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                            acc_score_kv[i, j] = T.if_then_else(i >= cur_q_seqlen or j >= cur_kv_seqlen, -1e9, 0)
                        
                        T.gemm(Q_shared, K_shared, acc_score_kv, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                        
                        T.copy(scores_max, scores_max_prev)
                        T.fill(scores_max, -T.infinity(ACCUM_DTYPE))
                        T.reduce_max(acc_score_kv, scores_max, dim=1, clear=False)
                        for i in T.Parallel(BLOCK_M):
                            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                        
                        for i in T.Parallel(BLOCK_M):
                            scores_scale[i] = T.exp2(scores_max_prev[i] * SCALE - scores_max[i] * SCALE)
                        
                        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                            acc_score_kv[i, j] = T.exp2(acc_score_kv[i, j] * SCALE - scores_max[i] * SCALE)
                            
                        T.reduce_sum(acc_score_kv, scores_sum, dim=1)
                        for i in T.Parallel(BLOCK_M):
                            log_sum[i] = log_sum[i] * scores_scale[i] + scores_sum[i]
                            
                        T.copy(acc_score_kv, acc_score_kv_cast)
                        
                        # Scale previous output
                        for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                            acc_output[i, j] *= scores_scale[i]
                        
                        T.copy(V[kv_start_idx : kv_start_idx + BLOCK_N, kv_head_idx, :], V_shared)
                        
                        # Accumulate current V contribution
                        T.gemm(acc_score_kv_cast, V_shared, acc_output, policy=T.GemmWarpPolicy.FullRow)
            
            # ==========================
            # Stage 3: Finalize
            # ==========================
            for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                acc_output[i, j] /= log_sum[i]

            T.copy(acc_output, O_shared)
            for i, d_idx in T.Parallel(BLOCK_M, HEAD_DIM):
                if i < cur_q_seqlen:
                    O[i + q_start_idx, head_idx, d_idx] = O_shared[i, d_idx] 
            
    return kernel


def _dllm_flash_attn_prefill_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase
) -> torch.Tensor:
    if attn_metadata.attn_type == "full_attention":
        return flash_attn_varlen_func(
            q, k, v, 
            attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
            attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
            softmax_scale=scale, block_table=None
        )
    elif attn_metadata.attn_type == "block_attention":
        if is_warming_up():
            global kernel_config
            with set_autotune_inputs([
                q, k, v,
                attn_metadata.cu_seqlens_q,
                attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q,
            ]):
                prefill_kernel = dllm_flash_attn_prefill_kernel(
                    attn_metadata.num_seqs,
                    q.shape[1] // k.shape[1],
                    q.shape[0],
                    k.shape[0],
                    q.shape[1],
                    q.shape[2],
                    attn_metadata.attn_type == "block_attention",
                    attn_metadata.diffusion_block_size
                )
            kernel_config = prefill_kernel.config
            # CHECK_FLASH_ATTN_PREFILL(
            #     q, k, v, 
            #     attn_metadata.cu_seqlens_q, 
            #     attn_metadata.cu_seqlens_k, 
            #     attn_metadata.max_seqlen_q, 
            #     prefill_kernel,
            #     diffusion_block_size=attn_metadata.diffusion_block_size,
            #     is_block_attn=(attn_metadata.attn_type == "block_attention"),
            # )
            return prefill_kernel(
                q, k, v, 
                attn_metadata.cu_seqlens_q, 
                attn_metadata.cu_seqlens_k, 
                attn_metadata.max_seqlen_q, 
            )
        else:
            prefill_kernel = dllm_flash_attn_prefill_kernel(
                attn_metadata.num_seqs,
                q.shape[1] // k.shape[1],
                q.shape[0],
                k.shape[0],
                q.shape[1],
                q.shape[2],
                attn_metadata.attn_type == "block_attention",
                attn_metadata.diffusion_block_size,
                **kernel_config
            )
            return prefill_kernel(
                q, k, v, 
                attn_metadata.cu_seqlens_q, 
                attn_metadata.cu_seqlens_k, 
                attn_metadata.max_seqlen_q, 
            )
            

def _dllm_flash_attn_decode_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase
) -> torch.Tensor:
    if attn_metadata.decode_mode == "static":
        # Use kernel_config from prefill if available, otherwise use empty dict
        config_kwargs = kernel_config if kernel_config is not None else {}
        decode_kernel = dllm_flash_attn_decode_kernel(
            attn_metadata.num_seqs,
            q.shape[1] // k.shape[1],
            k_cache.shape[0],
            q.shape[0],
            k.shape[0],
            q.shape[1],
            q.shape[2],
            attn_metadata.attn_type == "block_attention",
            attn_metadata.diffusion_block_size,
            attn_metadata.block_tables.shape[1],
            attn_metadata.page_block_size,
            **config_kwargs
        )
        if not is_warming_up():
            CHECK_FLASH_ATTN_DECODE(
                q, k, v,
                k_cache, v_cache,
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.cu_seqlens_q,
                attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q,
                decode_kernel,
                scale=scale,
                num_groups=q.shape[1] // k.shape[1],
                page_block_size=attn_metadata.page_block_size,
                diffusion_block_size=attn_metadata.diffusion_block_size,
                is_block_attn=(attn_metadata.attn_type == "block_attention"),
            )
        
        return decode_kernel(
            q, k, v, k_cache, v_cache,
            attn_metadata.block_tables,
            attn_metadata.context_lens,
            attn_metadata.cu_seqlens_q,
            attn_metadata.cu_seqlens_k,
            attn_metadata.max_seqlen_q,
        )
    elif attn_metadata.decode_mode == "varlen":
        do_profile = os.getenv("DIFFULEX_PROFILE_KVCACHE", "0") == "1"
        if do_profile and q.is_cuda:
            e0, e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
            e1.record()
            out = flash_attn_varlen_func(
                q, k_comb, v_comb,
                attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
                softmax_scale=scale, block_table=None
            )
            e2.record()
            e2.synchronize()
            print(
                f"[DIFFULEX_PROFILE_KVCACHE] decode(varlen,bf16kv) "
                f"load_kvcache={e0.elapsed_time(e1):.3f}ms flash_attn={e1.elapsed_time(e2):.3f}ms"
            )
            return out
        else:
            k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
            return flash_attn_varlen_func(
                q, k_comb, v_comb,
                attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
                softmax_scale=scale, block_table=None
            )


def _dllm_flash_attn_decode_bf16_q_fp8_kv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase
) -> torch.Tensor:
    """BF16 Q + FP8 KV decode helper function that uses BF16-Q/FP8-KV kernel with internal dequantization."""
    if attn_metadata.k_scale is None or attn_metadata.v_scale is None:
        raise ValueError("FP8 KV decode requires k_scale and v_scale in metadata")

    # KV cache is stored as uint8 for FP8, but TileLang expects float8 view dtype.
    from diffulex.utils.quantization.context import get_kv_cache_strategy
    strategy = get_kv_cache_strategy()
    if strategy is None or getattr(strategy, "kv_cache_format", "bf16") != "fp8":
        raise ValueError(f"Expected kv_cache_format='fp8', got strategy={type(strategy)}")
    k_cache = strategy.view_kv_cache_for_kernels(k_cache)
    v_cache = strategy.view_kv_cache_for_kernels(v_cache)
    
    if attn_metadata.decode_mode == "static":
        global kernel_config_bf16_q_fp8_kv_decode
        common_args = (
            attn_metadata.num_seqs,
            q.shape[1] // k.shape[1],
            k_cache.shape[0],
            q.shape[0],
            k.shape[0],
            q.shape[1],
            q.shape[2],
            attn_metadata.attn_type == "block_attention",
            attn_metadata.diffusion_block_size,
            attn_metadata.block_tables.shape[1],
            attn_metadata.page_block_size,
        )

        # BF16-Q/FP8-KV decode needs its own autotuned config; do not reuse prefill/BF16 config.
        # In some environments, TileLang autotuning may fail (e.g. no valid configs compile/validate).
        # In that case, fall back to the varlen path (Python dequant + flash-attn varlen) for correctness.
        try:
            if is_warming_up() or kernel_config_bf16_q_fp8_kv_decode is None:
                with set_autotune_inputs([
                    q, k, v,
                    k_cache, v_cache,
                    attn_metadata.k_scale,
                    attn_metadata.v_scale,
                    attn_metadata.block_tables,
                    attn_metadata.context_lens,
                    attn_metadata.cu_seqlens_q,
                    attn_metadata.cu_seqlens_k,
                    attn_metadata.max_seqlen_q,
                ]):
                    decode_kernel = dllm_flash_attn_decode_kernel_bf16_q_fp8_kv(*common_args)
                kernel_config_bf16_q_fp8_kv_decode = decode_kernel.config
            else:
                decode_kernel = dllm_flash_attn_decode_kernel_bf16_q_fp8_kv(
                    *common_args,
                    **kernel_config_bf16_q_fp8_kv_decode,
                )

            return decode_kernel(
                q, k, v, k_cache, v_cache,
                attn_metadata.k_scale,  # Pass K scale
                attn_metadata.v_scale,  # Pass V scale
                attn_metadata.block_tables,
                attn_metadata.context_lens,
                attn_metadata.cu_seqlens_q,
                attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q,
            )
        except RuntimeError as e:
            # Fall back if autotuning or runtime validation fails.
            if "Auto-tuning failed" in str(e) or "No configuration" in str(e):
                k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
                return flash_attn_varlen_func(
                    q, k_comb, v_comb,
                    attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
                    attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
                    softmax_scale=scale, block_table=None
                )
            raise
    elif attn_metadata.decode_mode == "varlen":
        # varlen模式使用load_kvcache：FP8 反量化/scale 融合应在 load_kvcache 内完成（Triton fused kernel）
        do_profile = os.getenv("DIFFULEX_PROFILE_KVCACHE", "0") == "1"
        if do_profile and q.is_cuda:
            e0, e1, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
            e1.record()
            out = flash_attn_varlen_func(
                q, k_comb, v_comb,
                attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
                softmax_scale=scale, block_table=None
            )
            e2.record()
            e2.synchronize()
            print(
                f"[DIFFULEX_PROFILE_KVCACHE] decode(varlen,fp8kv) "
                f"load_kvcache={e0.elapsed_time(e1):.3f}ms flash_attn={e1.elapsed_time(e2):.3f}ms"
            )
            return out
        else:
            k_comb, v_comb = load_kvcache(k_cache, v_cache, attn_metadata, k, v)
            return flash_attn_varlen_func(
                q, k_comb, v_comb,
                attn_metadata.cu_seqlens_q, attn_metadata.cu_seqlens_k,
                attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k,
                softmax_scale=scale, block_table=None
            )
    else:
        raise ValueError(f"Unsupported decode mode: {attn_metadata.decode_mode}")


def dllm_flash_attn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase
) -> torch.Tensor:
    """
    Prefill attention wrapper that dynamically selects kernel based on quantization strategy.
    
    Args:
        q: Query tensor [Q_LEN, NUM_HEADS, HEAD_DIM]
        k: Key tensor [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        v: Value tensor [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        scale: Attention scale factor
        attn_metadata: Attention metadata
    
    Returns:
        Output tensor [Q_LEN, NUM_HEADS, HEAD_DIM]
    """
    from diffulex.utils.quantization.context import get_kv_cache_strategy
    kv_strategy = get_kv_cache_strategy()
    kv_fmt = getattr(kv_strategy, "kv_cache_format", "bf16") if kv_strategy is not None else "bf16"

    # Q always uses BF16 (attn_q quantization is not supported)
    q_fmt = "bf16"

    # Prefill currently uses BF16 kernels for all formats (FP8 prefill kernel TBD).
    if q_fmt == "bf16" and kv_fmt in ("bf16", "fp8"):
        return _dllm_flash_attn_prefill_bf16(q, k, v, scale, attn_metadata)
    raise ValueError(
        f"Unsupported q_format={q_fmt!r} / kv_cache_format={kv_fmt!r} for prefill"
    )


def dllm_flash_attn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    scale: float,
    attn_metadata: AttnMetaDataBase
) -> torch.Tensor:
    """
    Decode attention wrapper that dynamically selects kernel based on quantization strategy.
    
    Args:
        q: Query tensor [Q_LEN, NUM_HEADS, HEAD_DIM]
        k: Key tensor [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        v: Value tensor [KV_LEN, NUM_KV_HEADS, HEAD_DIM]
        k_cache: Key cache tensor (shape depends on layout)
        v_cache: Value cache tensor (shape depends on layout)
        scale: Attention scale factor
        attn_metadata: Attention metadata
    
    Returns:
        Output tensor [Q_LEN, NUM_HEADS, HEAD_DIM]
    
    Note:
        For FP8 strategy:
        - Unified layout static mode: dequantization + scale fusion are handled inside the TileLang FP8 decode kernel
        - Unified layout varlen mode: dequantization is handled by load_kvcache (Python path)
        - Distinct layout: dequantization is handled by load_kvcache (Python path)
    """
    from diffulex.utils.quantization.context import get_kv_cache_strategy
    kv_strategy = get_kv_cache_strategy()
    kv_fmt = getattr(kv_strategy, "kv_cache_format", "bf16") if kv_strategy is not None else "bf16"

    # Q always uses BF16 (attn_q quantization is not supported)
    q_fmt = "bf16"

    if q_fmt == "bf16" and kv_fmt == "bf16":
        return _dllm_flash_attn_decode_bf16(q, k, v, k_cache, v_cache, scale, attn_metadata)
    if q_fmt == "bf16" and kv_fmt == "fp8":
        return _dllm_flash_attn_decode_bf16_q_fp8_kv(q, k, v, k_cache, v_cache, scale, attn_metadata)
    raise ValueError(
        f"Unsupported q_format={q_fmt!r} / kv_cache_format={kv_fmt!r} for decode"
    )