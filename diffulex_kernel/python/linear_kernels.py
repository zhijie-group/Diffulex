"""
W8A16, W4A16, W8A8, W4A8, FP8 W8A16, and FP8 W8A8 Linear GEMM kernels using TileLang.

- W8A16: int8 weight × bf16 activation matrix multiplication with per-channel dequantization.
- W4A16: int4 weight (packed in int8) × bf16 activation matrix multiplication with per-channel dequantization.
- W8A8: int8 activation × int8 weight matrix multiplication, output int32 accumulator.
- W4A8: int8 activation × int4 weight (packed in int8) matrix multiplication, output int32 accumulator.
- FP8 W8A16: FP8 weight (uint8 storage) × bf16 activation matrix multiplication with per-channel dequantization.
- FP8 W8A8: FP8 weight (uint8 storage) × FP8 activation (uint8 storage) matrix multiplication with fused scaling.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
from tvm import tir

from diffulex_kernel.python.auto_tuner import build_linear_configs

@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[3])
def w8a16_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """W8A16 GEMM kernel: bf16 activation × int8 weight (per-channel dequantized).
    
    Args:
        M: Number of rows in activation matrix A
        N: Number of output channels (rows in weight matrix B)
        K: Inner dimension (columns in A, rows in B)
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension
        num_stages: Number of pipeline stages
        threads: Number of threads per block
    
    Returns:
        Compiled TileLang kernel function with signature:
        kernel(A: bf16[M, K], B: int8[N, K], Scales: bf16[N], C: bf16[M, N]) -> None
    """
    # Fast path: only generate the simple copy-based kernel when all dims are perfectly tiled.
    # Otherwise, generate a masked (tail-safe) kernel to avoid falling back for non-multiple sizes.
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),           # activation, shape (M, K)
        B: T.Tensor((N, K), T.int8),              # quantized weight, shape (N, K)
        Scales: T.Tensor((N,), T.bfloat16),       # per-channel scales, shape (N,)
        C: T.Tensor((M, N), T.bfloat16),          # output, shape (M, N)
    ):
        """W8A16 GEMM kernel implementation.
        
        Computes C = (A @ q^T) * Scales where q is the int8 quantized weight and Scales is per-output-channel.
        This is mathematically equivalent to dequantizing weights inside the K loop, but avoids doing the
        multiply-by-scale for every (N, K) element in every K tile.
        
        This implementation follows the W4A8 pattern with fragments for proper pipelining.
        """
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_bf16 = tir.const(0, T.bfloat16)

            # Allocate shared memory buffers
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)
            
            # Allocate fragments (matching W4A8 pattern for proper pipelining)
            B_local = T.alloc_fragment((block_N, block_K), T.int8)
            B_bf16_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            B_bf16_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            
            # Allocate fragment for accumulation (use float32 for precision)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_scaled = T.alloc_fragment((block_M, block_N), T.bfloat16)
            
            # Optional: Add swizzled layout for B_shared (can improve performance)
            # T.annotate_layout({B_shared: tilelang.layout.make_swizzled_layout(B_shared)})
            
            # Clear accumulation buffer
            T.clear(C_local)
            
            # Pipeline over K dimension
            # Using the same pattern as W4A8: T.Pipelined(K // block_K, num_stages=num_stages)
            # The key: we copy B_shared -> B_local, dequantize to B_dequantize_local,
            # then copy to B_dequantize_prev_local before GEMM, matching W4A8 exactly
            # Note: num_stages must match the number of pipeline operations TileLang detects
            # For our case: copy A, copy B, copy B->local, dequantize, copy dequant->prev, gemm
            # This creates multiple pipeline stages, so we need to ensure num_stages is appropriate
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Load A and B tiles to shared memory
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                
                    # Copy B_shared to local fragment (required for proper pipelining)
                    T.copy(B_shared, B_local)
                
                    # Cast int8 -> bf16 (no scale here; apply scale once at output).
                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)
                
                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(B_bf16_local, B_bf16_prev_local)
                
                    # GEMM: C = A @ B_dequant^T
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            else:
                # Tail-safe kernel: mask-load A/B, mask-load scales (avoid OOB), store C with mask.
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Masked load A -> A_shared
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else(
                            (m < M) & (kk < K),
                            A[m, kk],
                            zero_bf16,
                        )

                    # Masked load B -> B_shared
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else(
                            (n < N) & (kk < K),
                            B[n, kk],
                            zero_i8,
                        )

                    # Copy B_shared to local fragment (required for proper pipelining)
                    T.copy(B_shared, B_local)

                    # Cast int8 -> bf16 (no scale here; apply scale once at output).
                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)

                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(B_bf16_local, B_bf16_prev_local)

                    # GEMM (padded with zeros for out-of-range A/B)
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            
            # Apply per-channel scale at output:
            # C[m, n] = (A @ q^T)[m, n] * Scales[n]
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    scale_f32 = Scales[bx * block_N + j].astype(T.float32)
                    C_scaled[i, j] = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                T.copy(
                    C_scaled,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    scale_bf16 = T.if_then_else(n < N, Scales[n], zero_bf16)
                    scale_f32 = scale_bf16.astype(T.float32)
                    C_scaled[i, j] = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = C_scaled[i, j]
    
    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[4])
def w8a16_gemm_bias(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """W8A16 GEMM kernel with fused bias: bf16 activation × int8 weight -> bf16 output, then add bias.

    Signature:
        kernel(A: bf16[M,K], B: int8[N,K], Scales: bf16[N], Bias: bf16[N], C: bf16[M,N]) -> None
    """
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.int8),
        Scales: T.Tensor((N,), T.bfloat16),
        # NOTE: keep Bias as fp16 to avoid adapter issues observed with 1D bf16 inputs.
        Bias: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_bf16 = tir.const(0, T.bfloat16)

            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)

            B_local = T.alloc_fragment((block_N, block_K), T.int8)
            B_bf16_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            B_bf16_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)

            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)

            T.clear(C_local)

            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)

                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)
                    T.copy(B_bf16_local, B_bf16_prev_local)

                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else(
                            (m < M) & (kk < K),
                            A[m, kk],
                            zero_bf16,
                        )
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else(
                            (n < N) & (kk < K),
                            B[n, kk],
                            zero_i8,
                        )

                    T.copy(B_shared, B_local)
                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)
                    T.copy(B_bf16_local, B_bf16_prev_local)

                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)

            # Apply per-channel scale and bias at output:
            # C[m,n] = (A@q^T)[m,n] * Scales[n] + Bias[n]
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    n = bx * block_N + j
                    scale_f32 = Scales[n].astype(T.float32)
                    bias_f32 = Bias[n].astype(T.float32)
                    C_out[i, j] = (C_local[i, j] * scale_f32 + bias_f32).astype(T.bfloat16)
                T.copy(
                    C_out,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    scale_bf16 = T.if_then_else(n < N, Scales[n], zero_bf16)
                    bias_f16 = T.if_then_else(n < N, Bias[n], tir.const(0, T.float16))
                    scale_f32 = scale_bf16.astype(T.float32)
                    bias_f32 = bias_f16.astype(T.float32)
                    val = (C_local[i, j] * scale_f32 + bias_f32).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val

    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[3])
def w4a16_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """W4A16 GEMM kernel: bf16 activation × int4 weight (packed in int8, per-channel dequantized).
    
    Args:
        M: Number of rows in activation matrix A
        N: Number of output channels (rows in weight matrix B)
        K: Inner dimension (columns in A, rows in B)
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension
        num_stages: Number of pipeline stages
        threads: Number of threads per block
    
    Returns:
        Compiled TileLang kernel function with signature:
        kernel(A: bf16[M, K], B_packed: int8[N, (K+1)//2], Scales: bf16[N], C: bf16[M, N]) -> None
        
    Note:
        B_packed is int4 weights packed into int8 format. Each int8 byte contains 2 int4 values:
        - Lower 4 bits: first int4 value (in range [0, 15], representing [-8, 7])
        - Upper 4 bits: second int4 value (in range [0, 15], representing [-8, 7])
    """
    # Fast path: only generate the simple copy-based kernel when all dims are perfectly tiled.
    # Otherwise, generate a masked (tail-safe) kernel to avoid falling back for non-multiple sizes.
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)
    
    # Packed size: (K + 1) // 2
    packed_K = (K + 1) // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),           # activation, shape (M, K)
        B_packed: T.Tensor((N, packed_K), T.int8), # packed int4 weight, shape (N, (K+1)//2)
        Scales: T.Tensor((N,), T.bfloat16),        # per-channel scales, shape (N,)
        C: T.Tensor((M, N), T.bfloat16),           # output, shape (M, N)
    ):
        """W4A16 GEMM kernel implementation.
        
        Computes C = A @ B_dequant^T where:
        - B_packed[i, j] contains 2 int4 values (packed in int8)
        - Each int4 value is unpacked to q in [-8, 7]
        - Per-channel dequantization is applied as: (A @ q^T) * Scales[n]  (Scales is per-output-channel)
        
        This implementation avoids per-element dequantization inside the K loop by
        factoring the per-channel scale to an output-side column scaling step, which
        substantially reduces work vs. dequantizing every weight element.
        """
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_bf16 = tir.const(0, T.bfloat16)
            
            # Constants for int4 unpacking
            int4_offset = tir.const(8, T.int8)  # Offset to convert [0, 15] to [-8, 7]
            mask_lower = tir.const(0x0F, T.int8)  # Mask for lower 4 bits
            mask_upper_shift = tir.const(4, T.int8)  # Shift for upper 4 bits

            # Allocate shared memory buffers
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_packed_shared = T.alloc_shared((block_N, (block_K + 1) // 2), T.int8)
            
            # Allocate fragments (matching W8A16 pattern for proper pipelining)
            B_packed_local = T.alloc_fragment((block_N, (block_K + 1) // 2), T.int8)
            B_unpacked_local = T.alloc_fragment((block_N, block_K), T.int8)  # Unpacked int4 (as int8)
            B_bf16_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            B_bf16_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            
            # Allocate fragment for accumulation (use float32 for precision)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_scaled = T.alloc_fragment((block_M, block_N), T.bfloat16)
            
            # Clear accumulation buffer
            T.clear(C_local)
            
            # Pipeline over K dimension
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Load A tile to shared memory
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    
                    # Load B_packed tile to shared memory
                    packed_k_start = (k * block_K) // 2  # Packed index for K dimension
                    T.copy(B_packed[bx * block_N, packed_k_start], B_packed_shared)
                    
                    # Copy B_packed_shared to local fragment
                    T.copy(B_packed_shared, B_packed_local)
                    
                    # Unpack int4 from packed int8 (TileLang-friendly indexing):
                    # B_unpacked_local is indexed by (i, j) directly to avoid indices-mismatch issues.
                    for i, j in T.Parallel(block_N, block_K):
                        j_packed = j // 2
                        packed_byte = B_packed_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        B_unpacked_local[i, j] = T.if_then_else(is_lower, lower_int4, upper_int4)

                    # Cast int4 (stored as int8) -> bf16 once per element (no scale here).
                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_unpacked_local[i, j].astype(T.float32).astype(T.bfloat16)

                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(B_bf16_local, B_bf16_prev_local)
                    
                    # GEMM: C = A @ B_dequant^T
                    # Here B is q (int4) cast to bf16; scale is applied once after K-accumulation.
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            else:
                # Tail-safe kernel: mask-load A/B_packed, unpack, dequantize, store C with mask
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Masked load A -> A_shared
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else(
                            (m < M) & (kk < K),
                            A[m, kk],
                            zero_bf16,
                        )

                    # Masked load B_packed -> B_packed_shared
                    packed_k_start = (k * block_K) // 2
                    packed_k_size = (block_K + 1) // 2
                    for i, j_packed in T.Parallel(block_N, packed_k_size):
                        n = bx * block_N + i
                        packed_idx = packed_k_start + j_packed
                        B_packed_shared[i, j_packed] = T.if_then_else(
                            (n < N) & (packed_idx < packed_K),
                            B_packed[n, packed_idx],
                            zero_i8,
                        )

                    # Copy B_packed_shared to local fragment
                    T.copy(B_packed_shared, B_packed_local)
                    
                    # Unpack int4 from int8 with boundary checks
                    for i, j in T.Parallel(block_N, block_K):
                        kk = k * block_K + j
                        # Convert to local packed index within this block
                        j_packed = j // 2
                        packed_byte = B_packed_local[i, j_packed]
                        
                        # Extract both lower and upper 4 bits
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset  # Convert [0, 15] to [-8, 7]
                        upper_int4 = upper_uint - int4_offset  # Convert [0, 15] to [-8, 7]
                        
                        # Select the appropriate value based on whether j is even (lower) or odd (upper)
                        is_lower = (j % 2) == 0
                        int4_val = T.if_then_else(is_lower, lower_int4, upper_int4)
                        
                        # Mask out-of-bound values to zero
                        in_bounds = (kk < K) & (j < block_K)
                        B_unpacked_local[i, j] = T.if_then_else(in_bounds, int4_val, zero_i8)
                    
                    # Cast int4 -> bf16 (no scale here).
                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_unpacked_local[i, j].astype(T.float32).astype(T.bfloat16)

                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(B_bf16_local, B_bf16_prev_local)

                    # GEMM (padded with zeros for out-of-range A/B)
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            
            # Apply per-channel scale at output (equivalent to weight-side dequantization):
            # C[m, n] = (A @ q^T)[m, n] * Scales[n]
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    scale_f32 = Scales[bx * block_N + j].astype(T.float32)
                    C_scaled[i, j] = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                T.copy(
                    C_scaled,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    scale_bf16 = T.if_then_else(n < N, Scales[n], zero_bf16)
                    scale_f32 = scale_bf16.astype(T.float32)
                    C_scaled[i, j] = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = C_scaled[i, j]
    
    return main


@tilelang.jit(out_idx=[2])
def w8a8_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """W8A8 GEMM kernel: int8 activation × int8 weight matrix multiplication.
    
    Args:
        M: Number of rows in activation matrix A
        N: Number of output channels (rows in weight matrix B)
        K: Inner dimension (columns in A, rows in B)
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension
        num_stages: Number of pipeline stages
        threads: Number of threads per block
    
    Returns:
        Compiled TileLang kernel function with signature:
        kernel(A: int8[M, K], B: int8[N, K], C: int32[M, N]) -> None
        
    Note:
        - Input A is int8 quantized activation [M, K]
        - Input B is int8 quantized weight [N, K] (GEMM uses transpose_B=True internally)
        - Output C is int32 accumulator [M, N]
        - Scales (activation scales and weight scales) are applied externally after this kernel
    """
    # Fast path: only generate the simple copy-based kernel when all dims are perfectly tiled.
    # Otherwise, generate a masked (tail-safe) kernel to avoid falling back for non-multiple sizes.
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),           # quantized activation, shape (M, K)
        B: T.Tensor((N, K), T.int8),           # quantized weight, shape (N, K)
        C: T.Tensor((M, N), T.int32),          # output accumulator, shape (M, N)
    ):
        """W8A8 GEMM kernel implementation.
        
        Computes C = A @ B where all inputs are int8 and output is int32.
        This avoids overflow during accumulation by using int32 intermediate results.
        """
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_i32 = tir.const(0, T.int32)

            # Allocate shared memory buffers
            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)
            
            # Allocate fragments for pipelining
            A_local = T.alloc_fragment((block_M, block_K), T.int8)
            B_local = T.alloc_fragment((block_N, block_K), T.int8)
            A_local_prev = T.alloc_fragment((block_M, block_K), T.int8)
            B_local_prev = T.alloc_fragment((block_N, block_K), T.int8)
            
            # Allocate fragment for accumulation (use int32 for precision)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            
            # Clear accumulation buffer
            T.clear(C_local)
            
            # Pipeline over K dimension
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Load A and B tiles to shared memory
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    # B is stored as [N, K]; GEMM uses transpose_B=True.
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                
                    # Copy to local fragments (required for proper pipelining)
                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)
                
                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)
                
                    # GEMM: C = A @ B^T (int8 x int8 -> int32 accumulation).
                    # Important: use int8 operands; TileLang lowers to the appropriate int8 GEMM path.
                    T.gemm(A_local_prev, B_local_prev, C_local, transpose_B=True)
            else:
                # Tail-safe kernel: mask-load A/B, store C with mask
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Masked load A -> A_shared
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else(
                            (m < M) & (kk < K),
                            A[m, kk],
                            zero_i8,
                        )

                    # Masked load B -> B_shared
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else(
                            (kk < K) & (n < N),
                            B[n, kk],
                            zero_i8,
                        )

                    # Copy to local fragments
                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)

                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)

                    # GEMM (padded with zeros for out-of-range A/B)
                    T.gemm(A_local_prev, B_local_prev, C_local, transpose_B=True)
            
            # Store result to output
            if aligned:
                T.copy(
                    C_local,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    if (m < M) & (n < N):
                        C[m, n] = C_local[i, j]
    
    return main


@tilelang.jit(out_idx=[1, 2])
def w8a8_act_quant(
    M: int,
    K: int,
    block_M: int = 64,
    block_K: int = 256,
    threads: int = 128,
):
    """Fused per-row symmetric int8 activation quantization (BF16 -> INT8 + per-row scales).

    This kernel replaces the Python aten chain:
        abs -> amax(reduce) -> div -> round -> clamp -> to(int8)

    For each row m:
        absmax = max(abs(x[m, :]))
        scale[m] = max(absmax, eps) / 127
        x_q[m, k] = clamp(round(x[m, k] / scale[m]), -127, 127).astype(int8)

    Returns:
        kernel(A: bf16[M, K], A_q: int8[M, K], Scales: float32[M]) -> None
        With out_idx=[1,2], the Python wrapper returns (A_q, Scales).
    """

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        A_q: T.Tensor((M, K), T.int8),
        Scales: T.Tensor((M,), T.float32),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx,):
            zero_f32 = tir.const(0.0, T.float32)
            eps_f32 = tir.const(1e-8, T.float32)
            inv127 = tir.const(1.0 / 127.0, T.float32)
            neg127 = tir.const(-127.0, T.float32)
            pos127 = tir.const(127.0, T.float32)

            # Tile buffers for abs/max reduction and scale broadcasting.
            abs_tile = T.alloc_fragment((block_M, block_K), T.float32)
            tile_max = T.alloc_fragment((block_M,), T.float32)
            row_max = T.alloc_fragment((block_M,), T.float32)
            scales_local = T.alloc_fragment((block_M,), T.float32)

            # Initialize running max to 0 (absmax is >=0).
            T.fill(row_max, zero_f32)

            # Pass 1: compute per-row absmax.
            for k0 in range(T.ceildiv(K, block_K)):
                for i, j in T.Parallel(block_M, block_K):
                    m = bx * block_M + i
                    kk = k0 * block_K + j
                    v = T.if_then_else(
                        (m < M) & (kk < K),
                        A[m, kk].astype(T.float32),
                        zero_f32,
                    )
                    # abs(v) without relying on optional intrinsics
                    abs_tile[i, j] = T.if_then_else(v < zero_f32, -v, v)

                T.fill(tile_max, zero_f32)
                T.reduce_max(abs_tile, tile_max, dim=1, clear=True)

                for i in T.Parallel(block_M):
                    row_max[i] = T.max(row_max[i], tile_max[i])

            # Compute scales once and optionally store to global output.
            for i in T.Parallel(block_M):
                m = bx * block_M + i
                s = T.max(row_max[i], eps_f32) * inv127
                scales_local[i] = s
                if m < M:
                    Scales[m] = s

            # Pass 2: quantize using the computed per-row scales.
            for k0 in range(T.ceildiv(K, block_K)):
                for i, j in T.Parallel(block_M, block_K):
                    m = bx * block_M + i
                    kk = k0 * block_K + j
                    if (m < M) & (kk < K):
                        s = scales_local[i]
                        x = A[m, kk].astype(T.float32) / s
                        q = T.min(T.max(T.round(x), neg127), pos127)
                        A_q[m, kk] = q.astype(T.int8)

    return main


@tilelang.jit(out_idx=[4])
def w8a8_scaled_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """W8A8 GEMM kernel with fused scaling: int8 activation × int8 weight -> bf16 output.

    This kernel computes:
        C[m, n] = (sum_k A_i8[m, k] * B_i8[k, n]) * x_scale[m] * w_scale[n]

    Args:
        M, N, K: GEMM sizes
        x_scales: float32[M] per-row scales for activation quantization
        w_scales: bf16[N] per-output-channel scales for weight quantization

    Returns:
        kernel(A: int8[M,K], B: int8[K,N], x_scales: float32[M], w_scales: bf16[N], C: bf16[M,N]) -> None
    """
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        B: T.Tensor((N, K), T.int8),
        XScales: T.Tensor((M,), T.float32),
        WScales: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_i32 = tir.const(0, T.int32)
            zero_f32 = tir.const(0.0, T.float32)
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f16 = tir.const(0, T.float16)

            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)

            A_local = T.alloc_fragment((block_M, block_K), T.int8)
            B_local = T.alloc_fragment((block_N, block_K), T.int8)
            A_local_prev = T.alloc_fragment((block_M, block_K), T.int8)
            B_local_prev = T.alloc_fragment((block_N, block_K), T.int8)

            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)

            T.clear(C_local)

            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    # B is stored as [N, K]; GEMM uses transpose_B=True.
                    T.copy(B[bx * block_N, k * block_K], B_shared)

                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)

                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)

                    # int8 x int8 -> int32 accumulation
                    T.gemm(A_local_prev, B_local_prev, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_i8)

                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else((kk < K) & (n < N), B[n, kk], zero_i8)

                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)

                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)

                    T.gemm(A_local_prev, B_local_prev, C_local, transpose_B=True)

            # Fused scaling + store
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = XScales[m]  # float32
                    w_s = WScales[n].astype(T.float32)
                    C_out[i, j] = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                T.copy(
                    C_out,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = T.if_then_else(m < M, XScales[m], zero_f32)
                    w_s_f16 = T.if_then_else(n < N, WScales[n], zero_f16)
                    w_s = w_s_f16.astype(T.float32)
                    val = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val

    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[3])
def w8a8_fused_act_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 3,
    threads: int = 128,
):
    """W8A8 GEMM with fused activation quantization: bf16 activation -> int8 GEMM -> bf16 output.

    This kernel computes per-row scales internally (absmax / 127), quantizes A on the fly,
    then runs int8 GEMM against B (int8) and applies per-row/per-channel scaling.
    
    Optimizations:
    - Removed unnecessary fragment copies (A_local, A_local_prev, B_local, B_local_prev)
    - Direct GEMM from shared memory (A_shared, B_shared -> C_local)
    - Added swizzled layout for shared memory to reduce bank conflicts
    - Increased num_stages to 3 for better latency hiding
    """
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.int8),
        WScales: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_i32 = tir.const(0, T.int32)
            zero_f32 = tir.const(0.0, T.float32)
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f16 = tir.const(0, T.float16)
            eps_f32 = tir.const(1e-8, T.float32)
            inv127 = tir.const(1.0 / 127.0, T.float32)
            neg127 = tir.const(-127.0, T.float32)
            pos127 = tir.const(127.0, T.float32)

            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)

            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)

            row_max = T.alloc_reducer((block_M,), T.float32, op="max")
            scales_smem = T.alloc_shared((block_M,), T.float32)

            # Add swizzled layout for shared memory to reduce bank conflicts
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_shared: tilelang.layout.make_swizzled_layout(B_shared),
            })

            T.clear(C_local)
            # absmax is non-negative; 0 is a safe initializer for max-reduction.
            T.fill(row_max, zero_f32)

            # Pass 1: compute per-row absmax.
            if aligned:
                num_k_blocks = K // block_K
                for k0 in range(num_k_blocks):
                    for i, j in T.Parallel(block_M, block_K):
                        v = A[by * block_M + i, k0 * block_K + j].astype(T.float32)
                        av = T.if_then_else(v < zero_f32, -v, v)
                        row_max[i] = T.max(row_max[i], av)
            else:
                for k0 in range(T.ceildiv(K, block_K)):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k0 * block_K + j
                        v = T.if_then_else((m < M) & (kk < K), A[m, kk].astype(T.float32), zero_f32)
                        av = T.if_then_else(v < zero_f32, -v, v)
                        row_max[i] = T.max(row_max[i], av)

            # Materialize reducer results.
            T.finalize_reducer(row_max)

            # Compute per-row scales.
            for i in T.Parallel(block_M):
                scales_smem[i] = T.max(row_max[i], eps_f32) * inv127

            # Pass 2: quantize A on the fly and GEMM.
            # Optimization: removed A_local, A_local_prev, B_local, B_local_prev
            # Direct GEMM from shared memory saves 4 fragment copies per iteration!
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Quantize A directly into A_shared
                    for i, j in T.Parallel(block_M, block_K):
                        s = scales_smem[i]
                        x = A[by * block_M + i, k * block_K + j].astype(T.float32) / s
                        q = T.min(T.max(T.round(x), neg127), pos127)
                        A_shared[i, j] = q.astype(T.int8)

                    # Load B directly into B_shared
                    # B is stored as [N, K]; GEMM uses transpose_B=True.
                    T.copy(B[bx * block_N, k * block_K], B_shared)

                    # Direct GEMM from shared memory - no fragment copies!
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Quantize A directly into A_shared with bounds checking
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        if (m < M) & (kk < K):
                            s = scales_smem[i]
                            x = A[m, kk].astype(T.float32) / s
                            q = T.min(T.max(T.round(x), neg127), pos127)
                            A_shared[i, j] = q.astype(T.int8)
                        else:
                            A_shared[i, j] = zero_i8

                    # Load B directly into B_shared with bounds checking
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else((kk < K) & (n < N), B[n, kk], zero_i8)

                    # Direct GEMM from shared memory - no fragment copies!
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            # Fused scaling + store
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = scales_smem[i]
                    w_s = WScales[n].astype(T.float32)
                    C_out[i, j] = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                T.copy(
                    C_out,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = T.if_then_else(m < M, scales_smem[i], zero_f32)
                    w_s_f16 = T.if_then_else(n < N, WScales[n], zero_f16)
                    w_s = w_s_f16.astype(T.float32)
                    val = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val

    return main


@tilelang.jit(out_idx=[2])
def w4a8_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """W4A8 GEMM kernel: int8 activation × int4 weight (packed in int8) matrix multiplication.
    
    Args:
        M: Number of rows in activation matrix A
        N: Number of output channels (rows in weight matrix B)
        K: Inner dimension (columns in A, rows in B)
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension
        num_stages: Number of pipeline stages
        threads: Number of threads per block
    
    Returns:
        Compiled TileLang kernel function with signature:
        kernel(A: int8[M, K], B_packed: int8[N, (K+1)//2], C: int32[M, N]) -> None
        
    Note:
        - Input A is int8 quantized activation [M, K]
        - Input B_packed is int4 weights packed into int8 format [N, (K+1)//2]
        - Output C is int32 accumulator [M, N]
        - Scales (activation scales and weight scales) are applied externally after this kernel
        - B_packed is int4 weights packed into int8 format. Each int8 byte contains 2 int4 values:
          - Lower 4 bits: first int4 value (in range [0, 15], representing [-8, 7])
          - Upper 4 bits: second int4 value (in range [0, 15], representing [-8, 7])
    """
    # Fast path: only generate the simple copy-based kernel when all dims are perfectly tiled.
    # Otherwise, generate a masked (tail-safe) kernel to avoid falling back for non-multiple sizes.
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)
    
    # Packed size: (K + 1) // 2
    packed_K = (K + 1) // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),           # quantized activation, shape (M, K)
        B_packed: T.Tensor((N, packed_K), T.int8), # packed int4 weight, shape (N, (K+1)//2)
        C: T.Tensor((M, N), T.int32),          # output accumulator, shape (M, N)
    ):
        """W4A8 GEMM kernel implementation.
        
        Computes C = A @ B_unpacked^T where:
        - B_packed[i, j] contains 2 int4 values (packed in int8)
        - Each int4 value is unpacked to q in [-8, 7]
        - All operations use int8/int32 to avoid overflow during accumulation
        """
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_i32 = tir.const(0, T.int32)
            
            # Constants for int4 unpacking
            int4_offset = tir.const(8, T.int8)  # Offset to convert [0, 15] to [-8, 7]
            mask_lower = tir.const(0x0F, T.int8)  # Mask for lower 4 bits
            mask_upper_shift = tir.const(4, T.int8)  # Shift for upper 4 bits

            # Allocate shared memory buffers
            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_packed_shared = T.alloc_shared((block_N, (block_K + 1) // 2), T.int8)
            
            # Allocate fragments for pipelining
            A_local = T.alloc_fragment((block_M, block_K), T.int8)
            B_packed_local = T.alloc_fragment((block_N, (block_K + 1) // 2), T.int8)
            B_unpacked_local = T.alloc_fragment((block_N, block_K), T.int8)  # Unpacked int4 (as int8)
            A_local_prev = T.alloc_fragment((block_M, block_K), T.int8)
            B_unpacked_local_prev = T.alloc_fragment((block_N, block_K), T.int8)
            
            # Allocate fragment for accumulation (use int32 for precision)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            
            # Clear accumulation buffer
            T.clear(C_local)
            
            # Pipeline over K dimension
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Load A tile to shared memory
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    
                    # Load B_packed tile to shared memory
                    packed_k_start = (k * block_K) // 2  # Packed index for K dimension
                    T.copy(B_packed[bx * block_N, packed_k_start], B_packed_shared)
                    
                    # Copy to local fragments
                    T.copy(A_shared, A_local)
                    T.copy(B_packed_shared, B_packed_local)
                    
                    # Unpack int4 from packed int8
                    for i, j in T.Parallel(block_N, block_K):
                        j_packed = j // 2
                        packed_byte = B_packed_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        B_unpacked_local[i, j] = T.if_then_else(is_lower, lower_int4, upper_int4)

                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(A_local, A_local_prev)
                    T.copy(B_unpacked_local, B_unpacked_local_prev)
                    
                    # GEMM: C = A @ B_unpacked^T (int8 x int8 -> int32 accumulation).
                    # Use int8 operands; TileLang lowers to the proper int8 GEMM path.
                    T.gemm(A_local_prev, B_unpacked_local_prev, C_local, transpose_B=True)
            else:
                # Tail-safe kernel: mask-load A/B_packed, unpack, store C with mask
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Masked load A -> A_shared
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else(
                            (m < M) & (kk < K),
                            A[m, kk],
                            zero_i8,
                        )

                    # Masked load B_packed -> B_packed_shared
                    packed_k_start = (k * block_K) // 2
                    packed_k_size = (block_K + 1) // 2
                    for i, j_packed in T.Parallel(block_N, packed_k_size):
                        n = bx * block_N + i
                        packed_idx = packed_k_start + j_packed
                        B_packed_shared[i, j_packed] = T.if_then_else(
                            (n < N) & (packed_idx < packed_K),
                            B_packed[n, packed_idx],
                            zero_i8,
                        )

                    # Copy to local fragments
                    T.copy(A_shared, A_local)
                    T.copy(B_packed_shared, B_packed_local)
                    
                    # Unpack int4 from int8 with boundary checks
                    for i, j in T.Parallel(block_N, block_K):
                        kk = k * block_K + j
                        j_packed = j // 2
                        packed_byte = B_packed_local[i, j_packed]
                        
                        # Extract both lower and upper 4 bits
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        
                        # Select the appropriate value based on whether j is even (lower) or odd (upper)
                        is_lower = (j % 2) == 0
                        int4_val = T.if_then_else(is_lower, lower_int4, upper_int4)
                        
                        # Mask out-of-bound values to zero
                        in_bounds = (kk < K) & (j < block_K)
                        B_unpacked_local[i, j] = T.if_then_else(in_bounds, int4_val, zero_i8)

                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(A_local, A_local_prev)
                    T.copy(B_unpacked_local, B_unpacked_local_prev)

                    # GEMM (padded with zeros for out-of-range A/B)
                    T.gemm(A_local_prev, B_unpacked_local_prev, C_local, transpose_B=True)
            
            # Store result to output
            if aligned:
                T.copy(
                    C_local,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    if (m < M) & (n < N):
                        C[m, n] = C_local[i, j]
    
    return main


@tilelang.jit(out_idx=[4])
def w4a8_scaled_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """W4A8 GEMM kernel with fused scaling: int8 activation × packed int4 weight -> bf16 output.

    Computes:
        C[m, n] = (sum_k A_i8[m,k] * q_i4[n,k]) * x_scale[m] * w_scale[n]

    Where q_i4 is unpacked from B_packed on the fly into int8 in [-8, 7].
    """
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)
    packed_K = (K + 1) // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),
        B_packed: T.Tensor((N, packed_K), T.int8),
        XScales: T.Tensor((M,), T.float32),
        WScales: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_i32 = tir.const(0, T.int32)
            zero_f32 = tir.const(0.0, T.float32)
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f16 = tir.const(0, T.float16)

            int4_offset = tir.const(8, T.int8)
            mask_lower = tir.const(0x0F, T.int8)
            mask_upper_shift = tir.const(4, T.int8)

            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_packed_shared = T.alloc_shared((block_N, (block_K + 1) // 2), T.int8)

            A_local = T.alloc_fragment((block_M, block_K), T.int8)
            B_packed_local = T.alloc_fragment((block_N, (block_K + 1) // 2), T.int8)
            B_unpacked_local = T.alloc_fragment((block_N, block_K), T.int8)
            A_local_prev = T.alloc_fragment((block_M, block_K), T.int8)
            B_unpacked_local_prev = T.alloc_fragment((block_N, block_K), T.int8)

            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)

            T.clear(C_local)

            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)

                    packed_k_start = (k * block_K) // 2
                    T.copy(B_packed[bx * block_N, packed_k_start], B_packed_shared)

                    T.copy(A_shared, A_local)
                    T.copy(B_packed_shared, B_packed_local)

                    for i, j in T.Parallel(block_N, block_K):
                        j_packed = j // 2
                        packed_byte = B_packed_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        B_unpacked_local[i, j] = T.if_then_else(is_lower, lower_int4, upper_int4)

                    T.copy(A_local, A_local_prev)
                    T.copy(B_unpacked_local, B_unpacked_local_prev)

                    T.gemm(A_local_prev, B_unpacked_local_prev, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_i8)

                    packed_k_start = (k * block_K) // 2
                    packed_k_size = (block_K + 1) // 2
                    for i, j_packed in T.Parallel(block_N, packed_k_size):
                        n = bx * block_N + i
                        packed_idx = packed_k_start + j_packed
                        B_packed_shared[i, j_packed] = T.if_then_else(
                            (n < N) & (packed_idx < packed_K),
                            B_packed[n, packed_idx],
                            zero_i8,
                        )

                    T.copy(A_shared, A_local)
                    T.copy(B_packed_shared, B_packed_local)

                    for i, j in T.Parallel(block_N, block_K):
                        kk = k * block_K + j
                        j_packed = j // 2
                        packed_byte = B_packed_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        int4_val = T.if_then_else(is_lower, lower_int4, upper_int4)
                        in_bounds = (kk < K) & (j < block_K)
                        B_unpacked_local[i, j] = T.if_then_else(in_bounds, int4_val, zero_i8)

                    T.copy(A_local, A_local_prev)
                    T.copy(B_unpacked_local, B_unpacked_local_prev)

                    T.gemm(A_local_prev, B_unpacked_local_prev, C_local, transpose_B=True)

            # Fused scaling + store
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = XScales[m]
                    w_s = WScales[n].astype(T.float32)
                    C_out[i, j] = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                T.copy(
                    C_out,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = T.if_then_else(m < M, XScales[m], zero_f32)
                    w_s_f16 = T.if_then_else(n < N, WScales[n], zero_f16)
                    w_s = w_s_f16.astype(T.float32)
                    val = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val

    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[3])
def w4a8_fused_act_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 3,
    threads: int = 128,
):
    """W4A8 GEMM with fused activation quantization: bf16 activation -> int8 GEMM -> bf16 output.

    This kernel computes per-row scales internally (absmax / 127), quantizes A on the fly,
    unpacks packed int4 weights, then applies fused scaling.
    
    Optimizations:
    - Reduced fragment copies: unpack B directly in shared memory
    - Added swizzled layout for shared memory
    - Increased num_stages to 3 for better latency hiding
    """
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)
    packed_K = (K + 1) // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B_packed: T.Tensor((N, packed_K), T.int8),
        WScales: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_i32 = tir.const(0, T.int32)
            zero_f32 = tir.const(0.0, T.float32)
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f16 = tir.const(0, T.float16)
            eps_f32 = tir.const(1e-8, T.float32)
            inv127 = tir.const(1.0 / 127.0, T.float32)
            neg127 = tir.const(-127.0, T.float32)
            pos127 = tir.const(127.0, T.float32)

            int4_offset = tir.const(8, T.int8)
            mask_lower = tir.const(0x0F, T.int8)
            mask_upper_shift = tir.const(4, T.int8)

            A_shared = T.alloc_shared((block_M, block_K), T.int8)
            B_packed_shared = T.alloc_shared((block_N, (block_K + 1) // 2), T.int8)
            B_unpacked_shared = T.alloc_shared((block_N, block_K), T.int8)

            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)

            row_max = T.alloc_reducer((block_M,), T.float32, op="max")
            scales_smem = T.alloc_shared((block_M,), T.float32)

            # Add swizzled layout for shared memory
            T.annotate_layout({
                A_shared: tilelang.layout.make_swizzled_layout(A_shared),
                B_unpacked_shared: tilelang.layout.make_swizzled_layout(B_unpacked_shared),
            })

            T.clear(C_local)
            # absmax is non-negative; 0 is a safe initializer for max-reduction.
            T.fill(row_max, zero_f32)

            # Pass 1: compute per-row absmax.
            if aligned:
                num_k_blocks = K // block_K
                for k0 in range(num_k_blocks):
                    for i, j in T.Parallel(block_M, block_K):
                        v = A[by * block_M + i, k0 * block_K + j].astype(T.float32)
                        av = T.if_then_else(v < zero_f32, -v, v)
                        row_max[i] = T.max(row_max[i], av)
            else:
                for k0 in range(T.ceildiv(K, block_K)):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k0 * block_K + j
                        v = T.if_then_else((m < M) & (kk < K), A[m, kk].astype(T.float32), zero_f32)
                        av = T.if_then_else(v < zero_f32, -v, v)
                        row_max[i] = T.max(row_max[i], av)

            # Materialize reducer results.
            T.finalize_reducer(row_max)

            # Compute per-row scales.
            for i in T.Parallel(block_M):
                scales_smem[i] = T.max(row_max[i], eps_f32) * inv127

            # Pass 2: quantize A, unpack B, GEMM.
            # Optimization: unpack B directly in shared memory, avoid fragment copies
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Quantize A directly into A_shared
                    for i, j in T.Parallel(block_M, block_K):
                        s = scales_smem[i]
                        x = A[by * block_M + i, k * block_K + j].astype(T.float32) / s
                        q = T.min(T.max(T.round(x), neg127), pos127)
                        A_shared[i, j] = q.astype(T.int8)

                    # Load B_packed into shared memory
                    packed_k_start = (k * block_K) // 2
                    T.copy(B_packed[bx * block_N, packed_k_start], B_packed_shared)

                    # Unpack B directly in shared memory
                    for i, j in T.Parallel(block_N, block_K):
                        j_packed = j // 2
                        packed_byte = B_packed_shared[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        # NOTE: Avoid introducing a let-bound var (e.g., `is_lower`) inside a fused/vectorized
                        # Parallel loop. Some TileLang/TVM lower passes may attempt to re-bind the same Var
                        # with different loop symbols and fail with:
                        #   "Trying to update var 'is_lower' with a different value"
                        B_unpacked_shared[i, j] = T.if_then_else((j % 2) == 0, lower_int4, upper_int4)

                    # Direct GEMM from shared memory - no fragment copies!
                    T.gemm(A_shared, B_unpacked_shared, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Quantize A directly into A_shared with bounds checking
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        if (m < M) & (kk < K):
                            s = scales_smem[i]
                            x = A[m, kk].astype(T.float32) / s
                            q = T.min(T.max(T.round(x), neg127), pos127)
                            A_shared[i, j] = q.astype(T.int8)
                        else:
                            A_shared[i, j] = zero_i8

                    # Load B_packed into shared memory with bounds checking
                    packed_k_start = (k * block_K) // 2
                    packed_k_size = (block_K + 1) // 2
                    for i, j_packed in T.Parallel(block_N, packed_k_size):
                        n = bx * block_N + i
                        packed_idx = packed_k_start + j_packed
                        B_packed_shared[i, j_packed] = T.if_then_else(
                            (n < N) & (packed_idx < packed_K),
                            B_packed[n, packed_idx],
                            zero_i8,
                        )

                    # Unpack B directly in shared memory with bounds checking
                    for i, j in T.Parallel(block_N, block_K):
                        kk = k * block_K + j
                        j_packed = j // 2
                        packed_byte = B_packed_shared[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        int4_val = T.if_then_else((j % 2) == 0, lower_int4, upper_int4)
                        in_bounds = (kk < K) & (j < block_K)
                        B_unpacked_shared[i, j] = T.if_then_else(in_bounds, int4_val, zero_i8)

                    # Direct GEMM from shared memory - no fragment copies!
                    T.gemm(A_shared, B_unpacked_shared, C_local, transpose_B=True)

            # Fused scaling + store
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = scales_smem[i]
                    w_s = WScales[n].astype(T.float32)
                    C_out[i, j] = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                T.copy(
                    C_out,
                    C[
                        by * block_M : (by + 1) * block_M,
                        bx * block_N : (bx + 1) * block_N,
                    ],
                )
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = T.if_then_else(m < M, scales_smem[i], zero_f32)
                    w_s_f16 = T.if_then_else(n < N, WScales[n], zero_f16)
                    w_s = w_s_f16.astype(T.float32)
                    val = (C_local[i, j].astype(T.float32) * x_s * w_s).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val

    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[3])
def fp8_e4m3_w8a16_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """FP8 E4M3 W8A16 GEMM kernel: bf16 activation × FP8 E4M3 weight (uint8 storage, per-channel dequantized)."""
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        # IMPORTANT: pass fp8 tensors from PyTorch by using `uint8_tensor.view(torch_fp8_dtype)`.
        # Do NOT pass raw uint8 storage here, otherwise we would need reinterpret logic and lose performance.
        B: T.Tensor((N, K), T.float8_e4m3fn),
        Scales: T.Tensor((N,), T.float32),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f32 = tir.const(0.0, T.float32)
            zero_fp8 = tir.const(0, T.float8_e4m3fn)

            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.float8_e4m3fn)

            # Follow the same pipeline pattern as int8 `w8a16_gemm`:
            # B_shared -> B_local -> (cast) B_bf16_local -> B_bf16_prev_local -> GEMM
            B_local = T.alloc_fragment((block_N, block_K), T.float8_e4m3fn)
            B_bf16_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            B_bf16_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_scaled = T.alloc_fragment((block_M, block_N), T.bfloat16)
            
            T.clear(C_local)
            
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.copy(B_shared, B_local)

                    # Cast fp8 -> fp32 -> bf16 (avoid fp16/half path, which can trigger cutlass bf16 ambiguity).
                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)

                    T.copy(B_bf16_local, B_bf16_prev_local)
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_bf16)

                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else((n < N) & (kk < K), B[n, kk], zero_fp8)

                    T.copy(B_shared, B_local)

                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)

                    T.copy(B_bf16_local, B_bf16_prev_local)
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            
            # Apply per-channel scale at output: C[m, n] = (A @ q_fp8^T)[m, n] * Scales[n]
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    scale_f32 = Scales[bx * block_N + j]
                    C_scaled[i, j] = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                T.copy(C_scaled, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    scale_f32 = T.if_then_else(n < N, Scales[n], zero_f32)
                    val = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val
    
    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[3])
def fp8_e5m2_w8a16_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """FP8 E5M2 W8A16 GEMM kernel: bf16 activation × FP8 E5M2 weight (uint8 storage, per-channel dequantized)."""
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        B: T.Tensor((N, K), T.float8_e5m2),
        Scales: T.Tensor((N,), T.float32),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f32 = tir.const(0.0, T.float32)
            zero_fp8 = tir.const(0, T.float8_e5m2)

            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.float8_e5m2)

            B_local = T.alloc_fragment((block_N, block_K), T.float8_e5m2)
            B_bf16_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            B_bf16_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_scaled = T.alloc_fragment((block_M, block_N), T.bfloat16)
            
            T.clear(C_local)
            
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.copy(B_shared, B_local)

                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)

                    T.copy(B_bf16_local, B_bf16_prev_local)
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_bf16)

                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else((n < N) & (kk < K), B[n, kk], zero_fp8)

                    T.copy(B_shared, B_local)

                    for i, j in T.Parallel(block_N, block_K):
                        B_bf16_local[i, j] = B_local[i, j].astype(T.float32).astype(T.bfloat16)

                    T.copy(B_bf16_local, B_bf16_prev_local)
                    T.gemm(A_shared, B_bf16_prev_local, C_local, transpose_B=True)
            
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    scale_f32 = Scales[bx * block_N + j]
                    C_scaled[i, j] = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                T.copy(C_scaled, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    scale_f32 = T.if_then_else(n < N, Scales[n], zero_f32)
                    val = (C_local[i, j] * scale_f32).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val
    
    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[4])
def fp8_e4m3_w8a8_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """FP8 E4M3 W8A8 GEMM kernel: FP8 E4M3 activation × FP8 E4M3 weight with fused scaling."""
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float8_e4m3fn),
        B: T.Tensor((N, K), T.float8_e4m3fn),
        XScales: T.Tensor((M,), T.float32),
        WScales: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_f32 = tir.const(0.0, T.float32)
            zero_f16 = tir.const(0, T.float16)
            zero_fp8 = tir.const(0, T.float8_e4m3fn)

            A_shared = T.alloc_shared((block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared((block_N, block_K), T.float8_e4m3fn)
            
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)
            
            T.clear(C_local)
            
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_fp8)

                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else((n < N) & (kk < K), B[n, kk], zero_fp8)

                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            # Fused scaling + store: C = (A@B^T) * x_scale[m] * w_scale[n]
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = XScales[m]
                    w_s = WScales[n].astype(T.float32)
                    C_out[i, j] = (C_local[i, j] * x_s * w_s).astype(T.bfloat16)
                T.copy(C_out, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = T.if_then_else(m < M, XScales[m], zero_f32)
                    w_s_f16 = T.if_then_else(n < N, WScales[n], zero_f16)
                    w_s = w_s_f16.astype(T.float32)
                    val = (C_local[i, j] * x_s * w_s).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val
    
    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[4])
def fp8_e5m2_w8a8_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """FP8 E5M2 W8A8 GEMM kernel: FP8 E5M2 activation × FP8 E5M2 weight with fused scaling."""
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float8_e5m2),
        B: T.Tensor((N, K), T.float8_e5m2),
        XScales: T.Tensor((M,), T.float32),
        WScales: T.Tensor((N,), T.float16),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_f32 = tir.const(0.0, T.float32)
            zero_f16 = tir.const(0, T.float16)
            zero_fp8 = tir.const(0, T.float8_e5m2)

            A_shared = T.alloc_shared((block_M, block_K), T.float8_e5m2)
            B_shared = T.alloc_shared((block_N, block_K), T.float8_e5m2)
            
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)
            
            T.clear(C_local)
            
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_fp8)

                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        B_shared[i, j] = T.if_then_else((n < N) & (kk < K), B[n, kk], zero_fp8)

                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = XScales[m]
                    w_s = WScales[n].astype(T.float32)
                    C_out[i, j] = (C_local[i, j] * x_s * w_s).astype(T.bfloat16)
                T.copy(C_out, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    x_s = T.if_then_else(m < M, XScales[m], zero_f32)
                    w_s_f16 = T.if_then_else(n < N, WScales[n], zero_f16)
                    w_s = w_s_f16.astype(T.float32)
                    val = (C_local[i, j] * x_s * w_s).astype(T.bfloat16)
                    if (m < M) & (n < N):
                        C[m, n] = val
    
    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[5])
def gptq_w4a16_gemm(
    M: int,
    N: int,
    K: int,
    num_groups: int,
    group_size: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """GPTQ W4A16 GEMM kernel: bf16 activation × GPTQ int4 weight (packed in int8, groupwise dequantized).
    
    Args:
        M: Number of rows in activation matrix A
        N: Number of output channels (rows in weight matrix B)
        K: Inner dimension (columns in A, rows in B)
        num_groups: Number of quantization groups
        group_size: Size of each group
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension
        num_stages: Number of pipeline stages
        threads: Number of threads per block
    
    Returns:
        Compiled TileLang kernel function with signature:
        kernel(A: bf16[M, K], QWeight: int8[N, (K+1)//2], QZeros: int8[num_groups, (K+1)//2], 
               Scales: float32[num_groups, K], GIdx: int32[N], C: bf16[M, N]) -> None
    """
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)
    packed_K = (K + 1) // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        QWeight: T.Tensor((N, packed_K), T.int8),
        QZeros: T.Tensor((num_groups, packed_K), T.int8),
        Scales: T.Tensor((num_groups, K), T.float32),
        GIdx: T.Tensor((N,), T.int32),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        """GPTQ W4A16 GEMM kernel implementation with groupwise dequantization."""
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f32 = tir.const(0, T.float32)
            
            # Constants for int4 unpacking
            int4_offset = tir.const(8, T.int8)
            mask_lower = tir.const(0x0F, T.int8)
            mask_upper_shift = tir.const(4, T.int8)

            # Allocate shared memory buffers
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            QWeight_shared = T.alloc_shared((block_N, (block_K + 1) // 2), T.int8)
            QZeros_shared = T.alloc_shared((num_groups, (block_K + 1) // 2), T.int8)
            
            # Allocate fragments
            QWeight_local = T.alloc_fragment((block_N, (block_K + 1) // 2), T.int8)
            QZeros_local = T.alloc_fragment((num_groups, (block_K + 1) // 2), T.int8)
            W_unpacked_local = T.alloc_fragment((block_N, block_K), T.int8)
            Z_unpacked_local = T.alloc_fragment((num_groups, block_K), T.int8)
            W_dequant_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            W_dequant_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            
            # Allocate fragment for accumulation
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            
            # Clear accumulation buffer
            T.clear(C_local)
            
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Load A tile
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    
                    # Load QWeight and QZeros tiles
                    packed_k_start = (k * block_K) // 2
                    T.copy(QWeight[bx * block_N, packed_k_start], QWeight_shared)
                    T.copy(QZeros[0:num_groups, packed_k_start], QZeros_shared)
                    
                    # Copy to local fragments
                    T.copy(QWeight_shared, QWeight_local)
                    T.copy(QZeros_shared, QZeros_local)
                    
                    # Unpack QWeight int4 -> int8
                    for i, j in T.Parallel(block_N, block_K):
                        j_packed = j // 2
                        packed_byte = QWeight_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        W_unpacked_local[i, j] = T.if_then_else(is_lower, lower_int4, upper_int4)
                    
                    # Unpack QZeros int4 -> int8
                    for g, j in T.Parallel(num_groups, block_K):
                        j_packed = j // 2
                        packed_byte = QZeros_local[g, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        Z_unpacked_local[g, j] = T.if_then_else(is_lower, lower_int4, upper_int4)
                    
                    # Dequantize weights: weight = quantized_int4 * scale + zero
                    # where zero = zero_quantized_int4 * scale
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        # Get group_id from GIdx, clamp to [0, num_groups-1]
                        group_id = GIdx[n]
                        group_id = T.if_then_else(group_id < 0, 0, group_id)
                        group_id = T.if_then_else(group_id >= num_groups, num_groups - 1, group_id)
                        
                        # Get scale and zero_quantized
                        scale = Scales[group_id, kk]
                        zero_quantized = Z_unpacked_local[group_id, j].astype(T.float32)
                        weight_quantized = W_unpacked_local[i, j].astype(T.float32)
                        
                        # Dequantize: weight = weight_quantized * scale + zero_quantized * scale
                        zero = zero_quantized * scale
                        weight_dequant = weight_quantized * scale + zero
                        W_dequant_local[i, j] = weight_dequant.astype(T.bfloat16)
                    
                    # Copy to prev_local for pipeline synchronization
                    T.copy(W_dequant_local, W_dequant_prev_local)
                    
                    # GEMM: C = A @ W_dequant^T
                    T.gemm(A_shared, W_dequant_prev_local, C_local, transpose_B=True)
            else:
                # Tail-safe kernel
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Masked load A
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_bf16)
                    
                    # Masked load QWeight
                    packed_k_start = (k * block_K) // 2
                    packed_k_size = (block_K + 1) // 2
                    for i, j_packed in T.Parallel(block_N, packed_k_size):
                        n = bx * block_N + i
                        packed_idx = packed_k_start + j_packed
                        QWeight_shared[i, j_packed] = T.if_then_else(
                            (n < N) & (packed_idx < packed_K),
                            QWeight[n, packed_idx],
                            zero_i8,
                        )
                    
                    # Masked load QZeros
                    for g, j_packed in T.Parallel(num_groups, packed_k_size):
                        packed_idx = packed_k_start + j_packed
                        QZeros_shared[g, j_packed] = T.if_then_else(
                            (g < num_groups) & (packed_idx < packed_K),
                            QZeros[g, packed_idx],
                            zero_i8,
                        )
                    
                    # Copy to local fragments
                    T.copy(QWeight_shared, QWeight_local)
                    T.copy(QZeros_shared, QZeros_local)
                    
                    # Unpack QWeight with boundary checks
                    for i, j in T.Parallel(block_N, block_K):
                        kk = k * block_K + j
                        j_packed = j // 2
                        packed_byte = QWeight_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        int4_val = T.if_then_else(is_lower, lower_int4, upper_int4)
                        in_bounds = (kk < K) & (j < block_K)
                        W_unpacked_local[i, j] = T.if_then_else(in_bounds, int4_val, zero_i8)
                    
                    # Unpack QZeros with boundary checks
                    for g, j in T.Parallel(num_groups, block_K):
                        kk = k * block_K + j
                        j_packed = j // 2
                        packed_byte = QZeros_local[g, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        int4_val = T.if_then_else(is_lower, lower_int4, upper_int4)
                        in_bounds = (kk < K) & (j < block_K) & (g < num_groups)
                        Z_unpacked_local[g, j] = T.if_then_else(in_bounds, int4_val, zero_i8)
                    
                    # Dequantize weights with boundary checks
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        in_bounds = (n < N) & (kk < K)
                        n = bx * block_N + i
                        kk = k * block_K + j
                        in_bounds = (n < N) & (kk < K)
                        in_bounds = (n < N) & (kk < K)

                        # Get group_id from GIdx, clamp to [0, num_groups-1]
                        group_id = GIdx[n]
                        group_id = T.if_then_else(group_id < 0, 0, group_id)
                        group_id = T.if_then_else(group_id >= num_groups, num_groups - 1, group_id)

                        # Get scale and zero_quantized (use safe values when out of bounds)
                        scale = T.if_then_else(in_bounds, Scales[group_id, kk], zero_f32)
                        zero_quantized = Z_unpacked_local[group_id, j].astype(T.float32)
                        weight_quantized = W_unpacked_local[i, j].astype(T.float32)

                        # Dequantize
                        zero = zero_quantized * scale
                        weight_dequant = weight_quantized * scale + zero
                        W_dequant_local[i, j] = T.if_then_else(
                            in_bounds,
                            weight_dequant.astype(T.bfloat16),
                            zero_bf16
                        )
                    
                    # Copy to prev_local
                    T.copy(W_dequant_local, W_dequant_prev_local)
                    
                    # GEMM
                    T.gemm(A_shared, W_dequant_prev_local, C_local, transpose_B=True)
            
            # Store output
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    C[m, n] = C_local[i, j].astype(T.bfloat16)
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    if (m < M) & (n < N):
                        C[m, n] = C_local[i, j].astype(T.bfloat16)
    
    return main


@tilelang.autotune(configs=build_linear_configs())
@tilelang.jit(out_idx=[4])
def awq_w4a16_gemm(
    M: int,
    N: int,
    K: int,
    num_groups: int,
    group_size: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
):
    """AWQ W4A16 GEMM kernel: bf16 activation × AWQ int4 weight (packed in int8, groupwise dequantized).
    
    Args:
        M: Number of rows in activation matrix A
        N: Number of output channels (rows in weight matrix B)
        K: Inner dimension (columns in A, rows in B)
        num_groups: Number of quantization groups
        group_size: Size of each group
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension
        num_stages: Number of pipeline stages
        threads: Number of threads per block
    
    Returns:
        Compiled TileLang kernel function with signature:
        kernel(A: bf16[M, K], QWeight: int8[N, (K+1)//2], QZeros: int8[num_groups, (K+1)//2], 
               Scales: float32[num_groups, K], C: bf16[M, N]) -> None
    """
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)
    packed_K = (K + 1) // 2

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),
        QWeight: T.Tensor((N, packed_K), T.int8),
        QZeros: T.Tensor((num_groups, packed_K), T.int8),
        Scales: T.Tensor((num_groups, K), T.float32),
        C: T.Tensor((M, N), T.bfloat16),
    ):
        """AWQ W4A16 GEMM kernel implementation with groupwise dequantization (sequential grouping)."""
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            zero_i8 = tir.const(0, T.int8)
            zero_bf16 = tir.const(0, T.bfloat16)
            zero_f32 = tir.const(0, T.float32)
            
            # Constants for int4 unpacking
            int4_offset = tir.const(8, T.int8)
            mask_lower = tir.const(0x0F, T.int8)
            mask_upper_shift = tir.const(4, T.int8)

            # Allocate shared memory buffers
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            QWeight_shared = T.alloc_shared((block_N, (block_K + 1) // 2), T.int8)
            QZeros_shared = T.alloc_shared((num_groups, (block_K + 1) // 2), T.int8)
            
            # Allocate fragments
            QWeight_local = T.alloc_fragment((block_N, (block_K + 1) // 2), T.int8)
            QZeros_local = T.alloc_fragment((num_groups, (block_K + 1) // 2), T.int8)
            W_unpacked_local = T.alloc_fragment((block_N, block_K), T.int8)
            Z_unpacked_local = T.alloc_fragment((num_groups, block_K), T.int8)
            W_dequant_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            W_dequant_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            
            # Allocate fragment for accumulation
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            
            # Clear accumulation buffer
            T.clear(C_local)
            
            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    # Load A tile
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    
                    # Load QWeight and QZeros tiles
                    packed_k_start = (k * block_K) // 2
                    T.copy(QWeight[bx * block_N, packed_k_start], QWeight_shared)
                    T.copy(QZeros[0:num_groups, packed_k_start], QZeros_shared)
                    
                    # Copy to local fragments
                    T.copy(QWeight_shared, QWeight_local)
                    T.copy(QZeros_shared, QZeros_local)
                    
                    # Unpack QWeight int4 -> int8
                    for i, j in T.Parallel(block_N, block_K):
                        j_packed = j // 2
                        packed_byte = QWeight_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        W_unpacked_local[i, j] = T.if_then_else(is_lower, lower_int4, upper_int4)
                    
                    # Unpack QZeros int4 -> int8
                    for g, j in T.Parallel(num_groups, block_K):
                        j_packed = j // 2
                        packed_byte = QZeros_local[g, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        Z_unpacked_local[g, j] = T.if_then_else(is_lower, lower_int4, upper_int4)
                    
                    # Dequantize weights: weight = quantized_int4 * scale + zero
                    # where zero = zero_quantized_int4 * scale
                    # AWQ uses sequential grouping: group_id = n // group_size
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        # Compute group_id using sequential grouping
                        group_id = n // group_size
                        # Clamp to [0, num_groups-1]
                        group_id = T.if_then_else(group_id < 0, 0, group_id)
                        group_id = T.if_then_else(group_id >= num_groups, num_groups - 1, group_id)
                        
                        # Get scale and zero_quantized
                        scale = Scales[group_id, kk]
                        zero_quantized = Z_unpacked_local[group_id, j].astype(T.float32)
                        weight_quantized = W_unpacked_local[i, j].astype(T.float32)
                        
                        # Dequantize: weight = weight_quantized * scale + zero_quantized * scale
                        zero = zero_quantized * scale
                        weight_dequant = weight_quantized * scale + zero
                        W_dequant_local[i, j] = weight_dequant.astype(T.bfloat16)
                    
                    # Copy to prev_local for pipeline synchronization
                    T.copy(W_dequant_local, W_dequant_prev_local)
                    
                    # GEMM: C = A @ W_dequant^T
                    T.gemm(A_shared, W_dequant_prev_local, C_local, transpose_B=True)
            else:
                # Tail-safe kernel
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    # Masked load A
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_bf16)
                    
                    # Masked load QWeight
                    packed_k_start = (k * block_K) // 2
                    packed_k_size = (block_K + 1) // 2
                    for i, j_packed in T.Parallel(block_N, packed_k_size):
                        n = bx * block_N + i
                        packed_idx = packed_k_start + j_packed
                        QWeight_shared[i, j_packed] = T.if_then_else(
                            (n < N) & (packed_idx < packed_K),
                            QWeight[n, packed_idx],
                            zero_i8,
                        )
                    
                    # Masked load QZeros
                    for g, j_packed in T.Parallel(num_groups, packed_k_size):
                        packed_idx = packed_k_start + j_packed
                        QZeros_shared[g, j_packed] = T.if_then_else(
                            (g < num_groups) & (packed_idx < packed_K),
                            QZeros[g, packed_idx],
                            zero_i8,
                        )
                    
                    # Copy to local fragments
                    T.copy(QWeight_shared, QWeight_local)
                    T.copy(QZeros_shared, QZeros_local)
                    
                    # Unpack QWeight with boundary checks
                    for i, j in T.Parallel(block_N, block_K):
                        kk = k * block_K + j
                        j_packed = j // 2
                        packed_byte = QWeight_local[i, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        int4_val = T.if_then_else(is_lower, lower_int4, upper_int4)
                        in_bounds = (kk < K) & (j < block_K)
                        W_unpacked_local[i, j] = T.if_then_else(in_bounds, int4_val, zero_i8)
                    
                    # Unpack QZeros with boundary checks
                    for g, j in T.Parallel(num_groups, block_K):
                        kk = k * block_K + j
                        j_packed = j // 2
                        packed_byte = QZeros_local[g, j_packed]
                        lower_uint = (packed_byte & mask_lower).astype(T.int8)
                        upper_uint = ((packed_byte >> mask_upper_shift) & mask_lower).astype(T.int8)
                        lower_int4 = lower_uint - int4_offset
                        upper_int4 = upper_uint - int4_offset
                        is_lower = (j % 2) == 0
                        int4_val = T.if_then_else(is_lower, lower_int4, upper_int4)
                        in_bounds = (kk < K) & (j < block_K) & (g < num_groups)
                        Z_unpacked_local[g, j] = T.if_then_else(in_bounds, int4_val, zero_i8)
                    
                    # Dequantize weights with boundary checks
                    # AWQ uses sequential grouping: group_id = n // group_size
                    for i, j in T.Parallel(block_N, block_K):
                        n = bx * block_N + i
                        kk = k * block_K + j
                        in_bounds = (n < N) & (kk < K)
                        # Compute group_id using sequential grouping
                        group_id = n // group_size
                        # Clamp to [0, num_groups-1]
                        group_id = T.if_then_else(group_id < 0, 0, group_id)
                        group_id = T.if_then_else(group_id >= num_groups, num_groups - 1, group_id)
                        
                        # Get scale and zero_quantized
                        scale = T.if_then_else(in_bounds, Scales[group_id, kk], zero_f32)
                        zero_quantized = Z_unpacked_local[group_id, j].astype(T.float32)
                        weight_quantized = W_unpacked_local[i, j].astype(T.float32)
                        
                        # Dequantize
                        zero = zero_quantized * scale
                        weight_dequant = weight_quantized * scale + zero
                        W_dequant_local[i, j] = T.if_then_else(
                            in_bounds,
                            weight_dequant.astype(T.bfloat16),
                            zero_bf16
                        )
                    
                    # Copy to prev_local
                    T.copy(W_dequant_local, W_dequant_prev_local)
                    
                    # GEMM
                    T.gemm(A_shared, W_dequant_prev_local, C_local, transpose_B=True)
            
            # Store output
            if aligned:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    C[m, n] = C_local[i, j].astype(T.bfloat16)
            else:
                for i, j in T.Parallel(block_M, block_N):
                    m = by * block_M + i
                    n = bx * block_N + j
                    if (m < M) & (n < N):
                        C[m, n] = C_local[i, j].astype(T.bfloat16)
    
    return main
