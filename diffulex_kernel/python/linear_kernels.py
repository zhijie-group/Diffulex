"""
W8A16, W4A16, W8A8, and W4A8 Linear GEMM kernels using TileLang.

- W8A16: int8 weight × bf16 activation matrix multiplication with per-channel dequantization.
- W4A16: int4 weight (packed in int8) × bf16 activation matrix multiplication with per-channel dequantization.
- W8A8: int8 activation × int8 weight matrix multiplication, output int32 accumulator.
- W4A8: int8 activation × int4 weight (packed in int8) matrix multiplication, output int32 accumulator.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
from tvm import tir


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
        N: Number of output channels (columns in weight matrix B)
        K: Inner dimension (columns in A, rows in B)
        block_M: Block size for M dimension
        block_N: Block size for N dimension
        block_K: Block size for K dimension
        num_stages: Number of pipeline stages
        threads: Number of threads per block
    
    Returns:
        Compiled TileLang kernel function with signature:
        kernel(A: int8[M, K], B: int8[K, N], C: int32[M, N]) -> None
        
    Note:
        - Input A is int8 quantized activation [M, K]
        - Input B is int8 quantized weight (transposed) [K, N]
        - Output C is int32 accumulator [M, N]
        - Scales (activation scales and weight scales) are applied externally after this kernel
    """
    # Fast path: only generate the simple copy-based kernel when all dims are perfectly tiled.
    # Otherwise, generate a masked (tail-safe) kernel to avoid falling back for non-multiple sizes.
    aligned = (M % block_M == 0) and (N % block_N == 0) and (K % block_K == 0)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int8),           # quantized activation, shape (M, K)
        B: T.Tensor((K, N), T.int8),           # quantized weight (transposed), shape (K, N)
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
            B_shared = T.alloc_shared((block_K, block_N), T.int8)
            
            # Allocate fragments for pipelining
            A_local = T.alloc_fragment((block_M, block_K), T.int8)
            B_local = T.alloc_fragment((block_K, block_N), T.int8)
            A_local_prev = T.alloc_fragment((block_M, block_K), T.int8)
            B_local_prev = T.alloc_fragment((block_K, block_N), T.int8)
            
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
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                
                    # Copy to local fragments (required for proper pipelining)
                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)
                
                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)
                
                    # GEMM: C = A @ B (int8 x int8 -> int32 accumulation).
                    # Important: use int8 operands; TileLang lowers to the appropriate int8 GEMM path.
                    T.gemm(A_local_prev, B_local_prev, C_local)
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
                    for i, j in T.Parallel(block_K, block_N):
                        kk = k * block_K + i
                        n = bx * block_N + j
                        B_shared[i, j] = T.if_then_else(
                            (kk < K) & (n < N),
                            B[kk, n],
                            zero_i8,
                        )

                    # Copy to local fragments
                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)

                    # Copy to prev_local (required for pipeline synchronization)
                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)

                    # GEMM (padded with zeros for out-of-range A/B)
                    T.gemm(A_local_prev, B_local_prev, C_local)
            
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
        B: T.Tensor((K, N), T.int8),
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
            B_shared = T.alloc_shared((block_K, block_N), T.int8)

            A_local = T.alloc_fragment((block_M, block_K), T.int8)
            B_local = T.alloc_fragment((block_K, block_N), T.int8)
            A_local_prev = T.alloc_fragment((block_M, block_K), T.int8)
            B_local_prev = T.alloc_fragment((block_K, block_N), T.int8)

            C_local = T.alloc_fragment((block_M, block_N), T.int32)
            C_out = T.alloc_fragment((block_M, block_N), T.bfloat16)

            T.clear(C_local)

            if aligned:
                num_k_blocks = K // block_K
                for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, bx * block_N], B_shared)

                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)

                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)

                    # int8 x int8 -> int32 accumulation
                    T.gemm(A_local_prev, B_local_prev, C_local)
            else:
                for k in T.Pipelined(0, T.ceildiv(K, block_K), num_stages=num_stages):
                    for i, j in T.Parallel(block_M, block_K):
                        m = by * block_M + i
                        kk = k * block_K + j
                        A_shared[i, j] = T.if_then_else((m < M) & (kk < K), A[m, kk], zero_i8)

                    for i, j in T.Parallel(block_K, block_N):
                        kk = k * block_K + i
                        n = bx * block_N + j
                        B_shared[i, j] = T.if_then_else((kk < K) & (n < N), B[kk, n], zero_i8)

                    T.copy(A_shared, A_local)
                    T.copy(B_shared, B_local)

                    T.copy(A_local, A_local_prev)
                    T.copy(B_local, B_local_prev)

                    T.gemm(A_local_prev, B_local_prev, C_local)

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
