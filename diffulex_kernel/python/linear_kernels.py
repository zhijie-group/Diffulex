"""
W8A16 Linear GEMM kernel using TileLang.

Implements int8 weight × bf16 activation matrix multiplication with per-channel dequantization.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


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
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.bfloat16),           # activation, shape (M, K)
        B: T.Tensor((N, K), T.int8),              # quantized weight, shape (N, K)
        Scales: T.Tensor((N,), T.bfloat16),       # per-channel scales, shape (N,)
        C: T.Tensor((M, N), T.bfloat16),          # output, shape (M, N)
    ):
        """W8A16 GEMM kernel implementation.
        
        Computes C = A @ B_dequant^T where B_dequant[i, j] = B[i, j] * Scales[i]
        
        This implementation follows the W4A8 pattern with fragments for proper pipelining.
        """
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # Allocate shared memory buffers
            A_shared = T.alloc_shared((block_M, block_K), T.bfloat16)
            B_shared = T.alloc_shared((block_N, block_K), T.int8)
            
            # Allocate fragments (matching W4A8 pattern for proper pipelining)
            B_local = T.alloc_fragment((block_N, block_K), T.int8)
            B_dequantize_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            B_dequantize_prev_local = T.alloc_fragment((block_N, block_K), T.bfloat16)
            
            # Allocate fragment for accumulation (use float32 for precision)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            
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
            num_k_blocks = K // block_K
            for k in T.Pipelined(num_k_blocks, num_stages=num_stages):
                # Load A and B tiles to shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                
                # Copy B_shared to local fragment (required for proper pipelining)
                T.copy(B_shared, B_local)
                
                # Per-channel dequantization: B_dequant[i, j] = B[i, j] * Scales[i]
                # Note: Scales[bx * block_N + i] accesses the correct scale for output channel i
                for i, j in T.Parallel(block_N, block_K):
                    # Convert int8 -> float32, multiply by scale, convert to bf16
                    B_dequantize_local[i, j] = (
                        B_local[i, j].astype(T.float32) * Scales[bx * block_N + i]
                    ).astype(T.bfloat16)
                
                # Copy dequantized local to prev_local (required for pipeline synchronization)
                T.copy(B_dequantize_local, B_dequantize_prev_local)
                
                # GEMM: C = A @ B_dequant^T
                # Note: B_dequantize_prev_local is (block_N, block_K), transpose_B=True computes A @ B^T
                T.gemm(A_shared, B_dequantize_prev_local, C_local, transpose_B=True)
            
            # Store result from local fragment to global memory
            T.copy(C_local, C[by * block_M : (by + 1) * block_M, bx * block_N : (bx + 1) * block_N])
    
    return main
