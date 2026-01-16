#pragma once

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>

// Minimal scalar conversion helpers (avoid vendoring vLLM marlin/core headers).
namespace diffulex_allspark {
template <typename T>
struct ScalarConvert;

template <>
struct ScalarConvert<half> {
  static __device__ __forceinline__ float num2float(const half x) {
    return __half2float(x);
  }
  static __host__ __device__ __forceinline__ half float2num(const float x) {
    return __float2half(x);
  }
};

template <>
struct ScalarConvert<nv_bfloat16> {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  static __device__ __forceinline__ float num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }
  static __host__ __device__ __forceinline__ nv_bfloat16 float2num(const float x) {
    return __float2bfloat16(x);
  }
#else
  static __device__ __forceinline__ float num2float(const nv_bfloat16) { return 0.f; }
  static __host__ __device__ __forceinline__ nv_bfloat16 float2num(const float) { return nv_bfloat16(); }
#endif
};
}  // namespace diffulex_allspark

namespace allspark {

#define CHECK_CUDA(cmd)                                             \
  do {                                                              \
    cudaError_t cuda_status = cmd;                                  \
    if (cuda_status != cudaSuccess) {                               \
      std::string err_str = cudaGetErrorString(cuda_status);        \
      std::cerr << "Failed: " << __FILE__ << ":" << __LINE__ << " " \
                << err_str;                                         \
      exit(-1);                                                     \
    }                                                               \
  } while (0)

#define CHECK_CUBLAS(cmd)                                            \
  do {                                                               \
    cublasStatus_t cublas_status = cmd;                              \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {                    \
      std::cerr << "Failed:  " << __FILE__ << ":" << __LINE__ << " " \
                << cublas_status << std::endl;                       \
      exit(-1);                                                      \
    }                                                                \
  } while (0)

template <typename FType, typename QType>
struct SM8x_GEMM_W8A16_Splitk_Params {
  const FType* A_ptr;
  const QType* B_ptr;
  const FType* B_scale_ptr;
  const FType* B_zero_ptr;
  FType* C_ptr;
  int M;
  int N;
  int K;
  int SplitK;
  int GroupCnt;
  int GroupSize;
  FType* C_split_ptr;       // for non-fused splitk reduce
  float* C_tmp_ptr;         // for fused splitk reduce
  uint32_t* red_count_ptr;  // for fused splitk reduce
};

struct alignas(16) BlockTileSplitkParams {
  int Mtile;
  int Ntile;
  int SplitK;
  bool EnableFuse;
};

// ---- the rest is copied from vLLM (gptq_allspark/allspark_utils.cuh) ----
// We keep it verbatim to preserve kernel correctness/perf.

__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(const void* const ptr) {
  uint32_t smem_ptr;
  asm("cvta.to.shared.u32 %0, %1;" : "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_group(int n) {
  asm volatile("cp.async.wait_group %0;" ::"n"(n));
}

template <int SizeInBytes>
__device__ __forceinline__ void cp_async(uint32_t smem_addr, const void* gmem_ptr,
                                         int src_size, bool pred_guard = true) {
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], %2, %3, %4;\n" ::"r"(smem_addr),
      "l"(gmem_ptr), "n"(SizeInBytes), "r"(src_size), "r"((int)pred_guard));
}

__device__ __forceinline__ void ldg128_cg_0(uint32_t& r0, uint32_t& r1,
                                           uint32_t& r2, uint32_t& r3,
                                           const void* ptr, bool guard = true) {
  if (guard) {
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(ptr));
  } else {
    r0 = r1 = r2 = r3 = 0;
  }
}

template <typename T>
__device__ __forceinline__ void ldg16_cg_0(T& r0, const void* ptr, bool guard = true) {
  if (guard) {
    asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(reinterpret_cast<uint16_t&>(r0)) : "l"(ptr));
  } else {
    reinterpret_cast<uint16_t&>(r0) = 0;
  }
}

__device__ __forceinline__ void ldg64_ca(uint32_t& r0, uint32_t& r1, const void* ptr,
                                        bool guard = true) {
  if (guard) {
    asm volatile("ld.global.ca.v2.u32 {%0, %1}, [%2];" : "=r"(r0), "=r"(r1) : "l"(ptr));
  } else {
    r0 = r1 = 0;
  }
}

__device__ __forceinline__ void lds128(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                      uint32_t& r3, uint32_t smem_addr) {
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_addr));
}

__device__ __forceinline__ void ldsm_4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                      uint32_t& r3, uint32_t smem_addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_addr));
}

__device__ __forceinline__ void cvt_8bx4_to_16bx4_bias128(const uint32_t& src, uint32_t* dst) {
  asm volatile(
      "prmt.b32 %0, %4, 0x80, 0x4440;\n"
      "prmt.b32 %1, %4, 0x80, 0x4441;\n"
      "prmt.b32 %2, %4, 0x80, 0x4442;\n"
      "prmt.b32 %3, %4, 0x80, 0x4443;\n"
      : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
      : "r"(src));
}

template <typename FType>
__device__ __forceinline__ void hmma16816_f32(float* d, const uint32_t* a, const uint32_t* b) {
  if constexpr (std::is_same<FType, half>::value) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
}

template <typename FType, int BLOCK, int N_MATRIX>
__global__ void f16_gemm_splitk_reduce_kernel(const FType* C_split, FType* C,
                                              uint32_t n, uint32_t n_matrix,
                                              uint32_t matrix_size) {
  auto idx = blockIdx.x * BLOCK + threadIdx.x;

  if (idx >= matrix_size) {
    return;
  }

  float sum = 0.f;

  int n_mat = N_MATRIX > 0 ? N_MATRIX : (int)n_matrix;
  for (int i = 0; i < n_mat; ++i) {
    sum += diffulex_allspark::ScalarConvert<FType>::num2float(C_split[idx + i * matrix_size]);
  }

  C[idx] = diffulex_allspark::ScalarConvert<FType>::float2num(sum);
}

template <typename FType>
void f16_gemm_splitk_reduce(const FType* C_split, FType* C, const uint32_t m,
                            const uint32_t n, const uint32_t n_matrix,
                            cudaStream_t stream) {
  const int BLOCK = 128;
  uint32_t matrix_size = m * n;
  int grid = (matrix_size + BLOCK - 1) / BLOCK;

  void (*kernel)(const FType*, FType*, uint32_t, uint32_t, uint32_t) = nullptr;

  switch (n_matrix) {
    case 4:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 4>;
      break;
    case 5:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 5>;
      break;
    case 6:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 6>;
      break;
    case 7:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 7>;
      break;
    case 8:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, 8>;
      break;
    default:
      kernel = f16_gemm_splitk_reduce_kernel<FType, BLOCK, -1>;
      break;
  }

  kernel<<<grid, BLOCK, 0, stream>>>(C_split, C, n, n_matrix, matrix_size);
}

}  // namespace allspark

