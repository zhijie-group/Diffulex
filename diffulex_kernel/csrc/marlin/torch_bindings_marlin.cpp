#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declarations implemented in .cu files.
torch::Tensor allspark_w8a16_gemm(
    torch::Tensor const& a, torch::Tensor const& b_qweight,
    torch::Tensor const& b_scales, c10::optional<torch::Tensor> const& b_qzeros,
    int64_t n, int64_t group_size, int64_t sm_count, int64_t sm_version,
    int64_t CUBLAS_M_THRESHOLD, bool has_zp, bool n32k16_reorder);

void rearrange_kn_weight_as_n32k16_order(
    torch::Tensor const& b_qweight, torch::Tensor const& b_scales,
    c10::optional<torch::Tensor> const& b_zeros, bool has_zp,
    torch::Tensor& b_qweight_reorder, torch::Tensor& b_scales_reorder,
    c10::optional<torch::Tensor> const& b_zeros_reorder, int64_t K, int64_t N,
    int64_t N_32align);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("allspark_w8a16_gemm", &allspark_w8a16_gemm,
        "AllSpark W8A16 fused GEMM (uint8 weight bias128 + bf16/fp16 act)");
  m.def("rearrange_kn_weight_as_n32k16_order",
        &rearrange_kn_weight_as_n32k16_order,
        "Repack (K,N) uint8 weight into N32K16 order + reorder/pad scales");
}

