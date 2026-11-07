#include <torch/extension.h>

#include <c10/util/Optional.h>

torch::Tensor mixnlq_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k,
    int64_t kernel_variant);

torch::Tensor mixnlq3_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& linear_scale,
    const torch::Tensor& exp_scale,
    const torch::Tensor& quant_bias,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k);

torch::Tensor linear3_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k);

torch::Tensor lut3_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k);

torch::Tensor linear_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& scale,
    const torch::Tensor& zero_point,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k);

torch::Tensor lut_decode_cuda(
    const torch::Tensor& x,
    const torch::Tensor& packed_weight,
    const torch::Tensor& group_lut,
    const c10::optional<torch::Tensor>& bias,
    int64_t group_size,
    int64_t requested_split_k);

TORCH_LIBRARY(mixnlq_cuda, library) {
  library.def(
      "decode(Tensor x, Tensor packed_weight, Tensor linear_scale, "
      "Tensor exp_scale, Tensor? bias, int group_size, int split_k=0, "
      "int kernel_variant=0) -> Tensor");
  library.def(
      "decode3(Tensor x, Tensor packed_weight, Tensor linear_scale, "
      "Tensor exp_scale, Tensor quant_bias, Tensor? bias, int group_size, "
      "int split_k=0) -> Tensor");
  library.def(
      "linear_decode3(Tensor x, Tensor packed_weight, Tensor scale, "
      "Tensor zero_point, Tensor? bias, int group_size, int split_k=0) "
      "-> Tensor");
  library.def(
      "lut_decode3(Tensor x, Tensor packed_weight, Tensor group_lut, "
      "Tensor? bias, int group_size, int split_k=0) -> Tensor");
  library.def(
      "linear_decode(Tensor x, Tensor packed_weight, Tensor scale, "
      "Tensor zero_point, Tensor? bias, int group_size, int split_k=0) "
      "-> Tensor");
  library.def(
      "lut_decode(Tensor x, Tensor packed_weight, Tensor group_lut, "
      "Tensor? bias, int group_size, int split_k=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(mixnlq_cuda, CUDA, library) {
  library.impl("decode", &mixnlq_decode_cuda);
  library.impl("decode3", &mixnlq3_decode_cuda);
  library.impl("linear_decode3", &linear3_decode_cuda);
  library.impl("lut_decode3", &lut3_decode_cuda);
  library.impl("linear_decode", &linear_decode_cuda);
  library.impl("lut_decode", &lut_decode_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.attr("MAX_SPLIT_K") = 64;
}
