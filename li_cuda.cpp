#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor li_cuda_forward_step(torch::Tensor i, torch::Tensor v,
                                   torch::Tensor leak, torch::Tensor tau_inv,
                                   float dt);

torch::Tensor li_cuda_forward_integral(torch::Tensor i, torch::Tensor v,
                                       torch::Tensor leak,
                                       torch::Tensor tau_inv, float dt);

// std::vector<torch::Tensor> li_cuda_backward(
//     torch::Tensor grad_h,
//     torch::Tensor grad_cell,
//     torch::Tensor new_cell,
//     torch::Tensor input_gate,
//     torch::Tensor output_gate,
//     torch::Tensor candidate_cell,
//     torch::Tensor X,
//     torch::Tensor gate_weights,
//     torch::Tensor weights);

// C++ interface

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor li_step_forward(torch::Tensor i, torch::Tensor v,
                              torch::Tensor leak, torch::Tensor tau_mem,
                              const float dt) {
  CHECK_INPUT(i);
  CHECK_INPUT(v);
  CHECK_INPUT(leak);
  CHECK_INPUT(tau_mem);

  return li_cuda_forward_step(i, v, leak, tau_mem, dt);
}

std::vector<torch::Tensor> li_integral_forward(torch::Tensor i, torch::Tensor v,
                                               torch::Tensor leak,
                                               torch::Tensor tau_mem,
                                               const float dt) {
  CHECK_INPUT(i);
  CHECK_INPUT(v);
  CHECK_INPUT(leak);
  CHECK_INPUT(tau_mem);

  const auto new_v = li_cuda_forward_integral(i, v, leak, tau_mem, dt);
  return {new_v, new_v.index({-1})};
}

// std::vector<torch::Tensor> lltm_backward(
//     torch::Tensor v,
//     torch::Tensor i) {
//   CHECK_INPUT(v);
//   CHECK_INPUT(i);

//   return lltm_cuda_backward(
//       grad_h,
//       grad_cell,
//       new_cell,
//       input_gate,
//       output_gate,
//       candidate_cell,
//       X,
//       gate_weights,
//       weights);
// }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step_forward", &li_step_forward, "LI integral forward (CUDA)");
  m.def("integral_forward", &li_integral_forward, "LI integral forward (CUDA)");
  // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}