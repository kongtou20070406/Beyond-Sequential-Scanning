#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> glrc9_f9_step_forward_cuda(
    torch::Tensor packed_f, torch::Tensor ar, torch::Tensor ai, int64_t steps);
std::vector<torch::Tensor> glrc9_f9_step_adjoint_cuda(
    torch::Tensor grad_endpoint, torch::Tensor ar, torch::Tensor ai,
    torch::Tensor trajectory, int64_t steps);

std::vector<torch::Tensor> glrc9_f9_step_forward(
    torch::Tensor packed_f, torch::Tensor ar, torch::Tensor ai, int64_t steps) {
  TORCH_CHECK(packed_f.is_cuda(), "GLRC9 step F9 extension was built for CUDA tensors");
  return glrc9_f9_step_forward_cuda(packed_f, ar, ai, steps);
}

std::vector<torch::Tensor> glrc9_f9_step_adjoint(
    torch::Tensor grad_endpoint, torch::Tensor ar, torch::Tensor ai,
    torch::Tensor trajectory, int64_t steps) {
  TORCH_CHECK(grad_endpoint.is_cuda(), "GLRC9 step F9 extension was built for CUDA tensors");
  return glrc9_f9_step_adjoint_cuda(grad_endpoint, ar, ai, trajectory, steps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &glrc9_f9_step_forward, "GLRC9 step-wise F9 forward (CUDA)");
  m.def("adjoint", &glrc9_f9_step_adjoint, "GLRC9 step-wise F9 adjoint (CUDA)");
}
