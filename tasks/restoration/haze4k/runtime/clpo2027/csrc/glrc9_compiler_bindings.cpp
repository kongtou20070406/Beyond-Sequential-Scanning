#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> glrc9_compiler_forward_cuda(
    torch::Tensor packed,
    torch::Tensor m_real,
    torch::Tensor m_imag,
    torch::Tensor n_real,
    torch::Tensor n_imag,
    torch::Tensor gain_real,
    torch::Tensor gain_imag,
    torch::Tensor basis_y,
    torch::Tensor basis_x,
    torch::Tensor head_weight,
    torch::Tensor head_bias);
std::vector<torch::Tensor> glrc9_q9_project_cuda(torch::Tensor raw, double rho_max);

std::vector<torch::Tensor> glrc9_compiler_forward(
    torch::Tensor packed,
    torch::Tensor m_real,
    torch::Tensor m_imag,
    torch::Tensor n_real,
    torch::Tensor n_imag,
    torch::Tensor gain_real,
    torch::Tensor gain_imag,
    torch::Tensor basis_y,
    torch::Tensor basis_x,
    torch::Tensor head_weight,
    torch::Tensor head_bias) {
  TORCH_CHECK(packed.is_cuda(), "GLRC9 compiler requires CUDA tensors");
  return glrc9_compiler_forward_cuda(packed, m_real, m_imag, n_real, n_imag, gain_real, gain_imag, basis_y, basis_x, head_weight, head_bias);
}

std::vector<torch::Tensor> glrc9_q9_project(torch::Tensor raw, double rho_max) {
  TORCH_CHECK(raw.is_cuda(), "GLRC9 projection requires a CUDA tensor");
  return glrc9_q9_project_cuda(raw, rho_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &glrc9_compiler_forward, "GLRC9 direct low-3 compiler (CUDA)");
  m.def("project", &glrc9_q9_project, "GLRC9 q9 legalizer (CUDA)");
}
