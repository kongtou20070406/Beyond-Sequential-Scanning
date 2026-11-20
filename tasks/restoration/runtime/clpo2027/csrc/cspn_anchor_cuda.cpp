#include <torch/extension.h>
#include <vector>

torch::Tensor cspn_anchor_forward(
    torch::Tensor initial, torch::Tensor center,
    torch::Tensor off, int64_t steps);

std::vector<torch::Tensor> cspn_anchor_backward(
    torch::Tensor grad_out, torch::Tensor states,
    torch::Tensor center, torch::Tensor off);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cspn_anchor_forward, "Anchored CSPN forward (CUDA)");
  m.def("backward", &cspn_anchor_backward, "Anchored CSPN backward (CUDA)");
}
