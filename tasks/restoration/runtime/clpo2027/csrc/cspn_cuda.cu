#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

template <typename scalar_t>
__global__ void cspn_forward_kernel(
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ center,
    const scalar_t* __restrict__ off,
    scalar_t* __restrict__ out,
    int planes, int height, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int hw = height * width;
  const int total = planes * hw;
  if (idx >= total) return;
  const int p = idx % hw;
  const int y = p / width;
  const int x = p - y * width;
  const int plane = idx / hw;
  float value = static_cast<float>(center[idx]) * static_cast<float>(state[idx]);
  constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int ny = y + dy[i];
    const int nx = x + dx[i];
    if ((unsigned)ny < (unsigned)height && (unsigned)nx < (unsigned)width) {
      const int nidx = plane * hw + ny * width + nx;
      const int oidx = (plane * 8 + i) * hw + p;
      value += static_cast<float>(off[oidx]) * static_cast<float>(state[nidx]);
    }
  }
  out[idx] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
__global__ void cspn_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ center,
    const scalar_t* __restrict__ off,
    scalar_t* __restrict__ grad_state,
    scalar_t* __restrict__ grad_center,
    scalar_t* __restrict__ grad_off,
    int planes, int height, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int hw = height * width;
  const int total = planes * hw;
  if (idx >= total) return;
  const int q = idx % hw;
  const int y = q / width;
  const int x = q - y * width;
  const int plane = idx / hw;
  const float go = static_cast<float>(grad_out[idx]);
  float gx = static_cast<float>(center[idx]) * go;
  float gc = static_cast<float>(grad_center[idx]);
  gc += go * static_cast<float>(state[idx]);
  grad_center[idx] = static_cast<scalar_t>(gc);
  constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int py = y - dy[i];
    const int px = x - dx[i];
    if ((unsigned)py < (unsigned)height && (unsigned)px < (unsigned)width) {
      const int p = py * width + px;
      const int pidx = plane * hw + p;
      const int oidx = (plane * 8 + i) * hw + p;
      gx += static_cast<float>(off[oidx]) * static_cast<float>(grad_out[pidx]);
    }
    const int ny = y + dy[i];
    const int nx = x + dx[i];
    const int goidx = (plane * 8 + i) * hw + q;
    float ga = static_cast<float>(grad_off[goidx]);
    if ((unsigned)ny < (unsigned)height && (unsigned)nx < (unsigned)width) {
      ga += go * static_cast<float>(state[plane * hw + ny * width + nx]);
    }
    grad_off[goidx] = static_cast<scalar_t>(ga);
  }
  grad_state[idx] = static_cast<scalar_t>(gx);
}

void check_inputs(const torch::Tensor& state, const torch::Tensor& center,
                  const torch::Tensor& off) {
  TORCH_CHECK(state.is_cuda() && center.is_cuda() && off.is_cuda(), "CSPN CUDA tensors required");
  TORCH_CHECK(state.is_contiguous() && center.is_contiguous() && off.is_contiguous(), "CSPN tensors must be contiguous");
  TORCH_CHECK(state.sizes() == center.sizes(), "state/center shape mismatch");
  TORCH_CHECK(off.dim() == 5 && off.size(2) == 8, "off must be BC8HW");
  TORCH_CHECK(state.scalar_type() == center.scalar_type() && state.scalar_type() == off.scalar_type(), "CSPN dtype mismatch");
}

}  // namespace

torch::Tensor cspn_forward(torch::Tensor state, torch::Tensor center,
                           torch::Tensor off, int64_t steps) {
  check_inputs(state, center, off);
  TORCH_CHECK(steps > 0, "steps must be positive");
  c10::cuda::CUDAGuard guard(state.device());
  const int64_t planes = state.size(0) * state.size(1);
  const int height = state.size(2);
  const int width = state.size(3);
  auto states = torch::empty({steps + 1, state.size(0), state.size(1), height, width}, state.options());
  states[0].copy_(state);
  const int total = static_cast<int>(planes * height * width);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getDefaultCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, state.scalar_type(), "cspn_forward_cuda", [&] {
    for (int64_t t = 0; t < steps; ++t) {
      cspn_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          states[t].data_ptr<scalar_t>(), center.data_ptr<scalar_t>(),
          off.data_ptr<scalar_t>(), states[t + 1].data_ptr<scalar_t>(),
          static_cast<int>(planes), height, width);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return states;
}

std::vector<torch::Tensor> cspn_backward(torch::Tensor grad_out,
                                         torch::Tensor states,
                                         torch::Tensor center,
                                         torch::Tensor off) {
  TORCH_CHECK(states.dim() == 5 && states.size(0) >= 2, "invalid saved states");
  auto state0 = states[0];
  check_inputs(state0, center, off);
  grad_out = grad_out.contiguous();
  c10::cuda::CUDAGuard guard(state0.device());
  const int64_t planes = state0.size(0) * state0.size(1);
  const int height = state0.size(2);
  const int width = state0.size(3);
  auto grad_state = grad_out;
  auto grad_center = torch::zeros_like(center);
  auto grad_off = torch::zeros_like(off);
  const int total = static_cast<int>(planes * height * width);
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getDefaultCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, state0.scalar_type(), "cspn_backward_cuda", [&] {
    for (int64_t t = states.size(0) - 2; t >= 0; --t) {
      auto next_grad = torch::empty_like(grad_state);
      cspn_backward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          grad_state.data_ptr<scalar_t>(), states[t].data_ptr<scalar_t>(),
          center.data_ptr<scalar_t>(), off.data_ptr<scalar_t>(),
          next_grad.data_ptr<scalar_t>(), grad_center.data_ptr<scalar_t>(),
          grad_off.data_ptr<scalar_t>(), static_cast<int>(planes), height, width);
      grad_state = next_grad;
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_state, grad_center, grad_off};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cspn_forward, "CSPN recurrent forward (CUDA)");
  m.def("backward", &cspn_backward, "CSPN recurrent backward (CUDA)");
}
