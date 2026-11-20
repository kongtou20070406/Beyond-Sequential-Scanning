#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ initial,
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ center,
    const scalar_t* __restrict__ off,
    scalar_t* __restrict__ out,
    int planes, int height, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int hw = height * width;
  if (idx >= planes * hw) return;
  const int p = idx % hw;
  const int y = p / width;
  const int x = p - y * width;
  const int plane = idx / hw;
  float value = static_cast<float>(center[idx]) * static_cast<float>(initial[idx]);
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
__global__ void backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ initial,
    const scalar_t* __restrict__ state,
    const scalar_t* __restrict__ center,
    const scalar_t* __restrict__ off,
    scalar_t* __restrict__ grad_state,
    scalar_t* __restrict__ grad_initial,
    scalar_t* __restrict__ grad_center,
    scalar_t* __restrict__ grad_off,
    int planes, int height, int width) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int hw = height * width;
  if (idx >= planes * hw) return;
  const int q = idx % hw;
  const int y = q / width;
  const int x = q - y * width;
  const int plane = idx / hw;
  const float go = static_cast<float>(grad_out[idx]);
  grad_initial[idx] = static_cast<scalar_t>(
      static_cast<float>(grad_initial[idx]) + static_cast<float>(center[idx]) * go);
  grad_center[idx] = static_cast<scalar_t>(
      static_cast<float>(grad_center[idx]) + go * static_cast<float>(initial[idx]));
  float gs = 0.0f;
  constexpr int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  constexpr int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int py = y - dy[i];
    const int px = x - dx[i];
    if ((unsigned)py < (unsigned)height && (unsigned)px < (unsigned)width) {
      const int p = py * width + px;
      gs += static_cast<float>(off[(plane * 8 + i) * hw + p])
          * static_cast<float>(grad_out[plane * hw + p]);
    }
    const int ny = y + dy[i];
    const int nx = x + dx[i];
    const int oidx = (plane * 8 + i) * hw + q;
    float ga = static_cast<float>(grad_off[oidx]);
    if ((unsigned)ny < (unsigned)height && (unsigned)nx < (unsigned)width) {
      ga += go * static_cast<float>(state[plane * hw + ny * width + nx]);
    }
    grad_off[oidx] = static_cast<scalar_t>(ga);
  }
  grad_state[idx] = static_cast<scalar_t>(gs);
}

void check(const at::Tensor& initial, const at::Tensor& center,
           const at::Tensor& off) {
  TORCH_CHECK(initial.is_cuda() && center.is_cuda() && off.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(initial.is_contiguous() && center.is_contiguous() && off.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(initial.sizes() == center.sizes(), "initial/center shape mismatch");
  TORCH_CHECK(off.dim() == 5 && off.size(2) == 8, "off must be BC8HW");
  TORCH_CHECK(initial.scalar_type() == center.scalar_type() && initial.scalar_type() == off.scalar_type(), "dtype mismatch");
  TORCH_CHECK(initial.scalar_type() == at::kFloat, "formal CSPN backend requires FP32");
}

}  // namespace

at::Tensor cspn_anchor_forward(at::Tensor initial, at::Tensor center,
                               at::Tensor off, int64_t steps) {
  check(initial, center, off);
  TORCH_CHECK(steps > 0, "steps must be positive");
  c10::cuda::CUDAGuard guard(initial.device());
  const int64_t planes = initial.size(0) * initial.size(1);
  const int height = initial.size(2), width = initial.size(3);
  auto states = at::empty({steps + 1, initial.size(0), initial.size(1), height, width}, initial.options());
  states[0].copy_(initial);
  const int total = static_cast<int>(planes * height * width);
  const int threads = 256, blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream(initial.device().index());
  for (int64_t t = 0; t < steps; ++t) {
    forward_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        states[0].data_ptr<float>(), states[t].data_ptr<float>(),
        center.data_ptr<float>(), off.data_ptr<float>(),
        states[t + 1].data_ptr<float>(), static_cast<int>(planes), height, width);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return states;
}

std::vector<at::Tensor> cspn_anchor_backward(
    at::Tensor grad_out, at::Tensor states,
    at::Tensor center, at::Tensor off) {
  auto initial = states[0];
  check(initial, center, off);
  c10::cuda::CUDAGuard guard(initial.device());
  const int64_t planes = initial.size(0) * initial.size(1);
  const int height = initial.size(2), width = initial.size(3);
  auto grad_state = grad_out.contiguous();
  auto grad_initial = at::zeros_like(initial);
  auto grad_center = at::zeros_like(center);
  auto grad_off = at::zeros_like(off);
  const int total = static_cast<int>(planes * height * width);
  const int threads = 256, blocks = (total + threads - 1) / threads;
  auto stream = at::cuda::getCurrentCUDAStream(initial.device().index());
  for (int64_t t = states.size(0) - 2; t >= 0; --t) {
    auto next_grad = at::empty_like(grad_state);
    backward_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        grad_state.data_ptr<float>(), initial.data_ptr<float>(),
        states[t].data_ptr<float>(), center.data_ptr<float>(),
        off.data_ptr<float>(), next_grad.data_ptr<float>(),
        grad_initial.data_ptr<float>(), grad_center.data_ptr<float>(),
        grad_off.data_ptr<float>(), static_cast<int>(planes), height, width);
    grad_state = next_grad;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  grad_initial.add_(grad_state);
  return {grad_initial, grad_center, grad_off};
}
