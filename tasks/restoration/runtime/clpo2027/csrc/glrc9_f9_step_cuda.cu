#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kTaps = 9;
constexpr int kThreads = 256;

__device__ __forceinline__ int tap_dy(int tap) {
  return tap == 1 || tap == 2 || tap == 3 ? -1 : (tap >= 6 ? 1 : 0);
}

__device__ __forceinline__ int tap_dx(int tap) {
  return tap == 1 || tap == 4 || tap == 6 ? -1 : (tap == 3 || tap == 5 || tap == 8 ? 1 : 0);
}

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t* ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_float<at::BFloat16>(const at::BFloat16* ptr) {
  return __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(ptr));
}

template <typename scalar_t>
__device__ __forceinline__ void store_float(scalar_t* ptr, float value) {
  *ptr = static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ void store_float<at::BFloat16>(at::BFloat16* ptr, float value) {
  *reinterpret_cast<nv_bfloat16*>(ptr) = __float2bfloat16(value);
}

template <typename input_t, typename coeff_t, typename endpoint_t, typename trajectory_t>
__global__ void f9_step_forward_kernel(
    const input_t* __restrict__ packed_f,
    const coeff_t* __restrict__ ar,
    const coeff_t* __restrict__ ai,
    const float* __restrict__ current_r,
    const float* __restrict__ current_i,
    float* __restrict__ next_r,
    float* __restrict__ next_i,
    endpoint_t* __restrict__ endpoint,
    trajectory_t* __restrict__ trajectory,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t plane_size,
    int64_t total_elements,
    int64_t step,
    int64_t trajectory_stride,
    bool final_step) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_elements) return;
  const int64_t state_batch_stride = channels * plane_size;
  const int64_t batch_index = linear / state_batch_stride;
  const int64_t in_batch = linear - batch_index * state_batch_stride;
  const int64_t channel = in_batch / plane_size;
  const int64_t pix = in_batch - channel * plane_size;
  const int64_t y = pix / width;
  const int64_t x = pix - y * width;
  const int64_t state_index = linear;
  const int64_t packed_batch_stride = 2 * state_batch_stride;
  const int64_t packed_real = batch_index * packed_batch_stride + channel * plane_size;
  const int64_t packed_imag = packed_real + state_batch_stride;
  const int64_t coeff_base = (batch_index * kTaps * channels + channel) * plane_size;
  const int64_t trajectory_base = step * trajectory_stride + packed_real + pix;

  const float f_r = load_float(packed_f + packed_real + pix);
  const float f_i = load_float(packed_f + packed_imag + pix);
  const float z_r = f_r + current_r[state_index];
  const float z_i = f_i + current_i[state_index];
  store_float(trajectory + trajectory_base, z_r);
  store_float(trajectory + trajectory_base + state_batch_stride, z_i);

  float out_r = 0.0f;
  float out_i = 0.0f;
  #pragma unroll
  for (int tap = 0; tap < kTaps; ++tap) {
    const int64_t ny = y + tap_dy(tap);
    const int64_t nx = x + tap_dx(tap);
    if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
    const int64_t neighbor = ny * width + nx;
    const int64_t state_neighbor = batch_index * state_batch_stride + channel * plane_size + neighbor;
    const float neighbor_r = load_float(packed_f + packed_real + neighbor) + current_r[state_neighbor];
    const float neighbor_i = load_float(packed_f + packed_imag + neighbor) + current_i[state_neighbor];
    const int64_t coeff_index = coeff_base + tap * state_batch_stride + pix;
    const float a_r = load_float(ar + coeff_index);
    const float a_i = load_float(ai + coeff_index);
    out_r = fmaf(a_r, neighbor_r, fmaf(-a_i, neighbor_i, out_r));
    out_i = fmaf(a_r, neighbor_i, fmaf(a_i, neighbor_r, out_i));
  }
  next_r[state_index] = out_r;
  next_i[state_index] = out_i;
  if (final_step) {
    store_float(endpoint + packed_real + pix, out_r);
    store_float(endpoint + packed_imag + pix, out_i);
  }
}

template <typename q_t>
__global__ void f9_init_adjoint_kernel(
    const q_t* __restrict__ grad_endpoint,
    float* __restrict__ current_r,
    float* __restrict__ current_i,
    int64_t channels,
    int64_t plane_size,
    int64_t total_elements) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_elements) return;
  const int64_t state_batch_stride = channels * plane_size;
  const int64_t batch_index = linear / state_batch_stride;
  const int64_t in_batch = linear - batch_index * state_batch_stride;
  const int64_t channel = in_batch / plane_size;
  const int64_t pix = in_batch - channel * plane_size;
  const int64_t packed_batch_stride = 2 * state_batch_stride;
  const int64_t packed_real = batch_index * packed_batch_stride + channel * plane_size;
  current_r[linear] = load_float(grad_endpoint + packed_real + pix);
  current_i[linear] = load_float(grad_endpoint + packed_real + state_batch_stride + pix);
}

template <typename q_t, typename coeff_t, typename trajectory_t, typename grad_f_t, typename grad_coeff_t>
__global__ void f9_step_adjoint_kernel(
    const q_t* __restrict__ grad_endpoint,
    const coeff_t* __restrict__ ar,
    const coeff_t* __restrict__ ai,
    const trajectory_t* __restrict__ trajectory,
    const float* __restrict__ current_r,
    const float* __restrict__ current_i,
    float* __restrict__ next_r,
    float* __restrict__ next_i,
    grad_f_t* __restrict__ grad_f,
    grad_coeff_t* __restrict__ grad_ar,
    grad_coeff_t* __restrict__ grad_ai,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t plane_size,
    int64_t total_elements,
    int64_t step,
    int64_t trajectory_stride) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total_elements) return;
  const int64_t state_batch_stride = channels * plane_size;
  const int64_t batch_index = linear / state_batch_stride;
  const int64_t in_batch = linear - batch_index * state_batch_stride;
  const int64_t channel = in_batch / plane_size;
  const int64_t pix = in_batch - channel * plane_size;
  const int64_t y = pix / width;
  const int64_t x = pix - y * width;
  const int64_t packed_batch_stride = 2 * state_batch_stride;
  const int64_t packed_real = batch_index * packed_batch_stride + channel * plane_size;
  const int64_t coeff_base = (batch_index * kTaps * channels + channel) * plane_size;
  const int64_t trajectory_slice_base = step * trajectory_stride + packed_real;
  const float q_r = current_r[linear];
  const float q_i = current_i[linear];

  #pragma unroll
  for (int tap = 0; tap < kTaps; ++tap) {
    const int64_t ny = y + tap_dy(tap);
    const int64_t nx = x + tap_dx(tap);
    if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
    const int64_t neighbor = ny * width + nx;
    const int64_t traj_neighbor = trajectory_slice_base + neighbor;
    const float z_r = load_float(trajectory + traj_neighbor);
    const float z_i = load_float(trajectory + traj_neighbor + state_batch_stride);
    const int64_t coeff_index = coeff_base + tap * state_batch_stride + pix;
    const float old_ar = load_float(grad_ar + coeff_index);
    const float old_ai = load_float(grad_ai + coeff_index);
    store_float(grad_ar + coeff_index, fmaf(q_r, z_r, fmaf(q_i, z_i, old_ar)));
    store_float(grad_ai + coeff_index, fmaf(-q_r, z_i, fmaf(q_i, z_r, old_ai)));
  }

  float propagated_r = 0.0f;
  float propagated_i = 0.0f;
  #pragma unroll
  for (int tap = 0; tap < kTaps; ++tap) {
    const int64_t qy = y - tap_dy(tap);
    const int64_t qx = x - tap_dx(tap);
    if (qy < 0 || qy >= height || qx < 0 || qx >= width) continue;
    const int64_t qpix = qy * width + qx;
    const int64_t qindex = batch_index * state_batch_stride + channel * plane_size + qpix;
    const int64_t coeff_index = coeff_base + tap * state_batch_stride + qpix;
    const float g_r = current_r[qindex];
    const float g_i = current_i[qindex];
    const float a_r = load_float(ar + coeff_index);
    const float a_i = load_float(ai + coeff_index);
    propagated_r = fmaf(a_r, g_r, fmaf(a_i, g_i, propagated_r));
    propagated_i = fmaf(-a_i, g_r, fmaf(a_r, g_i, propagated_i));
  }
  next_r[linear] = propagated_r;
  next_i[linear] = propagated_i;
  const float old_f_r = load_float(grad_f + packed_real + pix);
  const float old_f_i = load_float(grad_f + packed_real + state_batch_stride + pix);
  store_float(grad_f + packed_real + pix, old_f_r + propagated_r);
  store_float(grad_f + packed_real + state_batch_stride + pix, old_f_i + propagated_i);
}

void check_common(
    const torch::Tensor& packed_f,
    const torch::Tensor& ar,
    const torch::Tensor& ai) {
  TORCH_CHECK(
      packed_f.is_cuda() && ar.is_cuda() && ai.is_cuda(),
      "GLRC9 step F9 backend requires CUDA tensors");
  TORCH_CHECK(
      packed_f.is_contiguous() && ar.is_contiguous() && ai.is_contiguous(),
      "GLRC9 step F9 backend requires contiguous tensors");
  TORCH_CHECK(
      packed_f.scalar_type() == at::kFloat || packed_f.scalar_type() == at::kBFloat16,
      "state must use float32 or bfloat16");
  TORCH_CHECK(
      ar.scalar_type() == at::kFloat || ar.scalar_type() == at::kBFloat16,
      "coefficients must use float32 or bfloat16");
  TORCH_CHECK(ar.scalar_type() == ai.scalar_type(), "real and imaginary coefficients must use one dtype");
  TORCH_CHECK(
      packed_f.dim() == 4 && packed_f.size(1) % 2 == 0,
      "packed state must have shape [B,2D,H,W]");
  TORCH_CHECK(
      ar.dim() == 5 && ai.sizes() == ar.sizes() && ar.size(1) == kTaps,
      "coefficients must have shape [B,9,D,H,W]");
  TORCH_CHECK(
      ar.size(0) == packed_f.size(0) && ar.size(2) == packed_f.size(1) / 2 &&
          ar.size(3) == packed_f.size(2) && ar.size(4) == packed_f.size(3),
      "coefficient/state shapes do not match");
}

int grid_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

template <typename input_t, typename coeff_t, typename endpoint_t, typename trajectory_t>
void launch_forward(
    const torch::Tensor& packed_f,
    const torch::Tensor& ar,
    const torch::Tensor& ai,
    torch::Tensor& endpoint,
    torch::Tensor& trajectory,
    int64_t steps,
    cudaStream_t stream) {
  const int64_t batch = packed_f.size(0);
  const int64_t channels = packed_f.size(1) / 2;
  const int64_t height = packed_f.size(2);
  const int64_t width = packed_f.size(3);
  const int64_t plane_size = height * width;
  const int64_t state_elements = batch * channels * plane_size;
  const int64_t trajectory_stride = batch * 2 * channels * plane_size;
  auto work_options = packed_f.options().dtype(at::kFloat);
  auto p_a = torch::zeros({batch, channels, height, width}, work_options);
  auto p_b = torch::empty_like(p_a);
  float* current_r = p_a.data_ptr<float>();
  float* next_r = p_b.data_ptr<float>();
  float* current_i = nullptr;
  float* next_i = nullptr;
  // The ping-pong arrays are real/imag packed as separate contiguous planes.
  // p_a/p_b are allocated as [B,D,H,W], so use distinct real and imag tensors.
  auto p_ai = torch::zeros({batch, channels, height, width}, work_options);
  auto p_bi = torch::empty_like(p_ai);
  current_i = p_ai.data_ptr<float>();
  next_i = p_bi.data_ptr<float>();
  for (int64_t step = 0; step < steps; ++step) {
    const bool final_step = step + 1 == steps;
    f9_step_forward_kernel<input_t, coeff_t, endpoint_t, trajectory_t><<<
        grid_for(state_elements), kThreads, 0, stream>>>(
        packed_f.data_ptr<input_t>(), ar.data_ptr<coeff_t>(), ai.data_ptr<coeff_t>(),
        current_r, current_i, next_r, next_i, endpoint.data_ptr<endpoint_t>(),
        trajectory.data_ptr<trajectory_t>(), channels, height, width, plane_size,
        state_elements, step, trajectory_stride, final_step);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::swap(current_r, next_r);
    std::swap(current_i, next_i);
  }
}

template <typename q_t, typename coeff_t, typename trajectory_t, typename grad_f_t, typename grad_coeff_t>
void launch_adjoint(
    const torch::Tensor& grad_endpoint,
    const torch::Tensor& ar,
    const torch::Tensor& ai,
    const torch::Tensor& trajectory,
    torch::Tensor& grad_f,
    torch::Tensor& grad_ar,
    torch::Tensor& grad_ai,
    int64_t steps,
    cudaStream_t stream) {
  const int64_t batch = grad_endpoint.size(0);
  const int64_t channels = grad_endpoint.size(1) / 2;
  const int64_t height = grad_endpoint.size(2);
  const int64_t width = grad_endpoint.size(3);
  const int64_t plane_size = height * width;
  const int64_t state_elements = batch * channels * plane_size;
  const int64_t trajectory_stride = batch * 2 * channels * plane_size;
  auto work_options = grad_endpoint.options().dtype(at::kFloat);
  auto q_a_r = torch::empty({batch, channels, height, width}, work_options);
  auto q_a_i = torch::empty_like(q_a_r);
  auto q_b_r = torch::empty_like(q_a_r);
  auto q_b_i = torch::empty_like(q_a_r);
  const int blocks = grid_for(state_elements);
  f9_init_adjoint_kernel<q_t><<<blocks, kThreads, 0, stream>>>(
      grad_endpoint.data_ptr<q_t>(), q_a_r.data_ptr<float>(), q_a_i.data_ptr<float>(),
      channels, plane_size, state_elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  float* current_r = q_a_r.data_ptr<float>();
  float* current_i = q_a_i.data_ptr<float>();
  float* next_r = q_b_r.data_ptr<float>();
  float* next_i = q_b_i.data_ptr<float>();
  for (int64_t reverse = 0; reverse < steps; ++reverse) {
    const int64_t step = steps - reverse - 1;
    f9_step_adjoint_kernel<q_t, coeff_t, trajectory_t, grad_f_t, grad_coeff_t><<<
        blocks, kThreads, 0, stream>>>(
        grad_endpoint.data_ptr<q_t>(), ar.data_ptr<coeff_t>(), ai.data_ptr<coeff_t>(),
        trajectory.data_ptr<trajectory_t>(), current_r, current_i, next_r, next_i,
        grad_f.data_ptr<grad_f_t>(), grad_ar.data_ptr<grad_coeff_t>(),
        grad_ai.data_ptr<grad_coeff_t>(), channels, height, width, plane_size,
        state_elements, step, trajectory_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    std::swap(current_r, next_r);
    std::swap(current_i, next_i);
  }
}

}  // namespace

std::vector<torch::Tensor> glrc9_f9_step_forward_cuda(
    torch::Tensor packed_f, torch::Tensor ar, torch::Tensor ai, int64_t steps) {
  check_common(packed_f, ar, ai);
  TORCH_CHECK(steps > 0 && steps <= 1024, "steps must be in [1,1024]");
  const auto guard = c10::cuda::CUDAGuard(packed_f.device());
  const bool exact_mixed = packed_f.scalar_type() == at::kBFloat16 && ar.scalar_type() == at::kFloat;
  auto endpoint = exact_mixed
      ? torch::empty(packed_f.sizes(), packed_f.options().dtype(at::kFloat))
      : torch::empty_like(packed_f);
  auto trajectory = torch::empty(
      {steps, packed_f.size(0), packed_f.size(1), packed_f.size(2), packed_f.size(3)},
      packed_f.options());
  auto stream = at::cuda::getCurrentCUDAStream(packed_f.device().index());
  if (packed_f.scalar_type() == at::kFloat && ar.scalar_type() == at::kFloat) {
    launch_forward<float, float, float, float>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  } else if (packed_f.scalar_type() == at::kBFloat16 && ar.scalar_type() == at::kFloat) {
    launch_forward<at::BFloat16, float, float, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  } else if (packed_f.scalar_type() == at::kBFloat16) {
    launch_forward<at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  } else {
    launch_forward<float, at::BFloat16, float, float>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  }
  return {endpoint, trajectory};
}

std::vector<torch::Tensor> glrc9_f9_step_adjoint_cuda(
    torch::Tensor grad_endpoint, torch::Tensor ar, torch::Tensor ai,
    torch::Tensor trajectory, int64_t steps) {
  check_common(grad_endpoint, ar, ai);
  TORCH_CHECK(trajectory.is_cuda() && trajectory.is_contiguous(), "trajectory must be contiguous CUDA tensor");
  TORCH_CHECK(trajectory.scalar_type() == at::kFloat || trajectory.scalar_type() == at::kBFloat16,
              "trajectory must use float32 or bfloat16");
  TORCH_CHECK(trajectory.device() == grad_endpoint.device(), "trajectory and endpoint gradient must share device");
  TORCH_CHECK(
      trajectory.sizes() == torch::IntArrayRef({steps, grad_endpoint.size(0), grad_endpoint.size(1), grad_endpoint.size(2), grad_endpoint.size(3)}),
      "trajectory shape mismatch");
  const auto guard = c10::cuda::CUDAGuard(grad_endpoint.device());
  auto grad_f = torch::zeros_like(grad_endpoint);
  auto grad_ar = torch::zeros_like(ar);
  auto grad_ai = torch::zeros_like(ai);
  auto stream = at::cuda::getCurrentCUDAStream(grad_endpoint.device().index());
  if (grad_endpoint.scalar_type() == at::kFloat && ar.scalar_type() == at::kFloat) {
    TORCH_CHECK(trajectory.scalar_type() == at::kFloat || trajectory.scalar_type() == at::kBFloat16,
                "FP32 q requires FP32 or BF16 trajectory");
    if (trajectory.scalar_type() == at::kBFloat16) {
      launch_adjoint<float, float, at::BFloat16, float, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    } else {
      launch_adjoint<float, float, float, float, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    }
  } else if (grad_endpoint.scalar_type() == at::kBFloat16 && ar.scalar_type() == at::kFloat) {
    TORCH_CHECK(trajectory.scalar_type() == at::kBFloat16, "BF16 q with FP32 coefficients requires BF16 trajectory");
    launch_adjoint<at::BFloat16, float, at::BFloat16, at::BFloat16, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
  } else if (grad_endpoint.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(trajectory.scalar_type() == at::kBFloat16, "BF16 q requires BF16 trajectory");
    launch_adjoint<at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
  } else {
    TORCH_CHECK(trajectory.scalar_type() == at::kFloat, "FP32 q with BF16 coefficients requires FP32 trajectory");
    launch_adjoint<float, at::BFloat16, float, float, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
  }
  return {grad_f, grad_ar, grad_ai};
}
