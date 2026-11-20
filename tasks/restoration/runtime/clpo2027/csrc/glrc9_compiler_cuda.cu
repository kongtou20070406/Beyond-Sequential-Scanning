#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kModes = 9;
constexpr int kProjectionThreads = 256;
constexpr int kMaxHeadThreads = 1024;

__device__ __forceinline__ int q_index(int a, int b) { return a * 3 + b; }

__device__ __forceinline__ int shell_index(int q) {
  // q order is (0,0),(0,1),(0,2),(1,0),...,(2,2).
  if (q == 0) return 0;
  if (q == 1 || q == 3) return 1;
  if (q == 4) return 2;
  if (q == 2 || q == 6) return 3;
  return 4;
}

__device__ __forceinline__ int shell_slot(int shell) {
  // Physical q9 slots: DC, four global shells, four local atoms.
  constexpr int slots[5] = {0, 3, 4, 6, 7};
  return slots[shell];
}

__device__ __forceinline__ int local_slot(int local) {
  constexpr int slots[4] = {1, 2, 5, 8};
  return slots[local];
}

template <typename scalar_t>
__device__ __forceinline__ float load_state(const scalar_t* ptr) {
  return static_cast<float>(*ptr);
}

template <>
__device__ __forceinline__ float load_state<at::BFloat16>(const at::BFloat16* ptr) {
  return __bfloat162float(*reinterpret_cast<const nv_bfloat16*>(ptr));
}

template <typename scalar_t>
__device__ __forceinline__ void store_state(scalar_t* ptr, float value) {
  *ptr = static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ void store_state<at::BFloat16>(at::BFloat16* ptr, float value) {
  *reinterpret_cast<nv_bfloat16*>(ptr) = __float2bfloat16(value);
}

__device__ __forceinline__ float load_mixed_value(
    const float* mixed,
    int batch,
    int state,
    int imag,
    int y,
    int x,
    int states,
    int height,
    int width) {
  if (y < 0 || y >= height || x < 0 || x >= width) return 0.0f;
  const int plane = height * width;
  const int channel = imag ? states + state : state;
  return mixed[(batch * (2 * states) + channel) * plane + y * width + x];
}

template <typename state_t>
__device__ __forceinline__ float load_mixed_value(
    const state_t* mixed,
    int batch,
    int state,
    int imag,
    int y,
    int x,
    int states,
    int height,
    int width) {
  if (y < 0 || y >= height || x < 0 || x >= width) return 0.0f;
  const int plane = height * width;
  const int channel = imag ? states + state : state;
  return load_state(mixed + (batch * (2 * states) + channel) * plane + y * width + x);
}

template <typename state_t>
__global__ void widely_linear_kernel(
    const state_t* __restrict__ packed,
    const float* __restrict__ m_real,
    const float* __restrict__ m_imag,
    const float* __restrict__ n_real,
    const float* __restrict__ n_imag,
    state_t* __restrict__ mixed,
    int states,
    int height,
    int width) {
  const int plane = blockIdx.x;
  const int pixels = height * width;
  const int total = gridDim.x;
  if (plane >= total) return;
  const int batch = plane / pixels;
  const int pix = plane - batch * pixels;
  const int d = threadIdx.x;
  if (d >= states) return;
  float out_r = 0.0f;
  float out_i = 0.0f;
  const int input_base = batch * (2 * states) * pixels;
  const int output_base = input_base;
  for (int j = 0; j < states; ++j) {
    const float fr = load_state(packed + input_base + j * pixels + pix);
    const float fi = load_state(packed + input_base + (states + j) * pixels + pix);
    const float mr = m_real[d * states + j];
    const float mi = m_imag[d * states + j];
    const float nr = n_real[d * states + j];
    const float ni = n_imag[d * states + j];
    out_r = fmaf(mr + nr, fr, out_r);
    out_r = fmaf(-mi + ni, fi, out_r);
    out_i = fmaf(mi + ni, fr, out_i);
    out_i = fmaf(mr - nr, fi, out_i);
  }
  store_state(mixed + output_base + d * pixels + pix, out_r);
  store_state(mixed + output_base + (states + d) * pixels + pix, out_i);
}

template <typename state_t>
__global__ void low3_projection_kernel(
    const state_t* __restrict__ mixed,
    const float* __restrict__ basis_y,
    const float* __restrict__ basis_x,
    float* __restrict__ low_real,
    float* __restrict__ low_imag,
    float* __restrict__ conditioning,
    int states,
    int height,
    int width) {
  extern __shared__ float shared[];
  float* red_r = shared;
  float* red_i = red_r + kModes * blockDim.x;
  float* red_local = red_i + kModes * blockDim.x;
  const int bd = blockIdx.x;
  const int batch = bd / states;
  const int state = bd - batch * states;
  const int pixels = height * width;
  if (batch >= gridDim.x / states) return;
  float acc_r[kModes] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float acc_i[kModes] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float acc_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int pix = threadIdx.x; pix < pixels; pix += blockDim.x) {
    const int y = pix / width;
    const int x = pix - y * width;
    const float real = load_mixed_value(mixed, batch, state, 0, y, x, states, height, width);
    const float imag = load_mixed_value(mixed, batch, state, 1, y, x, states, height, width);
    for (int a = 0; a < 3; ++a) {
      const float by = basis_y[a * height + y];
      for (int b = 0; b < 3; ++b) {
        const float basis = by * basis_x[b * width + x];
        const int q = q_index(a, b);
        acc_r[q] = fmaf(real, basis, acc_r[q]);
        acc_i[q] = fmaf(imag, basis, acc_i[q]);
      }
    }
    const float tl_r = load_mixed_value(mixed, batch, state, 0, y - 1, x - 1, states, height, width);
    const float tr_r = load_mixed_value(mixed, batch, state, 0, y - 1, x + 1, states, height, width);
    const float bl_r = load_mixed_value(mixed, batch, state, 0, y + 1, x - 1, states, height, width);
    const float br_r = load_mixed_value(mixed, batch, state, 0, y + 1, x + 1, states, height, width);
    const float tl_i = load_mixed_value(mixed, batch, state, 1, y - 1, x - 1, states, height, width);
    const float tr_i = load_mixed_value(mixed, batch, state, 1, y - 1, x + 1, states, height, width);
    const float bl_i = load_mixed_value(mixed, batch, state, 1, y + 1, x - 1, states, height, width);
    const float br_i = load_mixed_value(mixed, batch, state, 1, y + 1, x + 1, states, height, width);
    const float a0r = 0.25f * (tl_r + tr_r + bl_r + br_r);
    const float a1r = 0.25f * (tl_r - tr_r + bl_r - br_r);
    const float a2r = 0.25f * (tl_r + tr_r - bl_r - br_r);
    const float a3r = 0.25f * (tl_r - tr_r - bl_r + br_r);
    const float a0i = 0.25f * (tl_i + tr_i + bl_i + br_i);
    const float a1i = 0.25f * (tl_i - tr_i + bl_i - br_i);
    const float a2i = 0.25f * (tl_i + tr_i - bl_i - br_i);
    const float a3i = 0.25f * (tl_i - tr_i - bl_i + br_i);
    acc_local[0] = fmaf(a0r, a0r, fmaf(a0i, a0i, acc_local[0]));
    acc_local[1] = fmaf(a1r, a1r, fmaf(a1i, a1i, acc_local[1]));
    acc_local[2] = fmaf(a2r, a2r, fmaf(a2i, a2i, acc_local[2]));
    acc_local[3] = fmaf(a3r, a3r, fmaf(a3i, a3i, acc_local[3]));
  }
  for (int q = 0; q < kModes; ++q) {
    red_r[q * blockDim.x + threadIdx.x] = acc_r[q];
    red_i[q * blockDim.x + threadIdx.x] = acc_i[q];
  }
  for (int local = 0; local < 4; ++local) {
    red_local[local * blockDim.x + threadIdx.x] = acc_local[local];
  }
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      for (int q = 0; q < kModes; ++q) {
        red_r[q * blockDim.x + threadIdx.x] += red_r[q * blockDim.x + threadIdx.x + stride];
        red_i[q * blockDim.x + threadIdx.x] += red_i[q * blockDim.x + threadIdx.x + stride];
      }
      for (int local = 0; local < 4; ++local) {
        red_local[local * blockDim.x + threadIdx.x] += red_local[local * blockDim.x + threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x != 0) return;
  float score[kModes];
  for (int q = 0; q < kModes; ++q) {
    const float r = red_r[q * blockDim.x];
    const float i = red_i[q * blockDim.x];
    low_real[(batch * states + state) * kModes + q] = r;
    low_imag[(batch * states + state) * kModes + q] = i;
  }
  const float e0 = red_r[0] * red_r[0] + red_i[0] * red_i[0];
  const float e1 = red_r[1] * red_r[1] + red_i[1] * red_i[1];
  const float e2 = red_r[2] * red_r[2] + red_i[2] * red_i[2];
  const float e3 = red_r[3] * red_r[3] + red_i[3] * red_i[3];
  const float e4 = red_r[4] * red_r[4] + red_i[4] * red_i[4];
  const float e5 = red_r[5] * red_r[5] + red_i[5] * red_i[5];
  const float e6 = red_r[6] * red_r[6] + red_i[6] * red_i[6];
  const float e7 = red_r[7] * red_r[7] + red_i[7] * red_i[7];
  const float e8 = red_r[8] * red_r[8] + red_i[8] * red_i[8];
  score[0] = e0;
  score[1] = 0.5f * (e1 + e3);
  score[2] = e4;
  score[3] = 0.5f * (e2 + e6);
  score[4] = (e5 + e7 + e8) / 3.0f;
  const float inv_pixels = 1.0f / static_cast<float>(pixels);
  for (int local = 0; local < 4; ++local) {
    // The fixed four-corner frame has squared mass 1/4.
    score[5 + local] = 4.0f * red_local[local * blockDim.x] * inv_pixels;
  }
  float denom = 0.0f;
  constexpr float eps = std::numeric_limits<float>::epsilon();
  for (int q = 0; q < kModes; ++q) denom += score[q] + eps;
  for (int q = 0; q < kModes; ++q) {
    conditioning[(batch * states + state) * kModes + q] = 0.5f + 4.5f * (score[q] + eps) / denom;
  }
}

template <typename state_t>
__global__ void latent_kernel(
    const state_t* __restrict__ mixed,
    const float* __restrict__ low_real,
    const float* __restrict__ low_imag,
    const float* __restrict__ conditioning,
    const float* __restrict__ gain_real,
    const float* __restrict__ gain_imag,
    const float* __restrict__ basis_y,
    const float* __restrict__ basis_x,
    state_t* __restrict__ latent,
    int states,
    int height,
    int width,
    float state_epsilon) {
  const int plane = blockIdx.x;
  const int pixels = height * width;
  const int batch = plane / pixels;
  const int pix = plane - batch * pixels;
  const int y = pix / width;
  const int x = pix - y * width;
  const int state = threadIdx.x;
  if (state >= states) return;
  const int bd = batch * states + state;
  float spectral_r = 0.0f;
  float spectral_i = 0.0f;
  const float* cond = conditioning + bd * kModes;
  const float* lr = low_real + bd * kModes;
  const float* li = low_imag + bd * kModes;
  const float* gr = gain_real + state * kModes;
  const float* gi = gain_imag + state * kModes;
  for (int a = 0; a < 3; ++a) {
    const float by = basis_y[a * height + y];
    for (int b = 0; b < 3; ++b) {
      const int q = q_index(a, b);
      const int shell = shell_index(q);
      const int slot = shell_slot(shell);
      const float route = cond[shell];
      const float multiplier_r = route * (gr[slot] - (slot == 0 ? 1.0f : 0.0f));
      const float multiplier_i = route * gi[slot];
      const float basis = by * basis_x[b * width + x];
      spectral_r = fmaf(basis, fmaf(multiplier_r, lr[q], fmaf(-multiplier_i, li[q], 0.0f)), spectral_r);
      spectral_i = fmaf(basis, fmaf(multiplier_i, lr[q], fmaf(multiplier_r, li[q], 0.0f)), spectral_i);
    }
  }
  float local_r = 0.0f;
  float local_i = 0.0f;
  const float tl_r = load_mixed_value(mixed, batch, state, 0, y - 1, x - 1, states, height, width);
  const float tr_r = load_mixed_value(mixed, batch, state, 0, y - 1, x + 1, states, height, width);
  const float bl_r = load_mixed_value(mixed, batch, state, 0, y + 1, x - 1, states, height, width);
  const float br_r = load_mixed_value(mixed, batch, state, 0, y + 1, x + 1, states, height, width);
  const float tl_i = load_mixed_value(mixed, batch, state, 1, y - 1, x - 1, states, height, width);
  const float tr_i = load_mixed_value(mixed, batch, state, 1, y - 1, x + 1, states, height, width);
  const float bl_i = load_mixed_value(mixed, batch, state, 1, y + 1, x - 1, states, height, width);
  const float br_i = load_mixed_value(mixed, batch, state, 1, y + 1, x + 1, states, height, width);
  const float ar[4] = {
      0.25f * (tl_r + tr_r + bl_r + br_r),
      0.25f * (tl_r - tr_r + bl_r - br_r),
      0.25f * (tl_r + tr_r - bl_r - br_r),
      0.25f * (tl_r - tr_r - bl_r + br_r),
  };
  const float ai[4] = {
      0.25f * (tl_i + tr_i + bl_i + br_i),
      0.25f * (tl_i - tr_i + bl_i - br_i),
      0.25f * (tl_i + tr_i - bl_i - br_i),
      0.25f * (tl_i - tr_i - bl_i + br_i),
  };
  for (int local = 0; local < 4; ++local) {
    const int slot = local_slot(local);
    const float route = cond[5 + local];
    const float multiplier_r = route * gr[slot];
    const float multiplier_i = route * gi[slot];
    local_r = fmaf(multiplier_r, ar[local], fmaf(-multiplier_i, ai[local], local_r));
    local_i = fmaf(multiplier_i, ar[local], fmaf(multiplier_r, ai[local], local_i));
  }
  const float mixed_r = load_mixed_value(mixed, batch, state, 0, y, x, states, height, width);
  const float mixed_i = load_mixed_value(mixed, batch, state, 1, y, x, states, height, width);
  const float paired_r = mixed_r + spectral_r + local_r;
  const float paired_i = mixed_i + spectral_i + local_i;
  const float magnitude = sqrtf(paired_r * paired_r + paired_i * paired_i + state_epsilon);
  const float radial = 1.0f / (1.0f + expf(-magnitude));
  const int plane_base = batch * (2 * states) * pixels + pix;
  store_state(latent + plane_base + (2 * state) * pixels, paired_r * radial);
  store_state(latent + plane_base + (2 * state + 1) * pixels, paired_i * radial);
}

template <typename state_t>
__global__ void grouped_head_kernel(
    const state_t* __restrict__ latent,
    const float* __restrict__ head_weight,
    const float* __restrict__ head_bias,
    state_t* __restrict__ raw,
    int states,
    int height,
    int width) {
  const int plane = blockIdx.x;
  const int pixels = height * width;
  const int batch = plane / pixels;
  const int pix = plane - batch * pixels;
  const int output_channel = threadIdx.x;
  const int output_channels = 9 * states;
  if (output_channel >= output_channels) return;
  const int state = output_channel / 9;
  const int tap = output_channel - state * 9;
  float value = head_bias[output_channel];
  for (int ky = 0; ky < 3; ++ky) {
    const int y = pix / width + ky - 1;
    for (int kx = 0; kx < 3; ++kx) {
      const int x = pix - (pix / width) * width + kx - 1;
      if (y < 0 || y >= height || x < 0 || x >= width) continue;
      const int src_pix = y * width + x;
      const float in_r = load_state(latent + (batch * (2 * states) + 2 * state) * pixels + src_pix);
      const float in_i = load_state(latent + (batch * (2 * states) + 2 * state + 1) * pixels + src_pix);
      const int weight_base = (output_channel * 2) * 9 + ky * 3 + kx;
      value = fmaf(head_weight[weight_base], in_r, value);
      value = fmaf(head_weight[weight_base + 9], in_i, value);
    }
  }
  store_state(raw + (batch * output_channels + output_channel) * pixels + pix, value);
}

template <typename raw_t, typename output_t>
__global__ void q9_project_kernel(
    const raw_t* __restrict__ raw,
    output_t* __restrict__ ar,
    output_t* __restrict__ ai,
    int states,
    int height,
    int width,
    float rho_max,
    float epsilon) {
  const int plane = blockIdx.x;
  const int pixels = height * width;
  const int batch = plane / states;
  const int state = plane - batch * states;
  if (batch >= gridDim.x / states) return;
  for (int pix = threadIdx.x; pix < pixels; pix += blockDim.x) {
    const int base = (batch * (9 * states) + state * 9) * pixels + pix;
    const float center_logit = load_state(raw + base + 0 * pixels);
    const float axr = load_state(raw + base + 1 * pixels);
    const float axi = load_state(raw + base + 2 * pixels);
    const float ayr = load_state(raw + base + 3 * pixels);
    const float ayi = load_state(raw + base + 4 * pixels);
    const float d1r = load_state(raw + base + 5 * pixels);
    const float d1i = load_state(raw + base + 6 * pixels);
    const float d2r = load_state(raw + base + 7 * pixels);
    const float d2i = load_state(raw + base + 8 * pixels);
    const float m0 = sqrtf(axr * axr + axi * axi + epsilon);
    const float m1 = sqrtf(ayr * ayr + ayi * ayi + epsilon);
    const float m2 = sqrtf(d1r * d1r + d1i * d1i + epsilon);
    const float m3 = sqrtf(d2r * d2r + d2i * d2i + epsilon);
    const float mass = 2.0f * (m0 + m1 + m2 + m3);
    const float safe_rho = rho_max - 32.0f * epsilon;
    const float scale = 0.5f * (safe_rho - 0.10f) / (1.0f + mass);
    const float saxr = scale * axr, saxi = scale * axi;
    const float sayr = scale * ayr, sayi = scale * ayi;
    const float sd1r = scale * d1r, sd1i = scale * d1i;
    const float sd2r = scale * d2r, sd2i = scale * d2i;
    const float off_mass = scale * mass;
    float q = 1.0f / (1.0f + expf(-center_logit));
    q = fminf(fmaxf(q, epsilon), 1.0f - epsilon);
    const float center = 0.10f + off_mass + (safe_rho - 0.10f - 2.0f * off_mass) * q;
    const int out_base = (batch * 9 * states + 0 * states + state) * pixels + pix;
    store_state(ar + out_base, center);
    store_state(ar + (batch * 9 * states + 1 * states + state) * pixels + pix, sd1r);
    store_state(ar + (batch * 9 * states + 2 * states + state) * pixels + pix, sayr);
    store_state(ar + (batch * 9 * states + 3 * states + state) * pixels + pix, sd2r);
    store_state(ar + (batch * 9 * states + 4 * states + state) * pixels + pix, saxr);
    store_state(ar + (batch * 9 * states + 5 * states + state) * pixels + pix, saxr);
    store_state(ar + (batch * 9 * states + 6 * states + state) * pixels + pix, sd2r);
    store_state(ar + (batch * 9 * states + 7 * states + state) * pixels + pix, sayr);
    store_state(ar + (batch * 9 * states + 8 * states + state) * pixels + pix, sd1r);
    store_state(ai + (batch * 9 * states + 0 * states + state) * pixels + pix, 0.0f);
    store_state(ai + (batch * 9 * states + 1 * states + state) * pixels + pix, -sd1i);
    store_state(ai + (batch * 9 * states + 2 * states + state) * pixels + pix, -sayi);
    store_state(ai + (batch * 9 * states + 3 * states + state) * pixels + pix, -sd2i);
    store_state(ai + (batch * 9 * states + 4 * states + state) * pixels + pix, -saxi);
    store_state(ai + (batch * 9 * states + 5 * states + state) * pixels + pix, saxi);
    store_state(ai + (batch * 9 * states + 6 * states + state) * pixels + pix, sd2i);
    store_state(ai + (batch * 9 * states + 7 * states + state) * pixels + pix, sayi);
    store_state(ai + (batch * 9 * states + 8 * states + state) * pixels + pix, sd1i);
  }
}

template <typename state_t>
void launch_compiler(
    const torch::Tensor& packed,
    const torch::Tensor& m_real,
    const torch::Tensor& m_imag,
    const torch::Tensor& n_real,
    const torch::Tensor& n_imag,
    const torch::Tensor& gain_real,
    const torch::Tensor& gain_imag,
    const torch::Tensor& basis_y,
    const torch::Tensor& basis_x,
    const torch::Tensor& head_weight,
    const torch::Tensor& head_bias,
    torch::Tensor& raw,
    cudaStream_t stream) {
  const int B = packed.size(0);
  const int states = packed.size(1) / 2;
  const int H = packed.size(2);
  const int W = packed.size(3);
  const int pixels = H * W;
  auto mixed = torch::empty_like(packed);
  auto low_real = torch::empty({B, states, kModes}, packed.options().dtype(torch::kFloat));
  auto low_imag = torch::empty_like(low_real);
  auto conditioning = torch::empty_like(low_real);
  auto latent = torch::empty_like(packed);
  const dim3 mixing_grid(B * pixels);
  widely_linear_kernel<state_t><<<mixing_grid, 256, 0, stream>>>(
      packed.data_ptr<state_t>(), m_real.data_ptr<float>(), m_imag.data_ptr<float>(),
      n_real.data_ptr<float>(), n_imag.data_ptr<float>(), mixed.data_ptr<state_t>(),
      states, H, W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const std::size_t shared_bytes = static_cast<std::size_t>(22 * kProjectionThreads * sizeof(float));
  low3_projection_kernel<state_t><<<B * states, kProjectionThreads, shared_bytes, stream>>>(
      mixed.data_ptr<state_t>(), basis_y.data_ptr<float>(), basis_x.data_ptr<float>(),
      low_real.data_ptr<float>(), low_imag.data_ptr<float>(), conditioning.data_ptr<float>(),
      states, H, W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  latent_kernel<state_t><<<B * pixels, 256, 0, stream>>>(
      mixed.data_ptr<state_t>(), low_real.data_ptr<float>(), low_imag.data_ptr<float>(),
      conditioning.data_ptr<float>(), gain_real.data_ptr<float>(), gain_imag.data_ptr<float>(),
      basis_y.data_ptr<float>(), basis_x.data_ptr<float>(), latent.data_ptr<state_t>(),
      states, H, W, packed.scalar_type() == at::kBFloat16 ? 0.0078125f : std::numeric_limits<float>::epsilon());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  int head_threads = 256;
  while (head_threads < 9 * states && head_threads < kMaxHeadThreads) head_threads <<= 1;
  grouped_head_kernel<state_t><<<B * pixels, head_threads, 0, stream>>>(
      latent.data_ptr<state_t>(), head_weight.data_ptr<float>(), head_bias.data_ptr<float>(),
      raw.data_ptr<state_t>(), states, H, W);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename raw_t, typename output_t>
void launch_project(
    const torch::Tensor& raw,
    torch::Tensor& ar,
    torch::Tensor& ai,
    float rho_max,
    cudaStream_t stream) {
  const int B = raw.size(0);
  const int states = raw.size(1) / 9;
  const int H = raw.size(2);
  const int W = raw.size(3);
  // Projection is performed in FP32 even when the compiler stores its raw
  // grouped-H output in BF16.  Keep the stabilization constant on the same
  // FP32 boundary as ar/ai rather than inheriting BF16 machine epsilon.
  const float eps = std::numeric_limits<float>::epsilon();
  q9_project_kernel<raw_t, output_t><<<B * states, kProjectionThreads, 0, stream>>>(
      raw.data_ptr<raw_t>(), ar.data_ptr<output_t>(), ai.data_ptr<output_t>(), states, H, W, rho_max, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_tensor_cuda(const torch::Tensor& value, const char* name) {
  TORCH_CHECK(value.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(value.is_contiguous(), name, " must be contiguous");
}

void check_compiler_inputs(
    const torch::Tensor& packed,
    const torch::Tensor& m_real,
    const torch::Tensor& m_imag,
    const torch::Tensor& n_real,
    const torch::Tensor& n_imag,
    const torch::Tensor& gain_real,
    const torch::Tensor& gain_imag,
    const torch::Tensor& basis_y,
    const torch::Tensor& basis_x,
    const torch::Tensor& head_weight,
    const torch::Tensor& head_bias) {
  check_tensor_cuda(packed, "packed_f");
  check_tensor_cuda(m_real, "mix_m_real");
  check_tensor_cuda(m_imag, "mix_m_imag");
  check_tensor_cuda(n_real, "mix_n_real");
  check_tensor_cuda(n_imag, "mix_n_imag");
  check_tensor_cuda(gain_real, "gain_real");
  check_tensor_cuda(gain_imag, "gain_imag");
  check_tensor_cuda(basis_y, "basis_y");
  check_tensor_cuda(basis_x, "basis_x");
  check_tensor_cuda(head_weight, "head_weight");
  check_tensor_cuda(head_bias, "head_bias");
  TORCH_CHECK(packed.dim() == 4 && packed.size(1) % 2 == 0, "packed_f must be [B,2D,H,W]");
  const auto D = packed.size(1) / 2;
  TORCH_CHECK(packed.scalar_type() == at::kFloat || packed.scalar_type() == at::kBFloat16, "packed_f must be float32 or bfloat16");
  for (const auto& t : {m_real, m_imag, n_real, n_imag, gain_real, gain_imag, head_weight, head_bias, basis_y, basis_x}) {
    TORCH_CHECK(t.scalar_type() == at::kFloat, "compiler parameters and bases must be float32");
  }
  TORCH_CHECK(m_real.sizes() == torch::IntArrayRef({D, D}) && m_imag.sizes() == m_real.sizes(), "M shape mismatch");
  TORCH_CHECK(n_real.sizes() == m_real.sizes() && n_imag.sizes() == m_real.sizes(), "N shape mismatch");
  TORCH_CHECK(gain_real.sizes() == torch::IntArrayRef({D, 9}) && gain_imag.sizes() == gain_real.sizes(), "gain shape must be [D,9]");
  TORCH_CHECK(basis_y.sizes() == torch::IntArrayRef({3, packed.size(2)}) && basis_x.sizes() == torch::IntArrayRef({3, packed.size(3)}), "basis shape mismatch");
  TORCH_CHECK(head_weight.sizes() == torch::IntArrayRef({9 * D, 2, 3, 3}), "head_weight must be [9D,2,3,3]");
  TORCH_CHECK(head_bias.sizes() == torch::IntArrayRef({9 * D}), "head_bias must be [9D]");
}

}  // namespace

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
    torch::Tensor head_bias) {
  check_compiler_inputs(packed, m_real, m_imag, n_real, n_imag, gain_real, gain_imag, basis_y, basis_x, head_weight, head_bias);
  const auto guard = c10::cuda::CUDAGuard(packed.device());
  auto raw = torch::empty({packed.size(0), 9 * (packed.size(1) / 2), packed.size(2), packed.size(3)}, packed.options());
  auto stream = at::cuda::getCurrentCUDAStream(packed.device().index());
  if (packed.scalar_type() == at::kFloat) {
    launch_compiler<float>(packed, m_real, m_imag, n_real, n_imag, gain_real, gain_imag, basis_y, basis_x, head_weight, head_bias, raw, stream.stream());
  } else {
    launch_compiler<at::BFloat16>(packed, m_real, m_imag, n_real, n_imag, gain_real, gain_imag, basis_y, basis_x, head_weight, head_bias, raw, stream.stream());
  }
  return {raw};
}

std::vector<torch::Tensor> glrc9_q9_project_cuda(torch::Tensor raw, double rho_max) {
  check_tensor_cuda(raw, "raw");
  TORCH_CHECK(raw.dim() == 4 && raw.size(1) % 9 == 0, "raw must be [B,9D,H,W]");
  TORCH_CHECK(raw.scalar_type() == at::kFloat || raw.scalar_type() == at::kBFloat16, "raw must be float32 or bfloat16");
  TORCH_CHECK(rho_max > 0.0 && rho_max < 1.0, "rho_max must be in (0,1)");
  const auto guard = c10::cuda::CUDAGuard(raw.device());
  // The legalizer is the precision boundary of the compiler ABI: regardless
  // of packed/raw storage (FP32 or BF16), ar/ai are accumulated and exposed
  // as FP32 for the subsequent K-step propagation.
  auto ar = torch::empty(
      {raw.size(0), 9, raw.size(1) / 9, raw.size(2), raw.size(3)},
      raw.options().dtype(at::kFloat));
  auto ai = torch::empty_like(ar);
  auto stream = at::cuda::getCurrentCUDAStream(raw.device().index());
  if (raw.scalar_type() == at::kFloat) {
    launch_project<float, float>(raw, ar, ai, static_cast<float>(rho_max), stream.stream());
  } else {
    launch_project<at::BFloat16, float>(raw, ar, ai, static_cast<float>(rho_max), stream.stream());
  }
  return {ar, ai};
}



