#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kTaps = 9;
constexpr int kThreads = 256;
#if CUDART_VERSION >= 13000
// CUDA 13.0's Blackwell toolchain can launch the large-register plane kernel
// with 1024 threads.  CUDA 12.8 builds of the same source can reject that
// launch with ``too many resources requested``; use the validated 512-thread
// fallback there without changing the recurrence or tensor contract.
constexpr int kLargePlaneThreads = 1024;
#else
constexpr int kLargePlaneThreads = 512;
#endif
// The temporal loop remains globally ordered through the per-plane barrier,
// while each step is evaluated through a 32x32 shared-memory spatial tile.
// One block still owns one (B,D) plane, so no cross-block synchronization is
// required and the custom op keeps the same ABI.
constexpr int kTile = 32;
constexpr int kSharedWidth = kTile + 2;
constexpr int kSharedElements = kSharedWidth * kSharedWidth;

int launch_threads(int64_t plane_size);

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

__device__ __forceinline__ int tap_dy(int tap) {
  return tap == 1 || tap == 2 || tap == 3 ? -1 : (tap >= 6 ? 1 : 0);
}

__device__ __forceinline__ int tap_dx(int tap) {
  return tap == 1 || tap == 4 || tap == 6 ? -1 : (tap == 3 || tap == 5 || tap == 8 ? 1 : 0);
}

template <typename input_t, typename work_t, typename coeff_t, typename endpoint_t, typename trajectory_t>
__global__ void f9_forward_kernel(
    const input_t* __restrict__ f,
    const coeff_t* __restrict__ ar,
    const coeff_t* __restrict__ ai,
    work_t* __restrict__ p_a_r,
    work_t* __restrict__ p_a_i,
    work_t* __restrict__ p_b_r,
    work_t* __restrict__ p_b_i,
    endpoint_t* __restrict__ endpoint,
    trajectory_t* __restrict__ trajectory,
    int channels,
    int height,
    int width,
    int plane_size,
    int total_planes,
    int steps) {
  // Correctness baseline: one block owns a complete (B,D) plane.  The
  // recurrence is globally coupled over the 3x3 neighborhood, so the
  // per-step __syncthreads is a legal temporal barrier rather than an
  // independent spatial-tile approximation.
  const int plane = blockIdx.x;
  if (plane >= total_planes) return;
  const int batch = plane / channels;
  const int batch_count = total_planes / channels;
  const int channel = plane - batch * channels;
  const int plane_base = plane * plane_size;
  const int f_batch_stride = 2 * channels * plane_size;
  const int f_real_base = batch * f_batch_stride + channel * plane_size;
  const int f_imag_base = f_real_base + channels * plane_size;
  const int coeff_base = (batch * kTaps * channels + channel) * plane_size;
  const int trajectory_batch_stride = 2 * channels * plane_size;

  for (int pix = threadIdx.x; pix < plane_size; pix += blockDim.x) {
    store_float(p_a_r + plane_base + pix, 0.0f);
    store_float(p_a_i + plane_base + pix, 0.0f);
  }
  __syncthreads();

  for (int step = 0; step < steps; ++step) {
    work_t* current_r = (step & 1) ? p_b_r + plane_base : p_a_r + plane_base;
    work_t* current_i = (step & 1) ? p_b_i + plane_base : p_a_i + plane_base;
    work_t* next_r = (step & 1) ? p_a_r + plane_base : p_b_r + plane_base;
    work_t* next_i = (step & 1) ? p_a_i + plane_base : p_b_i + plane_base;
    const int trajectory_base = step * (batch_count * trajectory_batch_stride) +
                                batch * trajectory_batch_stride + channel * plane_size;
    for (int pix = threadIdx.x; pix < plane_size; pix += blockDim.x) {
      const int y = pix / width;
      const int x = pix - y * width;
      const float f_r = load_float(f + f_real_base + pix);
      const float f_i = load_float(f + f_imag_base + pix);
      const float state_r = f_r + load_float(current_r + pix);
      const float state_i = f_i + load_float(current_i + pix);
      store_float(trajectory + trajectory_base + pix, state_r);
      store_float(trajectory + trajectory_base + channels * plane_size + pix, state_i);

      float out_r = 0.0f;
      float out_i = 0.0f;
      #pragma unroll
      for (int tap = 0; tap < kTaps; ++tap) {
        const int ny = y + tap_dy(tap);
        const int nx = x + tap_dx(tap);
        if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
        const int neighbor = ny * width + nx;
        const float z_r = load_float(f + f_real_base + neighbor) + load_float(current_r + neighbor);
        const float z_i = load_float(f + f_imag_base + neighbor) + load_float(current_i + neighbor);
        const int coeff_offset = coeff_base + tap * channels * plane_size + pix;
        const float a_r = load_float(ar + coeff_offset);
        const float a_i = load_float(ai + coeff_offset);
        out_r = fmaf(a_r, z_r, fmaf(-a_i, z_i, out_r));
        out_i = fmaf(a_r, z_i, fmaf(a_i, z_r, out_i));
      }
      store_float(next_r + pix, out_r);
      store_float(next_i + pix, out_i);
      if (step == steps - 1) {
        store_float(endpoint + f_real_base + pix, out_r);
        store_float(endpoint + f_imag_base + pix, out_i);
      }
    }
    __syncthreads();
  }
}

// Spatially tiled temporal propagation.  The block still owns a complete
// (B,D) plane, but each recurrence step loads one 3x3 halo into shared memory
// before evaluating a 16x16 tile.  The plane-wide barrier after the tile loop
// makes the global ping-pong state visible to the next recurrence step.  This
// is exact (unlike an independent-tile K-step shortcut) and keeps the same
// trajectory layout and floating-point boundary as f9_forward_kernel.
template <typename input_t, typename work_t, typename coeff_t, typename endpoint_t, typename trajectory_t>
__global__ void f9_forward_tiled_kernel(
    const input_t* __restrict__ f,
    const coeff_t* __restrict__ ar,
    const coeff_t* __restrict__ ai,
    work_t* __restrict__ p_a_r,
    work_t* __restrict__ p_a_i,
    work_t* __restrict__ p_b_r,
    work_t* __restrict__ p_b_i,
    endpoint_t* __restrict__ endpoint,
    trajectory_t* __restrict__ trajectory,
    int channels,
    int height,
    int width,
    int plane_size,
    int total_planes,
    int steps) {
  const int plane = blockIdx.x;
  if (plane >= total_planes) return;
  const int batch = plane / channels;
  const int batch_count = total_planes / channels;
  const int channel = plane - batch * channels;
  const int plane_base = plane * plane_size;
  const int f_batch_stride = 2 * channels * plane_size;
  const int f_real_base = batch * f_batch_stride + channel * plane_size;
  const int f_imag_base = f_real_base + channels * plane_size;
  const int coeff_base = (batch * kTaps * channels + channel) * plane_size;
  const int trajectory_batch_stride = 2 * channels * plane_size;
  const int tiles_x = (width + kTile - 1) / kTile;
  const int tiles_y = (height + kTile - 1) / kTile;
  const int tile_count = tiles_x * tiles_y;

  __shared__ float s_fr[kSharedElements];
  __shared__ float s_fi[kSharedElements];
  __shared__ float s_pr[kSharedElements];
  __shared__ float s_pi[kSharedElements];

  for (int step = 0; step < steps; ++step) {
    work_t* current_r = (step & 1) ? p_b_r + plane_base : p_a_r + plane_base;
    work_t* current_i = (step & 1) ? p_b_i + plane_base : p_a_i + plane_base;
    work_t* next_r = (step & 1) ? p_a_r + plane_base : p_b_r + plane_base;
    work_t* next_i = (step & 1) ? p_a_i + plane_base : p_b_i + plane_base;
    const int trajectory_base = step * (batch_count * trajectory_batch_stride) +
                                batch * trajectory_batch_stride + channel * plane_size;

    for (int tile = 0; tile < tile_count; ++tile) {
      const int tile_y = (tile / tiles_x) * kTile;
      const int tile_x = (tile - (tile / tiles_x) * tiles_x) * kTile;

      // Load f and the previous recurrent state once per halo element.  The
      // zero fill exactly matches the reference zero boundary.
      for (int sid = threadIdx.x; sid < kSharedElements; sid += blockDim.x) {
        const int ly = sid / kSharedWidth;
        const int lx = sid - ly * kSharedWidth;
        const int gy = tile_y + ly - 1;
        const int gx = tile_x + lx - 1;
        if (gy >= 0 && gy < height && gx >= 0 && gx < width) {
          const int gpix = gy * width + gx;
          s_fr[sid] = load_float(f + f_real_base + gpix);
          s_fi[sid] = load_float(f + f_imag_base + gpix);
          s_pr[sid] = load_float(current_r + gpix);
          s_pi[sid] = load_float(current_i + gpix);
        } else {
          s_fr[sid] = 0.0f;
          s_fi[sid] = 0.0f;
          s_pr[sid] = 0.0f;
          s_pi[sid] = 0.0f;
        }
      }
      __syncthreads();

      for (int local = threadIdx.x; local < kTile * kTile; local += blockDim.x) {
        const int ly = local / kTile;
        const int lx = local - ly * kTile;
        const int gy = tile_y + ly;
        const int gx = tile_x + lx;
        if (gy < height && gx < width) {
          const int pix = gy * width + gx;
          const int sid = (ly + 1) * kSharedWidth + (lx + 1);
          const float f_r = s_fr[sid];
          const float f_i = s_fi[sid];
          const float state_r = f_r + s_pr[sid];
          const float state_i = f_i + s_pi[sid];
          store_float(trajectory + trajectory_base + pix, state_r);
          store_float(trajectory + trajectory_base + channels * plane_size + pix, state_i);

          float out_r = 0.0f;
          float out_i = 0.0f;
          #pragma unroll
          for (int tap = 0; tap < kTaps; ++tap) {
            const int ny = gy + tap_dy(tap);
            const int nx = gx + tap_dx(tap);
            if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
            const int nsid = sid + tap_dy(tap) * kSharedWidth + tap_dx(tap);
            const float z_r = s_fr[nsid] + s_pr[nsid];
            const float z_i = s_fi[nsid] + s_pi[nsid];
            const int coeff_offset = coeff_base + tap * channels * plane_size + pix;
            const float a_r = load_float(ar + coeff_offset);
            const float a_i = load_float(ai + coeff_offset);
            out_r = fmaf(a_r, z_r, fmaf(-a_i, z_i, out_r));
            out_i = fmaf(a_r, z_i, fmaf(a_i, z_r, out_i));
          }
          store_float(next_r + pix, out_r);
          store_float(next_i + pix, out_i);
          if (step == steps - 1) {
            store_float(endpoint + f_real_base + pix, out_r);
            store_float(endpoint + f_imag_base + pix, out_i);
          }
        }
      }
      // The shared tile is reused by the next tile.  All global next-state
      // writes are complete before the plane-wide step barrier below.
      __syncthreads();
    }
    __syncthreads();
  }
}

template <typename q_t, typename work_t, typename coeff_t, typename trajectory_t, typename grad_coeff_t>
__global__ void f9_adjoint_kernel(
    const q_t* __restrict__ grad_endpoint,
    const coeff_t* __restrict__ ar,
    const coeff_t* __restrict__ ai,
    const trajectory_t* __restrict__ trajectory,
    work_t* __restrict__ adj_a,
    work_t* __restrict__ adj_b,
    work_t* __restrict__ grad_f,
    grad_coeff_t* __restrict__ grad_ar,
    grad_coeff_t* __restrict__ grad_ai,
    int channels,
    int height,
    int width,
    int plane_size,
    int total_planes,
    int steps) {
  const int plane = blockIdx.x;
  if (plane >= total_planes) return;
  const int batch = plane / channels;
  const int batch_count = total_planes / channels;
  const int channel = plane - batch * channels;
  const int f_batch_stride = 2 * channels * plane_size;
  const int real_base = batch * f_batch_stride + channel * plane_size;
  const int imag_base = real_base + channels * plane_size;
  const int coeff_base = (batch * kTaps * channels + channel) * plane_size;
  const int trajectory_batch_stride = 2 * channels * plane_size;
  for (int pix = threadIdx.x; pix < plane_size; pix += blockDim.x) {
    store_float(adj_a + real_base + pix, load_float(grad_endpoint + real_base + pix));
    store_float(adj_a + imag_base + pix, load_float(grad_endpoint + imag_base + pix));
  }
  __syncthreads();

  // The reverse-time barrier is outside the pixel loop.  Every thread must
  // execute each barrier, so state/coefficients are accumulated in output
  // tensors instead of per-thread arrays across time.
  for (int reverse = 0; reverse < steps; ++reverse) {
    const int step = steps - 1 - reverse;
    const int trajectory_base = step * (batch_count * trajectory_batch_stride) +
                                batch * trajectory_batch_stride + channel * plane_size;
    work_t* current_adj = (reverse & 1) ? adj_b : adj_a;
    work_t* next_adj = (reverse & 1) ? adj_a : adj_b;
    for (int pix = threadIdx.x; pix < plane_size; pix += blockDim.x) {
      const int y = pix / width;
      const int x = pix - y * width;
      const float current_r = load_float(current_adj + real_base + pix);
      const float current_i = load_float(current_adj + imag_base + pix);
      // Coefficient VJP at output pixel (y,x): current adjoint times
      // z_t(y+dy,x+dx), accumulated directly in the unique output element.
      #pragma unroll
      for (int tap = 0; tap < kTaps; ++tap) {
        const int ny = y + tap_dy(tap);
        const int nx = x + tap_dx(tap);
        if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
        const int neighbor = ny * width + nx;
        const float z_r = load_float(trajectory + trajectory_base + neighbor);
        const float z_i = load_float(trajectory + trajectory_base + channels * plane_size + neighbor);
        const int coeff_offset = coeff_base + tap * channels * plane_size + pix;
        const float old_ar = load_float(grad_ar + coeff_offset);
        const float old_ai = load_float(grad_ai + coeff_offset);
        store_float(grad_ar + coeff_offset, fmaf(current_r, z_r, fmaf(current_i, z_i, old_ar)));
        store_float(grad_ai + coeff_offset, fmaf(-current_r, z_i, fmaf(current_i, z_r, old_ai)));
      }

      // Transpose stencil at state pixel (y,x): q=(y-dy,x-dx).
      float next_r = 0.0f;
      float next_i = 0.0f;
      #pragma unroll
      for (int tap = 0; tap < kTaps; ++tap) {
        const int qy = y - tap_dy(tap);
        const int qx = x - tap_dx(tap);
        if (qy < 0 || qy >= height || qx < 0 || qx >= width) continue;
        const int q = qy * width + qx;
        const int coeff_offset = coeff_base + tap * channels * plane_size + q;
        const float g_r = load_float(current_adj + real_base + q);
        const float g_i = load_float(current_adj + imag_base + q);
        const float a_r = load_float(ar + coeff_offset);
        const float a_i = load_float(ai + coeff_offset);
        next_r = fmaf(a_r, g_r, fmaf(a_i, g_i, next_r));
        next_i = fmaf(-a_i, g_r, fmaf(a_r, g_i, next_i));
      }
      store_float(next_adj + real_base + pix, next_r);
      store_float(next_adj + imag_base + pix, next_i);
      const float old_f_r = load_float(grad_f + real_base + pix);
      const float old_f_i = load_float(grad_f + imag_base + pix);
      store_float(grad_f + real_base + pix, old_f_r + next_r);
      store_float(grad_f + imag_base + pix, old_f_i + next_i);
    }
    __syncthreads();
  }
}

// Exact reverse-time counterpart of f9_forward_tiled_kernel.  Each tile owns
// its coefficient-gradient outputs, while the shared one-cell halo supplies
// the transpose stencil and the saved BF16 trajectory.  The plane-wide
// barrier between reverse steps is the legal global ordering for ping-pong
// adjoints; no atomics or cross-block reductions are needed.
template <typename q_t, typename work_t, typename coeff_t, typename trajectory_t, typename grad_coeff_t>
__global__ void f9_adjoint_tiled_kernel(
    const q_t* __restrict__ grad_endpoint,
    const coeff_t* __restrict__ ar,
    const coeff_t* __restrict__ ai,
    const trajectory_t* __restrict__ trajectory,
    work_t* __restrict__ adj_a,
    work_t* __restrict__ adj_b,
    work_t* __restrict__ grad_f,
    grad_coeff_t* __restrict__ grad_ar,
    grad_coeff_t* __restrict__ grad_ai,
    int channels,
    int height,
    int width,
    int plane_size,
    int total_planes,
    int steps) {
  const int plane = blockIdx.x;
  if (plane >= total_planes) return;
  const int batch = plane / channels;
  const int batch_count = total_planes / channels;
  const int channel = plane - batch * channels;
  const int f_batch_stride = 2 * channels * plane_size;
  const int real_base = batch * f_batch_stride + channel * plane_size;
  const int imag_base = real_base + channels * plane_size;
  const int coeff_base = (batch * kTaps * channels + channel) * plane_size;
  const int trajectory_batch_stride = 2 * channels * plane_size;
  const int tiles_x = (width + kTile - 1) / kTile;
  const int tiles_y = (height + kTile - 1) / kTile;
  const int tile_count = tiles_x * tiles_y;

  __shared__ float s_ar[kSharedElements];
  __shared__ float s_ai[kSharedElements];
  __shared__ float s_zr[kSharedElements];
  __shared__ float s_zi[kSharedElements];

  for (int pix = threadIdx.x; pix < plane_size; pix += blockDim.x) {
    store_float(adj_a + real_base + pix, load_float(grad_endpoint + real_base + pix));
    store_float(adj_a + imag_base + pix, load_float(grad_endpoint + imag_base + pix));
  }
  __syncthreads();

  for (int reverse = 0; reverse < steps; ++reverse) {
    const int step = steps - 1 - reverse;
    const int trajectory_base = step * (batch_count * trajectory_batch_stride) +
                                batch * trajectory_batch_stride + channel * plane_size;
    work_t* current_adj = (reverse & 1) ? adj_b + real_base : adj_a + real_base;
    work_t* current_adj_i = (reverse & 1) ? adj_b + imag_base : adj_a + imag_base;
    work_t* next_adj = (reverse & 1) ? adj_a + real_base : adj_b + real_base;
    work_t* next_adj_i = (reverse & 1) ? adj_a + imag_base : adj_b + imag_base;

    for (int tile = 0; tile < tile_count; ++tile) {
      const int tile_y = (tile / tiles_x) * kTile;
      const int tile_x = (tile - (tile / tiles_x) * tiles_x) * kTile;
      for (int sid = threadIdx.x; sid < kSharedElements; sid += blockDim.x) {
        const int ly = sid / kSharedWidth;
        const int lx = sid - ly * kSharedWidth;
        const int gy = tile_y + ly - 1;
        const int gx = tile_x + lx - 1;
        if (gy >= 0 && gy < height && gx >= 0 && gx < width) {
          const int gpix = gy * width + gx;
          s_ar[sid] = load_float(current_adj + gpix);
          s_ai[sid] = load_float(current_adj_i + gpix);
          s_zr[sid] = load_float(trajectory + trajectory_base + gpix);
          s_zi[sid] = load_float(trajectory + trajectory_base + channels * plane_size + gpix);
        } else {
          s_ar[sid] = 0.0f;
          s_ai[sid] = 0.0f;
          s_zr[sid] = 0.0f;
          s_zi[sid] = 0.0f;
        }
      }
      __syncthreads();

      for (int local = threadIdx.x; local < kTile * kTile; local += blockDim.x) {
        const int ly = local / kTile;
        const int lx = local - ly * kTile;
        const int gy = tile_y + ly;
        const int gx = tile_x + lx;
        if (gy < height && gx < width) {
          const int pix = gy * width + gx;
          const int sid = (ly + 1) * kSharedWidth + (lx + 1);
          const float current_r = s_ar[sid];
          const float current_i = s_ai[sid];

          // The output coordinate (tap, pix) is owned by this tile, so its
          // temporal coefficient accumulation is race-free.
          #pragma unroll
          for (int tap = 0; tap < kTaps; ++tap) {
            const int ny = gy + tap_dy(tap);
            const int nx = gx + tap_dx(tap);
            if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
            const int nsid = sid + tap_dy(tap) * kSharedWidth + tap_dx(tap);
            const float z_r = s_zr[nsid];
            const float z_i = s_zi[nsid];
            const int coeff_offset = coeff_base + tap * channels * plane_size + pix;
            const float old_ar = load_float(grad_ar + coeff_offset);
            const float old_ai = load_float(grad_ai + coeff_offset);
            store_float(grad_ar + coeff_offset, fmaf(current_r, z_r, fmaf(current_i, z_i, old_ar)));
            store_float(grad_ai + coeff_offset, fmaf(-current_r, z_i, fmaf(current_i, z_r, old_ai)));
          }

          // Transpose stencil: the source coefficient lives at q=(y-dy,x-dx).
          float next_r = 0.0f;
          float next_i = 0.0f;
          #pragma unroll
          for (int tap = 0; tap < kTaps; ++tap) {
            const int qy = gy - tap_dy(tap);
            const int qx = gx - tap_dx(tap);
            if (qy < 0 || qy >= height || qx < 0 || qx >= width) continue;
            const int qsid = sid - tap_dy(tap) * kSharedWidth - tap_dx(tap);
            const int q = qy * width + qx;
            const int coeff_offset = coeff_base + tap * channels * plane_size + q;
            const float g_r = s_ar[qsid];
            const float g_i = s_ai[qsid];
            const float a_r = load_float(ar + coeff_offset);
            const float a_i = load_float(ai + coeff_offset);
            next_r = fmaf(a_r, g_r, fmaf(a_i, g_i, next_r));
            next_i = fmaf(-a_i, g_r, fmaf(a_r, g_i, next_i));
          }
          store_float(next_adj + pix, next_r);
          store_float(next_adj_i + pix, next_i);
          const float old_f_r = load_float(grad_f + real_base + pix);
          const float old_f_i = load_float(grad_f + imag_base + pix);
          store_float(grad_f + real_base + pix, old_f_r + next_r);
          store_float(grad_f + imag_base + pix, old_f_i + next_i);
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

template <typename input_t, typename work_t, typename coeff_t, typename endpoint_t, typename trajectory_t>
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
  const int64_t total_planes = batch * channels;
  // The endpoint options carry the recurrent work dtype for all supported
  // paths.  In the formal BF16-input/FP32-coefficient path this deliberately
  // allocates FP32 work buffers while the saved trajectory remains BF16.
  auto work_options = endpoint.options();
  auto p_a_r = torch::zeros({batch, channels, height, width}, work_options);
  auto p_a_i = torch::zeros({batch, channels, height, width}, work_options);
  auto p_b_r = torch::empty({batch, channels, height, width}, work_options);
  auto p_b_i = torch::empty({batch, channels, height, width}, work_options);
  f9_forward_kernel<input_t, work_t, coeff_t, endpoint_t, trajectory_t><<<total_planes, launch_threads(plane_size), 0, stream>>>(
      packed_f.data_ptr<input_t>(), ar.data_ptr<coeff_t>(), ai.data_ptr<coeff_t>(),
      p_a_r.data_ptr<work_t>(), p_a_i.data_ptr<work_t>(), p_b_r.data_ptr<work_t>(), p_b_i.data_ptr<work_t>(),
      endpoint.data_ptr<endpoint_t>(), trajectory.data_ptr<trajectory_t>(),
      channels, height, width, plane_size, total_planes, steps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t, typename work_t, typename coeff_t, typename endpoint_t, typename trajectory_t>
void launch_forward_tiled(
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
  const int64_t total_planes = batch * channels;
  auto work_options = endpoint.options();
  auto p_a_r = torch::zeros({batch, channels, height, width}, work_options);
  auto p_a_i = torch::zeros({batch, channels, height, width}, work_options);
  auto p_b_r = torch::empty({batch, channels, height, width}, work_options);
  auto p_b_i = torch::empty({batch, channels, height, width}, work_options);
  f9_forward_tiled_kernel<input_t, work_t, coeff_t, endpoint_t, trajectory_t><<<total_planes, launch_threads(plane_size), 0, stream>>>(
      packed_f.data_ptr<input_t>(), ar.data_ptr<coeff_t>(), ai.data_ptr<coeff_t>(),
      p_a_r.data_ptr<work_t>(), p_a_i.data_ptr<work_t>(), p_b_r.data_ptr<work_t>(), p_b_i.data_ptr<work_t>(),
      endpoint.data_ptr<endpoint_t>(), trajectory.data_ptr<trajectory_t>(),
      channels, height, width, plane_size, total_planes, steps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename q_t, typename work_t, typename coeff_t, typename trajectory_t, typename grad_coeff_t>
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
  const int64_t total_planes = batch * channels;
  // grad_f carries the recurrent work dtype; the formal mixed path therefore
  // uses FP32 adjoint state even though trajectory is stored as BF16.
  auto adj_a = torch::empty_like(grad_f);
  auto adj_b = torch::empty_like(grad_f);
  f9_adjoint_kernel<q_t, work_t, coeff_t, trajectory_t, grad_coeff_t><<<total_planes, launch_threads(plane_size), 0, stream>>>(
      grad_endpoint.data_ptr<q_t>(), ar.data_ptr<coeff_t>(), ai.data_ptr<coeff_t>(),
      trajectory.data_ptr<trajectory_t>(), adj_a.data_ptr<work_t>(), adj_b.data_ptr<work_t>(),
      grad_f.data_ptr<work_t>(), grad_ar.data_ptr<grad_coeff_t>(), grad_ai.data_ptr<grad_coeff_t>(),
      channels, height, width, plane_size, total_planes, steps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename q_t, typename work_t, typename coeff_t, typename trajectory_t, typename grad_coeff_t>
void launch_adjoint_tiled(
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
  const int64_t total_planes = batch * channels;
  auto adj_a = torch::empty_like(grad_f);
  auto adj_b = torch::empty_like(grad_f);
  f9_adjoint_tiled_kernel<q_t, work_t, coeff_t, trajectory_t, grad_coeff_t><<<total_planes, launch_threads(plane_size), 0, stream>>>(
      grad_endpoint.data_ptr<q_t>(), ar.data_ptr<coeff_t>(), ai.data_ptr<coeff_t>(),
      trajectory.data_ptr<trajectory_t>(), adj_a.data_ptr<work_t>(), adj_b.data_ptr<work_t>(),
      grad_f.data_ptr<work_t>(), grad_ar.data_ptr<grad_coeff_t>(), grad_ai.data_ptr<grad_coeff_t>(),
      channels, height, width, plane_size, total_planes, steps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool use_tiled_kernel() {
  const char* mode = std::getenv("GLRC9_F9_KERNEL");
  // The tiled kernel remains selectable for research benchmarks.  The plane
  // kernel is the default because the first Blackwell timing gate showed the
  // shared-halo prototype slower once adjoint traffic is included.
  return mode != nullptr && std::strcmp(mode, "tiled") == 0;
}

bool use_bf16_coeff_mixed() {
  const char* mode = std::getenv("GLRC9_F9_BF16_COEFF");
  // Opt-in candidate: keep BF16 coefficient traffic while retaining the
  // formal FP32 recurrent/endpoint/adjoint state boundary.
  return mode != nullptr &&
      (std::strcmp(mode, "1") == 0 || std::strcmp(mode, "true") == 0);
}

int launch_threads(int64_t plane_size) {
  const char* override_threads = std::getenv("GLRC9_F9_THREADS");
  if (override_threads != nullptr) {
    const int value = std::atoi(override_threads);
    if (value == 128 || value == 256 || value == 512 || value == 1024) return value;
  }
  // One block owns a complete plane.  Larger planes benefit from more
  // resident threads (fewer serial pixel chunks); the 14x14 tail does not.
  return plane_size >= 784 ? kLargePlaneThreads : kThreads;
}

void check_common(const torch::Tensor& packed_f, const torch::Tensor& ar, const torch::Tensor& ai) {
  TORCH_CHECK(packed_f.is_cuda() && ar.is_cuda() && ai.is_cuda(), "GLRC9 CUDA backend requires CUDA tensors");
  TORCH_CHECK(packed_f.is_contiguous() && ar.is_contiguous() && ai.is_contiguous(), "GLRC9 CUDA backend requires contiguous tensors");
  TORCH_CHECK(packed_f.scalar_type() == at::kFloat || packed_f.scalar_type() == at::kBFloat16, "state must use float32 or bfloat16");
  TORCH_CHECK(ar.scalar_type() == at::kFloat || ar.scalar_type() == at::kBFloat16, "coefficients must use float32 or bfloat16");
  TORCH_CHECK(ar.scalar_type() == ai.scalar_type(), "real and imaginary coefficients must use one dtype");
  TORCH_CHECK(packed_f.dim() == 4 && packed_f.size(1) % 2 == 0, "packed state must have shape [B,2D,H,W]");
  TORCH_CHECK(ar.dim() == 5 && ai.sizes() == ar.sizes() && ar.size(1) == 9, "coefficients must have shape [B,9,D,H,W]");
  TORCH_CHECK(ar.size(0) == packed_f.size(0) && ar.size(2) == packed_f.size(1) / 2 && ar.size(3) == packed_f.size(2) && ar.size(4) == packed_f.size(3), "coefficient/state shapes do not match");
}

}  // namespace

std::vector<torch::Tensor> glrc9_f9_forward_cuda(
    torch::Tensor packed_f, torch::Tensor ar, torch::Tensor ai, int64_t steps) {
  check_common(packed_f, ar, ai);
  TORCH_CHECK(steps > 0 && steps <= 1024, "steps must be in [1,1024]");
  const auto guard = c10::cuda::CUDAGuard(packed_f.device());
  // The production classification path uses BF16 packed input with FP32
  // coefficients. Keep the recurrent state in FP32, expose an FP32 endpoint,
  // and only compress the saved trajectory to BF16 for the adjoint interface.
  const bool exact_mixed = packed_f.scalar_type() == at::kBFloat16 && ar.scalar_type() == at::kFloat;
  const bool bf16_coeff_mixed = packed_f.scalar_type() == at::kBFloat16 &&
      ar.scalar_type() == at::kBFloat16 && use_bf16_coeff_mixed();
  auto endpoint = (exact_mixed || bf16_coeff_mixed)
      ? torch::empty(packed_f.sizes(), packed_f.options().dtype(at::kFloat))
      : torch::empty_like(packed_f);
  auto trajectory = torch::empty(
      {steps, packed_f.size(0), packed_f.size(1), packed_f.size(2), packed_f.size(3)},
      packed_f.options());
  auto stream = at::cuda::getCurrentCUDAStream(packed_f.device().index());
  const bool tiled = use_tiled_kernel();
  if (packed_f.scalar_type() == at::kFloat && ar.scalar_type() == at::kFloat) {
    if (tiled) launch_forward_tiled<float, float, float, float, float>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
    else launch_forward<float, float, float, float, float>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  } else if (packed_f.scalar_type() == at::kBFloat16 && ar.scalar_type() == at::kFloat) {
    if (tiled) launch_forward_tiled<at::BFloat16, float, float, float, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
    else launch_forward<at::BFloat16, float, float, float, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  } else if (bf16_coeff_mixed) {
    if (tiled) launch_forward_tiled<at::BFloat16, float, at::BFloat16, float, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
    else launch_forward<at::BFloat16, float, at::BFloat16, float, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  } else if (packed_f.scalar_type() == at::kBFloat16) {
    if (tiled) launch_forward_tiled<at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
    else launch_forward<at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  } else {
    if (tiled) launch_forward_tiled<float, float, at::BFloat16, float, float>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
    else launch_forward<float, float, at::BFloat16, float, float>(packed_f, ar, ai, endpoint, trajectory, steps, stream.stream());
  }
  return {endpoint, trajectory};
}

std::vector<torch::Tensor> glrc9_f9_adjoint_cuda(
    torch::Tensor grad_endpoint, torch::Tensor ar, torch::Tensor ai, torch::Tensor trajectory, int64_t steps) {
  check_common(grad_endpoint, ar, ai);
  TORCH_CHECK(trajectory.is_cuda() && trajectory.is_contiguous(), "trajectory must be a contiguous CUDA tensor");
  TORCH_CHECK(trajectory.scalar_type() == at::kFloat || trajectory.scalar_type() == at::kBFloat16, "trajectory must use float32 or bfloat16");
  TORCH_CHECK(trajectory.device() == grad_endpoint.device(), "trajectory and endpoint gradient must be on the same device");
  TORCH_CHECK(trajectory.sizes() == torch::IntArrayRef({steps, grad_endpoint.size(0), grad_endpoint.size(1), grad_endpoint.size(2), grad_endpoint.size(3)}), "trajectory shape mismatch");
  const auto guard = c10::cuda::CUDAGuard(grad_endpoint.device());
  auto grad_f = torch::zeros_like(grad_endpoint);
  auto grad_ar = torch::zeros_like(ar);
  auto grad_ai = torch::zeros_like(ai);
  auto stream = at::cuda::getCurrentCUDAStream(grad_endpoint.device().index());
  const bool tiled = use_tiled_kernel();
  const bool bf16_coeff_mixed = grad_endpoint.scalar_type() == at::kFloat &&
      ar.scalar_type() == at::kBFloat16 && use_bf16_coeff_mixed();
  if (grad_endpoint.scalar_type() == at::kFloat && ar.scalar_type() == at::kFloat) {
    if (trajectory.scalar_type() == at::kBFloat16) {
      // Formal mixed path: q, recurrence work, and all VJPs are FP32 while
      // the saved trajectory is read back from BF16.
      if (tiled) launch_adjoint_tiled<float, float, float, at::BFloat16, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
      else launch_adjoint<float, float, float, at::BFloat16, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    } else {
      if (tiled) launch_adjoint_tiled<float, float, float, float, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
      else launch_adjoint<float, float, float, float, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    }
  } else if (grad_endpoint.scalar_type() == at::kBFloat16 && ar.scalar_type() == at::kFloat) {
    TORCH_CHECK(trajectory.scalar_type() == at::kBFloat16, "BF16 endpoint with FP32 coefficients requires BF16 trajectory");
    if (tiled) launch_adjoint_tiled<at::BFloat16, at::BFloat16, float, at::BFloat16, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    else launch_adjoint<at::BFloat16, at::BFloat16, float, at::BFloat16, float>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
  } else if (bf16_coeff_mixed) {
    TORCH_CHECK(trajectory.scalar_type() == at::kBFloat16, "BF16 coefficients in mixed mode require BF16 trajectory");
    if (tiled) launch_adjoint_tiled<float, float, at::BFloat16, at::BFloat16, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    else launch_adjoint<float, float, at::BFloat16, at::BFloat16, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
  } else if (grad_endpoint.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(trajectory.scalar_type() == at::kBFloat16, "BF16 endpoint requires BF16 trajectory");
    if (tiled) launch_adjoint_tiled<at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    else launch_adjoint<at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
  } else {
    TORCH_CHECK(trajectory.scalar_type() == at::kFloat, "FP32 endpoint with BF16 coefficients requires FP32 trajectory");
    if (tiled) launch_adjoint_tiled<float, float, at::BFloat16, float, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
    else launch_adjoint<float, float, at::BFloat16, float, at::BFloat16>(grad_endpoint, ar, ai, trajectory, grad_f, grad_ar, grad_ai, steps, stream.stream());
  }
  return {grad_f, grad_ar, grad_ai};
}
