"""Inference-only FP32 kernels for the unchanged five-step recurrence."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gate_drive(Pr, Pi, Strength, DriveR, DriveI, Zr, Zi,
                D: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group = tl.program_id(1)
    channel = tl.arange(0, 4)[:, None]
    offset = (group * 4 + channel) * N + spatial[None, :]
    valid = spatial[None, :] < N
    pr = tl.load(Pr + offset, mask=valid, other=0)
    pi = tl.load(Pi + offset, mask=valid, other=0)
    strength = tl.load(Strength + offset, mask=valid, other=0)
    mean_strength = tl.sum(strength, axis=0) * 0.25
    energy = tl.sum(pr * pr + pi * pi, axis=0) * 0.25
    scale = tl.rsqrt(1.0 + mean_strength * energy)
    drive_r = tl.load(DriveR + offset, mask=valid, other=0)
    drive_i = tl.load(DriveI + offset, mask=valid, other=0)
    tl.store(Zr + offset, drive_r + pr * scale[None, :], mask=valid)
    tl.store(Zi + offset, drive_i + pi * scale[None, :], mask=valid)


@triton.jit
def _stencil_increment(Zr, Zi, Ar, Ai, PrevR, PrevI, NextR, NextI,
                       Accum, Read, Weights,
                       B: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
                       STEP: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr,
                       BLOCK: tl.constexpr):
    n: tl.constexpr = H * W
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < B * D * n
    row = (index // W) % H
    col = index % W
    batch = index // (D * n)
    coefficient_base = batch * 9 * D * n + index % (D * n)
    nr = tl.full((BLOCK,), 0, tl.float32)
    ni = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(9):
        if tap == 0:
            dy = 0
            dx = 0
        else:
            square = tap - 1 if tap <= 4 else tap
            dy = square // 3 - 1
            dx = square % 3 - 1
        source_valid = valid & (row + dy >= 0) & (row + dy < H) & (col + dx >= 0) & (col + dx < W)
        source = index + dy * W + dx
        sr = tl.load(Zr + source, mask=source_valid, other=0)
        si = tl.load(Zi + source, mask=source_valid, other=0)
        ar = tl.load(Ar + coefficient_base + tap * D * n, mask=valid, other=0)
        ai = tl.load(Ai + coefficient_base + tap * D * n, mask=valid, other=0)
        nr = nr + ar * sr - ai * si
        ni = ni + ar * si + ai * sr
    real_offset = index + batch * D * n
    imag_offset = real_offset + D * n
    weight = tl.load(Weights + STEP)
    if FIRST:
        out_r = weight * nr
        out_i = weight * ni
    else:
        pr = tl.load(PrevR + index, mask=valid, other=0)
        pi = tl.load(PrevI + index, mask=valid, other=0)
        accum_r = tl.load(Accum + real_offset, mask=valid, other=0)
        accum_i = tl.load(Accum + imag_offset, mask=valid, other=0)
        out_r = accum_r + weight * (nr - pr)
        out_i = accum_i + weight * (ni - pi)
    if LAST:
        read = tl.load(Read + index, mask=valid, other=0)
        out_r = read * out_r
        out_i = read * out_i
    else:
        tl.store(NextR + index, nr, mask=valid)
        tl.store(NextI + index, ni, mask=valid)
    tl.store(Accum + real_offset, out_r, mask=valid)
    tl.store(Accum + imag_offset, out_i, mask=valid)


def direct_rollout(drive_r, drive_i, strength, ar, ai, weights, read, group_size):
    if torch.is_grad_enabled():
        raise RuntimeError('The direct rollout is inference-only')
    tensors = (drive_r, drive_i, strength, ar, ai, weights, read)
    if not all(t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError('Expected contiguous CUDA FP32 tensors')
    b, d, h, w = drive_r.shape
    if group_size != 4 or d % 4 or weights.shape != (5,):
        raise ValueError('Expected five updates and groups of four channels')
    if any(t.shape != drive_r.shape for t in (drive_i, strength, read)):
        raise ValueError('State field shapes differ')
    if ar.shape != (b, 9, d, h, w) or ai.shape != ar.shape:
        raise ValueError('Coefficient field shape differs')
    if any(t.device != drive_r.device for t in tensors):
        raise ValueError('All tensors must be on one device')
    state_r = [torch.empty_like(drive_r), torch.empty_like(drive_r)]
    state_i = [torch.empty_like(drive_i), torch.empty_like(drive_i)]
    zr = torch.empty_like(drive_r)
    zi = torch.empty_like(drive_i)
    packed = torch.empty((b, 2 * d, h, w), dtype=drive_r.dtype, device=drive_r.device)
    for step in range(5):
        previous = (step - 1) % 2
        current = step % 2
        if step:
            _gate_drive[(triton.cdiv(h * w, 128), b * d // 4)](
                state_r[previous], state_i[previous], strength, drive_r, drive_i,
                zr, zi, d, h * w, 128, num_warps=4, enable_fp_fusion=True)
            source_r, source_i = zr, zi
        else:
            source_r, source_i = drive_r, drive_i
        _stencil_increment[(triton.cdiv(b * d * h * w, 128),)](
            source_r, source_i, ar, ai, state_r[previous], state_i[previous],
            state_r[current], state_i[current], packed, read, weights,
            b, d, h, w, step, step == 0, step == 4, 128,
            num_warps=4, enable_fp_fusion=True)
    return packed
