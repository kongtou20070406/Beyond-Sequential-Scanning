"""Generate the fixed FP32 propagation coefficients and controls."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _joint_head(Hidden, CoeffWeight, CoeffBias, ControlWeight, ControlBias,
                Ar, Ai, Write, Read, Strength,
                D: tl.constexpr, N: tl.constexpr, RHO: tl.constexpr,
                ALPHA: tl.constexpr, BLOCK: tl.constexpr):
    spatial = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bc = tl.program_id(1)
    channel = bc % D
    batch = bc // D
    tap = tl.arange(0, 16)
    coeff_row = (channel // 4) * 72 + tap * 4 + channel % 4
    control_row = (channel // 4) * 12 + channel % 4
    rr = tl.load(CoeffBias + coeff_row, mask=tap < 9, other=0)[:, None]
    ii = tl.load(CoeffBias + coeff_row + 36, mask=tap < 9, other=0)[:, None]
    write_raw = tl.full((BLOCK,), 0, tl.float32) + tl.load(ControlBias + control_row)
    read_raw = tl.full((BLOCK,), 0, tl.float32) + tl.load(ControlBias + control_row + 4)
    strength_raw = tl.full((BLOCK,), 0, tl.float32) + tl.load(ControlBias + control_row + 8)
    for j in tl.static_range(4):
        value = tl.load(Hidden + (batch * D + (channel // 4) * 4 + j) * N + spatial,
                        mask=spatial < N, other=0)
        wr = tl.load(CoeffWeight + coeff_row * 4 + j, mask=tap < 9, other=0)[:, None]
        wi = tl.load(CoeffWeight + (coeff_row + 36) * 4 + j, mask=tap < 9, other=0)[:, None]
        rr = rr + value[None, :] * wr
        ii = ii + value[None, :] * wi
        write_raw = write_raw + value * tl.load(ControlWeight + control_row * 4 + j)
        read_raw = read_raw + value * tl.load(ControlWeight + (control_row + 4) * 4 + j)
        strength_raw = strength_raw + value * tl.load(ControlWeight + (control_row + 8) * 4 + j)
    magnitude = tl.sqrt(rr * rr + ii * ii)
    mass = tl.sum(magnitude, axis=0)
    scale = RHO / (1.0 + mass)
    coefficient_offset = ((batch * 9 + tap[:, None]) * D + channel) * N + spatial[None, :]
    coefficient_mask = (tap[:, None] < 9) & (spatial[None, :] < N)
    tl.store(Ar + coefficient_offset, rr * scale[None, :], mask=coefficient_mask)
    tl.store(Ai + coefficient_offset, ii * scale[None, :], mask=coefficient_mask)
    write = 1.0 + ALPHA * libdevice.tanh(write_raw)
    read = 1.0 + ALPHA * libdevice.tanh(read_raw)
    strength = tl.where(strength_raw > 20.0, strength_raw, libdevice.log1p(tl.exp(strength_raw)))
    control_offset = (batch * D + channel) * N + spatial
    tl.store(Write + control_offset, write, mask=spatial < N)
    tl.store(Read + control_offset, read, mask=spatial < N)
    tl.store(Strength + control_offset, strength, mask=spatial < N)


def fused_heads(operator, hidden):
    if torch.is_grad_enabled():
        raise RuntimeError('Joint heads are inference-only')
    cw, cb = operator.coeff_out.weight, operator.coeff_out.bias
    sw, sb = operator.selective_head.weight, operator.selective_head.bias
    if hidden.ndim != 4 or cb is None or sb is None:
        raise ValueError('Expected an NCHW hidden field and biased projection heads')
    tensors = (hidden, cw, cb, sw, sb)
    if not all(t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in tensors):
        raise ValueError('Expected contiguous CUDA FP32 tensors')
    b, d, h, w = hidden.shape
    if (operator.tap_count != 9 or operator.selective_group_size != 4
            or operator.state_gate != 'radial' or d != operator.state_channels or d % 4
            or cw.shape != (18*d, 4, 1, 1) or sw.shape != (3*d, 4, 1, 1)):
        raise ValueError('The joint heads require the frozen nine-tap radial group-four layout')
    if any(t.device != hidden.device for t in tensors):
        raise ValueError('All head tensors must be on the same device')
    ar = torch.empty((b, 9, d, h, w), dtype=hidden.dtype, device=hidden.device)
    ai = torch.empty_like(ar)
    write, read, strength = [torch.empty_like(hidden) for _ in range(3)]
    _joint_head[(triton.cdiv(h*w, 128), b*d)](
        hidden, cw, cb, sw, sb, ar, ai, write, read, strength,
        d, h*w, float(operator.rho_max), float(operator.selective_gate_range), 128,
        num_warps=4, enable_fp_fusion=True)
    return ar, ai, write, read, strength
