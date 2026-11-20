"""Support-preserving diagonal half-delay; history stays in the autograd graph."""
import torch
import torch.nn.functional as F
from .fused_stencil import F9_OFFSETS, _shift_from_padded, _clone_graph_code, _tensor_compile_signatures

def diagonal_half_delay_step(pr, pi, previous_r, previous_i, dr, di, strength, ar, ai,
                             group_size, first=False, alpha=0.5):
    b, c, h, w = pr.shape
    shape = (b, c // group_size, group_size, h, w)
    rg, ig = pr.reshape(shape), pi.reshape(shape)
    sg = strength.reshape(shape).mean(dim=2, keepdim=True)
    energy = (rg.square() + ig.square()).mean(dim=2, keepdim=True)
    scale = torch.rsqrt(1.0 + sg * energy)
    source_r = dr + (rg * scale).reshape_as(pr)
    source_i = di + (ig * scale).reshape_as(pi)
    # At depth 1 the only available source is the current continuously injected drive.
    history_r = source_r if first else (1.0-alpha)*source_r + alpha*previous_r
    history_i = source_i if first else (1.0-alpha)*source_i + alpha*previous_i
    current_r, current_i = F.pad(source_r, (1,1,1,1)), F.pad(source_i, (1,1,1,1))
    delayed_r, delayed_i = F.pad(history_r, (1,1,1,1)), F.pad(history_i, (1,1,1,1))
    nr, ni = torch.zeros_like(pr), torch.zeros_like(pi)
    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        rr = delayed_r if tap in (1,3,6,8) else current_r
        ii = delayed_i if tap in (1,3,6,8) else current_i
        rr = _shift_from_padded(rr, dy, dx, h, w)
        ii = _shift_from_padded(ii, dy, dx, h, w)
        cr, ci = ar[:,tap], ai[:,tap]
        nr = nr + cr*rr - ci*ii
        ni = ni + cr*ii + ci*rr
    return nr, ni, source_r, source_i

_CACHE = {}
def compiled_diagonal_half_delay_step(pr, pi, previous_r, previous_i, dr, di, strength,
                                      ar, ai, group_size, first=False, alpha=0.5):
    tensors = (pr, pi, previous_r, previous_i, dr, di, strength, ar, ai)
    assert pr.is_cuda and all(t.device == pr.device for t in tensors)
    assert all(t.dtype in (torch.float32, torch.bfloat16) for t in tensors)
    assert all(t.shape == pr.shape for t in tensors[:7])
    assert ar.shape == ai.shape == (pr.shape[0], 9, *pr.shape[1:])
    assert group_size > 0 and pr.shape[1] % group_size == 0
    assert 0.0 <= alpha <= 1.0
    key = (int(group_size), bool(first), float(alpha), _tensor_compile_signatures(tensors))
    entry = _CACHE.get(key)
    if entry is None:
        target = _clone_graph_code(diagonal_half_delay_step,
                                  name=f'diagonal_half_delay_step__static_{len(_CACHE)}')
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _CACHE[key] = entry
    return entry[0](*tensors, int(group_size), bool(first), float(alpha))
