"""Thin adapter for the official MaIR VMM token mixer.

Source: XLearning-SCU/2025-CVPR-MaIR, commit
07d58af10175f52bba0bb6f82edba1ea5d891e12.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn


class _SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, oflex
    ):
        import selective_scan_cuda_oflex

        out, x, *_ = selective_scan_cuda_oflex.fwd(
            u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, oflex
        )
        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout):
        import selective_scan_cuda_oflex

        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        grads = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (*grads[:7], None, None, None, None)


def _selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
):
    if z is not None or return_last_state:
        raise NotImplementedError("MaIR VMM uses neither z nor return_last_state")
    return _SelectiveScanOflex.apply(
        u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True
    )


def _load_official_mair():
    project_root = Path(__file__).resolve().parents[1]
    arch_dir = project_root / "third_party" / "2025-CVPR-MaIR" / "basicsr" / "archs"
    source = arch_dir / "mair_arch.py"
    if not source.is_file():
        raise FileNotFoundError(f"official MaIR source not found: {source}")

    names = ("mamba_ssm", "mamba_ssm.ops", "mamba_ssm.ops.selective_scan_interface")
    previous = {name: sys.modules.get(name) for name in names}
    mamba = types.ModuleType(names[0])
    ops = types.ModuleType(names[1])
    interface = types.ModuleType(names[2])
    interface.selective_scan_fn = _selective_scan_fn
    interface.selective_scan_ref = _selective_scan_fn
    mamba.ops = ops
    ops.selective_scan_interface = interface
    sys.modules.update({names[0]: mamba, names[1]: ops, names[2]: interface})
    sys.path.insert(0, str(arch_dir))
    try:
        spec = importlib.util.spec_from_file_location("_official_mair_arch", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(arch_dir))
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


_MAIR = _load_official_mair()


def _vectorized_ids_scan(inp, ids, bkdl=False, K=4):
    B, C, H, W = inp.shape
    L = H * W
    index = ids.reshape(1, K, 1, L).expand(B, K, C, L)
    out = torch.gather(inp.reshape(B, 1, C, L).expand(B, K, C, L), -1, index)
    return out if bkdl else out.reshape(B, K, C * L)


def _vectorized_ids_inverse(inp, ids, shape=None):
    B, K, C, L = inp.shape
    index = ids.reshape(1, K, 1, L).expand(B, K, C, L)
    out = torch.gather(inp, -1, index)
    if shape is None:
        return out.reshape(B, K * C, L)
    _, _, H, W = shape
    return out.reshape(B, K * C, H, W)


_MAIR.mair_ids_scan = _vectorized_ids_scan
_MAIR.mair_ids_inverse = _vectorized_ids_inverse
_ID_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


class MaIRVMMMixer(nn.Module):
    """Official VMM in a BCHW-in/BCHW-out VIVNet mixer slot."""

    def __init__(self, channels: int, *, d_state: int, block_index: int, scan_len: int = 4):
        super().__init__()
        self.scan_len = scan_len
        self.shift = 0 if block_index % 2 == 0 else scan_len // 2
        self.vmm = _MAIR.VMM(d_model=channels, d_state=d_state, expand=2.0)

    def _ids(self, x: torch.Tensor):
        _, _, h, w = x.shape
        key = (h, w, self.scan_len, self.shift, x.device.type, x.device.index)
        if key not in _ID_CACHE:
            generate = (
                _MAIR.mair_ids_generate if self.shift == 0 else _MAIR.mair_shift_ids_generate
            )
            kwargs = {"inp_shape": (1, 1, h, w), "scan_len": self.scan_len}
            if self.shift:
                kwargs["shift_len"] = self.shift
            ids = generate(**kwargs)
            _ID_CACHE[key] = tuple(index.to(device=x.device) for index in ids)
        return _ID_CACHE[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.vmm(x.permute(0, 2, 3, 1).contiguous(), self._ids(x))
        return y.permute(0, 3, 1, 2).contiguous()


def build_official_mair_mixer(
    channels: int, *, stage_index: int, block_index: int
) -> nn.Module:
    d_state = (4, 8, 16, 16, 8, 4, 4)[stage_index]
    return MaIRVMMMixer(channels, d_state=d_state, block_index=block_index)

