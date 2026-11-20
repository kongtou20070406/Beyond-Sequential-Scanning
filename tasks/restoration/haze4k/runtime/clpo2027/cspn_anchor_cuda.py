"""Lazy CUDA bridge for CSPN's initial-state-anchored recurrence."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor


@lru_cache(maxsize=1)
def _extension():
    from torch.utils.cpp_extension import load

    return load(
        name="clpo_cspn_anchor_cuda_v1",
        sources=[
            str(Path(__file__).with_name("csrc") / "cspn_anchor_cuda.cpp"),
            str(Path(__file__).with_name("csrc") / "cspn_anchor_cuda.cu"),
        ],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=False,
    )


class _CSPNAnchorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial: Tensor, center: Tensor, off: Tensor, steps: int):
        states = _extension().forward(
            initial.contiguous(), center.contiguous(), off.contiguous(), steps
        )
        ctx.save_for_backward(states, center, off)
        return states[-1]

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        states, center, off = ctx.saved_tensors
        grad_initial, grad_center, grad_off = _extension().backward(
            grad_out.contiguous(), states, center, off
        )
        return grad_initial, grad_center, grad_off, None


def propagate(initial: Tensor, center: Tensor, off: Tensor, steps: int) -> Tensor:
    return _CSPNAnchorFunction.apply(
        initial,
        center.to(dtype=initial.dtype),
        off.to(dtype=initial.dtype),
        steps,
    )
