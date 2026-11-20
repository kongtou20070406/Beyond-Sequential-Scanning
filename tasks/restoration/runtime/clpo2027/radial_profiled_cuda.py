"""Blackwell CUDA backend for five radial-profiled propagation updates."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import sysconfig
from typing import Any

import torch


_EXT: Any | None = None


def _source_paths() -> list[str]:
    root = Path(__file__).resolve().parent
    return [
        str(root / "csrc" / "radial_profiled_bindings.cpp"),
        str(root / "csrc" / "radial_profiled_cuda.cu"),
    ]


def load_extension(*, verbose: bool = False) -> Any:
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError("RadialProfiled CUDA backend requires CUDA")
    from torch.utils.cpp_extension import load

    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    cuda_env_root = Path(sys.prefix) / "targets" / "x86_64-linux"
    cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_120,code=sm_120",
    ]
    ldflags: list[str] = []
    if (cuda_env_root / "include").is_dir():
        cuda_flags.append(f"-I{cuda_env_root / 'include'}")
    if (cuda_env_root / "lib").is_dir():
        ldflags.append(f"-L{cuda_env_root / 'lib'}")
    nvidia_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for include in sorted(nvidia_root.glob("*/include")):
        cuda_flags.append(f"-I{include}")
    for library in sorted(nvidia_root.glob("*/lib")):
        ldflags.append(f"-L{library}")
    _EXT = load(
        name="radial_profiled_k6_cuda",
        sources=_source_paths(),
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags,
        extra_ldflags=ldflags,
        with_cuda=True,
        verbose=verbose,
    )
    return _EXT


class _RadialProfiledCUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        drive_r: torch.Tensor,
        drive_i: torch.Tensor,
        strength: torch.Tensor,
        ar: torch.Tensor,
        ai: torch.Tensor,
        weights: torch.Tensor,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fast_dtype = torch.bfloat16
        inputs = tuple(
            value.contiguous().to(dtype=fast_dtype)
            for value in (drive_r, drive_i, strength, ar, ai, weights)
        )
        profiled_r, profiled_i, states = load_extension().forward(
            *inputs, int(group_size)
        )
        ctx.save_for_backward(*inputs, states)
        ctx.group_size = int(group_size)
        ctx.original_dtypes = tuple(value.dtype for value in (drive_r, drive_i, strength, ar, ai, weights))
        return profiled_r, profiled_i

    @staticmethod
    def backward(ctx, grad_r: torch.Tensor, grad_i: torch.Tensor):
        drive_r, drive_i, strength, ar, ai, weights, states = ctx.saved_tensors
        grads = load_extension().backward(
            grad_r.contiguous().to(torch.bfloat16),
            grad_i.contiguous().to(torch.bfloat16),
            drive_r,
            drive_i,
            strength,
            ar,
            ai,
            weights,
            states,
            ctx.group_size,
        )
        restored = tuple(
            grad.to(dtype=dtype) for grad, dtype in zip(grads, ctx.original_dtypes, strict=True)
        )
        return (*restored, None)


class _RadialCoefficientsCUDA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        rho_max: float,
        group_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = tuple(
            value.contiguous().to(dtype=torch.bfloat16)
            for value in (hidden, weight, bias)
        )
        ar, ai = load_extension().coeff_forward(
            *inputs, float(rho_max), int(group_size)
        )
        ctx.save_for_backward(*inputs)
        ctx.rho_max = float(rho_max)
        ctx.group_size = int(group_size)
        ctx.original_dtypes = tuple(value.dtype for value in (hidden, weight, bias))
        return ar, ai

    @staticmethod
    def backward(ctx, grad_ar: torch.Tensor, grad_ai: torch.Tensor):
        hidden, weight, bias = ctx.saved_tensors
        grad_raw = load_extension().coeff_backward(
            grad_ar.contiguous().to(torch.bfloat16),
            grad_ai.contiguous().to(torch.bfloat16),
            hidden,
            weight,
            bias,
            ctx.rho_max,
            ctx.group_size,
        )
        batch, channels, height, width = hidden.shape
        groups = channels // ctx.group_size
        samples = batch * height * width
        hidden_g = hidden.view(
            batch, groups, ctx.group_size, height, width
        ).permute(1, 0, 3, 4, 2).reshape(groups, samples, ctx.group_size)
        grad_g = grad_raw.view(
            batch, groups, 18 * ctx.group_size, height, width
        ).permute(1, 0, 3, 4, 2).reshape(
            groups, samples, 18 * ctx.group_size
        )
        weight_g = weight.view(groups, 18 * ctx.group_size, ctx.group_size)
        grad_hidden = torch.bmm(grad_g, weight_g).reshape(
            groups, batch, height, width, ctx.group_size
        ).permute(1, 0, 4, 2, 3).reshape_as(hidden)
        grad_weight = torch.bmm(grad_g.transpose(1, 2), hidden_g).reshape_as(weight)
        grad_bias = grad_g.sum(dim=1).reshape_as(bias)
        grads = (grad_hidden, grad_weight, grad_bias)
        restored = tuple(
            grad.to(dtype=dtype)
            for grad, dtype in zip(grads, ctx.original_dtypes, strict=True)
        )
        return (*restored, None, None)


def radial_profiled_cuda(
    drive_r: torch.Tensor,
    drive_i: torch.Tensor,
    strength: torch.Tensor,
    ar: torch.Tensor,
    ai: torch.Tensor,
    weights: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not drive_r.is_cuda:
        raise RuntimeError("RadialProfiled CUDA backend requires CUDA tensors")
    return _RadialProfiledCUDA.apply(
        drive_r, drive_i, strength, ar, ai, weights, int(group_size)
    )


def radial_coefficients_cuda(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    rho_max: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _RadialCoefficientsCUDA.apply(
        hidden, weight, bias, float(rho_max), int(group_size)
    )


__all__ = ["load_extension", "radial_coefficients_cuda", "radial_profiled_cuda"]
