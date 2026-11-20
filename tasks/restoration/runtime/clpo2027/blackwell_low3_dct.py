"""Exact low-3x3 DCT projection/reconstruction used by GLRC9."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_BASIS_CACHE: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def _basis3(
    length: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (length, device.type, device.index, dtype)
    basis = _BASIS_CACHE.get(key)
    if basis is None:
        n = torch.arange(length, device=device, dtype=dtype)
        u = torch.arange(3, device=device, dtype=dtype)[:, None]
        scale = torch.full(
            (3, 1), (2.0 / length) ** 0.5, device=device, dtype=dtype
        )
        scale[0] = (1.0 / length) ** 0.5
        basis = scale * torch.cos(torch.pi * (n[None] + 0.5) * u / length)
        _BASIS_CACHE[key] = basis
    return basis


def direct_low3_dct(value: torch.Tensor) -> torch.Tensor:
    """Return the exact low 3x3 DCT coefficients in the legacy full shape."""

    height, width = value.shape[-2:]
    working_dtype = (
        torch.float32
        if value.dtype in {torch.float16, torch.bfloat16}
        else value.dtype
    )
    by = _basis3(height, value.device, working_dtype)
    bx = _basis3(width, value.device, working_dtype)
    reduced_x = torch.einsum("...yx,vx->...yv", value.to(working_dtype), bx)
    low = torch.einsum("...yv,uy->...uv", reduced_x, by)
    return F.pad(low, (0, width - 3, 0, height - 3))


def direct_low3_idct(
    value: torch.Tensor, *, output_dtype: torch.dtype
) -> torch.Tensor:
    """Reconstruct an exactly 3x3-supported spectrum without a full IDCT."""

    height, width = value.shape[-2:]
    by = _basis3(height, value.device, value.dtype)
    bx = _basis3(width, value.device, value.dtype)
    expanded_y = torch.einsum("...uv,uy->...yv", value[..., :3, :3], by)
    return torch.einsum("...yv,vx->...yx", expanded_y, bx).to(output_dtype)
