"""Shared GoPro tiled inference and RGB metric implementation."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def tile_starts(length: int, tile: int, overlap: int) -> tuple[int, ...]:
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError("tile must be positive and overlap must lie in [0,tile)")
    if length <= tile:
        return (0,)
    last = length - tile
    starts = list(range(0, last + 1, tile - overlap))
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


@torch.inference_mode()
def tiled_inference(
    model: torch.nn.Module,
    image: Tensor,
    *,
    tile: int,
    overlap: int,
) -> Tensor:
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("tiled inference expects one BCHW image")
    original_height, original_width = image.shape[-2:]
    pad_height = max(0, tile - original_height)
    pad_width = max(0, tile - original_width)
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    height, width = image.shape[-2:]
    output = torch.zeros_like(image)
    weight = torch.zeros_like(image)
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            patch = image[..., y : y + tile, x : x + tile]
            prediction = model(patch)
            output[..., y : y + tile, x : x + tile].add_(prediction)
            weight[..., y : y + tile, x : x + tile].add_(1.0)
    if not bool(torch.all(weight > 0)):
        raise RuntimeError("tile stitching left uncovered pixels")
    return (output / weight)[..., :original_height, :original_width]


def positive_hann_window(
    tile: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a separable Hann window with strictly positive boundary weights."""

    if tile <= 0:
        raise ValueError("tile must be positive")
    vector = torch.hann_window(
        tile + 2,
        periodic=False,
        device=device,
        dtype=dtype,
    )[1:-1]
    window = torch.outer(vector, vector)
    return (window / window.max()).view(1, 1, tile, tile)


@torch.inference_mode()
def hann_tiled_inference(
    model: torch.nn.Module,
    image: Tensor,
    *,
    tile: int,
    overlap: int,
) -> Tensor:
    """Run tiled inference with normalized positive-Hann overlap blending."""

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Hann tiled inference expects one BCHW image")
    if overlap <= 0:
        raise ValueError("Hann tiled inference requires positive overlap")
    original_height, original_width = image.shape[-2:]
    pad_height = max(0, tile - original_height)
    pad_width = max(0, tile - original_width)
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    height, width = image.shape[-2:]
    # Stitch in FP32 even when the model itself runs in BF16/FP16.  Constructing
    # a Hann window directly in a reduced dtype can round its near-boundary
    # samples to zero, and low-precision overlap accumulation needlessly changes
    # the reported restoration metric.
    output = torch.zeros(
        image.shape,
        device=image.device,
        dtype=torch.float32,
    )
    weight = torch.zeros(
        1,
        1,
        height,
        width,
        device=image.device,
        dtype=torch.float32,
    )
    window = positive_hann_window(
        tile,
        device=image.device,
        dtype=torch.float32,
    )
    for y in tile_starts(height, tile, overlap):
        for x in tile_starts(width, tile, overlap):
            patch = image[..., y : y + tile, x : x + tile]
            prediction = model(patch).float()
            output[..., y : y + tile, x : x + tile].add_(
                prediction * window
            )
            weight[..., y : y + tile, x : x + tile].add_(window)
    if not bool(torch.all(weight > 0)):
        raise RuntimeError("Hann tile stitching left uncovered pixels")
    return (output / weight)[..., :original_height, :original_width]


@torch.inference_mode()
def halo_tiled_inference(
    model: torch.nn.Module,
    image: Tensor,
    *,
    core_tile: int,
    halo: int,
    overlap: int,
) -> Tensor:
    """Predict output cores from larger context tiles and blend core overlap."""

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("halo tiled inference expects one BCHW image")
    if core_tile <= 0 or halo <= 0:
        raise ValueError("core_tile and halo must be positive")
    if overlap < 0 or overlap >= core_tile:
        raise ValueError("overlap must lie in [0,core_tile)")
    original_height, original_width = image.shape[-2:]
    pad_height = max(0, core_tile - original_height)
    pad_width = max(0, core_tile - original_width)
    if pad_height or pad_width:
        image = F.pad(image, (0, pad_width, 0, pad_height), mode="reflect")
    height, width = image.shape[-2:]
    contextual = F.pad(
        image,
        (halo, halo, halo, halo),
        mode="constant",
        value=0.0,
    )
    output = torch.zeros(
        image.shape,
        device=image.device,
        dtype=torch.float32,
    )
    weight = torch.zeros(
        1,
        1,
        height,
        width,
        device=image.device,
        dtype=torch.float32,
    )
    expected_patch = core_tile + 2 * halo
    window = (
        positive_hann_window(
            core_tile,
            device=image.device,
            dtype=torch.float32,
        )
        if overlap > 0
        else torch.ones(
            1,
            1,
            core_tile,
            core_tile,
            device=image.device,
            dtype=torch.float32,
        )
    )
    for y in tile_starts(height, core_tile, overlap):
        for x in tile_starts(width, core_tile, overlap):
            patch = contextual[
                ...,
                y : y + expected_patch,
                x : x + expected_patch,
            ]
            if patch.shape[-2:] != (expected_patch, expected_patch):
                raise RuntimeError("halo extraction produced a partial patch")
            contextual_prediction = model(patch).float()
            core = contextual_prediction[
                ...,
                halo : halo + core_tile,
                halo : halo + core_tile,
            ]
            output[
                ...,
                y : y + core_tile,
                x : x + core_tile,
            ].add_(core * window)
            weight[
                ...,
                y : y + core_tile,
                x : x + core_tile,
            ].add_(window)
    if not bool(torch.all(weight > 0)):
        raise RuntimeError("halo tile stitching left uncovered pixels")
    return (output / weight)[..., :original_height, :original_width]


def rgb_psnr(prediction: Tensor, target: Tensor) -> float:
    mse = (prediction - target).square().mean()
    return float(-10.0 * torch.log10(mse.clamp_min(1.0e-12)).item())


def gaussian_kernel(
    channels: int,
    *,
    size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    coordinate = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    vector = torch.exp(-(coordinate.square()) / (2.0 * sigma**2))
    vector = vector / vector.sum()
    kernel = torch.outer(vector, vector)
    return kernel.expand(channels, 1, size, size).contiguous()


def rgb_ssim(prediction: Tensor, target: Tensor) -> float:
    channels = prediction.shape[1]
    kernel = gaussian_kernel(
        channels,
        size=11,
        sigma=1.5,
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mu_x = F.conv2d(prediction, kernel, groups=channels)
    mu_y = F.conv2d(target, kernel, groups=channels)
    sigma_x = F.conv2d(prediction.square(), kernel, groups=channels) - mu_x.square()
    sigma_y = F.conv2d(target.square(), kernel, groups=channels) - mu_y.square()
    sigma_xy = F.conv2d(prediction * target, kernel, groups=channels) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    value = (
        (2.0 * mu_x * mu_y + c1)
        * (2.0 * sigma_xy + c2)
        / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    )
    return float(value.mean().item())
