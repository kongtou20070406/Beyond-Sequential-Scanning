"""Adapter for the official VMamba SS2D token mixer."""

from torch import nn


def build(channels: int, *, stage_index: int = 0, block_index: int = 0) -> nn.Module:
    from clpo2027.vmamba_ss2d_adapter import build_official_ss2d_mixer

    return build_official_ss2d_mixer(
        channels=channels,
        stage_index=stage_index,
        block_index=block_index,
    )


__all__ = ["build"]
