"""Restormer MDTA baseline token mixer."""

from torch import nn


def build(channels: int, *, stage_index: int = 0) -> nn.Module:
    from clpo2027.mixers import MDTA

    return MDTA(channels, num_heads=1 << stage_index, inner_multiplier=1.0, bias=False)


__all__ = ["build"]
