"""Expanded depthwise convolution baseline."""

from torch import nn


def build(channels: int, *, inner_multiplier: float = 1.5, bias: bool = False) -> nn.Module:
    from clpo2027.mixers import ExpandedDWConv

    return ExpandedDWConv(channels, inner_multiplier=inner_multiplier, bias=bias)


__all__ = ["build"]
