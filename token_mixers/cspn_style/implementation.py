"""CSPN-style baseline token mixer."""

from torch import nn


def build(channels: int, *, propagation_steps: int = 5) -> nn.Module:
    from clpo2027.mixers import CSPNStyleMixer

    return CSPNStyleMixer(channels, steps=propagation_steps)


__all__ = ["build"]
