"""Build token mixers for carrier models.

Carrier code imports this module. Each mixer has its own subdirectory.
"""

from __future__ import annotations

from torch import nn


_MIXERS = ("native", "cspn_style", "dwconv", "mdta", "swin", "ss2d", "ours")


def mixer_names() -> tuple[str, ...]:
    return _MIXERS


def build_token_mixer(
    name: str,
    channels: int,
    *,
    stage_index: int = 0,
    block_index: int = 0,
    propagation_steps: int = 5,
    inner_multiplier: float | None = None,
    bias: bool = False,
) -> nn.Module:
    """Build a BCHW-to-BCHW token mixer from its paper name."""
    if name == "cspn_style":
        from .cspn_style.implementation import build

        return build(channels, propagation_steps=propagation_steps)
    if name == "dwconv":
        from .dwconv.implementation import build

        return build(channels, inner_multiplier=inner_multiplier or 1.5, bias=bias)
    if name == "mdta":
        from .mdta.implementation import build

        return build(channels, stage_index=stage_index)
    if name == "swin":
        from .swin.implementation import build

        return build(
            channels,
            stage_index=stage_index,
            block_index=block_index,
            inner_multiplier=inner_multiplier or 1.0,
        )
    if name == "ss2d":
        from .ss2d.implementation import build

        return build(channels, stage_index=stage_index, block_index=block_index)
    if name == "ours":
        from .ours.implementation import build

        return build(
            channels,
            stage_index=stage_index,
            block_index=block_index,
            propagation_steps=propagation_steps,
        )
    raise ValueError(f"cannot directly construct token mixer: {name!r}")


__all__ = ["build_token_mixer", "mixer_names"]
