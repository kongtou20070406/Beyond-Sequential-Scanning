"""Alternating W-MSA and SW-MSA baseline token mixer."""

from torch import nn


def build(
    channels: int,
    *,
    stage_index: int = 0,
    block_index: int = 0,
    inner_multiplier: float = 1.0,
) -> nn.Module:
    from clpo2027.mixers import ShiftedWindowAttention

    return ShiftedWindowAttention(
        channels,
        num_heads=1 << stage_index,
        window_size=7,
        shift_size=0 if block_index % 2 == 0 else 3,
        inner_multiplier=inner_multiplier,
    )


__all__ = ["build"]
