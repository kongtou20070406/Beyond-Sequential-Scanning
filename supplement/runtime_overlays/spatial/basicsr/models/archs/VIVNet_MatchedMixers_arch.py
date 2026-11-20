"""VIVNet-S carrier and parameter-matched token-mixer baselines."""

from __future__ import annotations

import torch
from torch import nn

from .VIVNet_arch import AttBlock, VIVNet
from clpo2027.mixers import (
    CSPNOriginalStyleMixer,
    CSPNStyleMixer,
    ExpandedDWConv,
    MDTA,
    ShiftedWindowAttention,
)


MIXER_TYPES = {
    "dwconv", "mdta", "swin", "ss2d", "cspn_style",
    "cspn_original_style", "mair", "spatial_mamba",
}


def _make_mixer(
    mixer_type: str,
    channels: int,
    *,
    base_dim: int,
    stage_index: int,
    block_index: int,
    K: int,
) -> nn.Module:
    if mixer_type == "spatial_mamba":
        from clpo2027.spatial_mamba_adapter import build_spatial_mamba
        return build_spatial_mamba(channels)
    if mixer_type == "mair":
        from clpo2027.mair_vmm_adapter import build_official_mair_mixer
        return build_official_mair_mixer(
            channels, stage_index=stage_index, block_index=block_index
        )
    heads = channels // base_dim
    if mixer_type == "dwconv":
        return ExpandedDWConv(channels, inner_multiplier=3.0)
    if mixer_type == "mdta":
        return MDTA(channels, num_heads=heads, inner_multiplier=1.0)
    if mixer_type == "swin":
        return ShiftedWindowAttention(
            channels,
            num_heads=heads,
            window_size=7,
            shift_size=0 if block_index % 2 == 0 else 3,
            inner_multiplier=1.75,
        )
    if mixer_type == "cspn_style":
        return CSPNOriginalStyleMixer(channels, steps=K)
    if mixer_type == "cspn_original_style":
        return CSPNOriginalStyleMixer(channels, steps=K)
    from clpo2027.vmamba_ss2d_adapter import build_official_ss2d_mixer

    return build_official_ss2d_mixer(
        channels, stage_index=stage_index, block_index=block_index
    )


class VIVNetMatchedMixer(VIVNet):
    """Replace every VIVNet ``AttBlock.attn`` under one frozen carrier."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        num_blocks: list[int] = [2, 3, 4],
        num_refinement_blocks: int = 4,
        ffn_expansion_factor: float = 3,
        bias: bool = False,
        mixer_type: str = "dwconv",
        K: int = 5,
    ) -> None:
        if mixer_type not in MIXER_TYPES:
            raise ValueError(f"unknown mixer_type={mixer_type!r}")
        if K < 2:
            raise ValueError("K must be >= 2")
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
        )
        self.mixer_type = mixer_type
        self.K = int(K)
        stages = (
            self.encoder_level1,
            self.encoder_level2,
            self.encoder_level3,
            self.decoder_level3,
            self.decoder_level2,
            self.decoder_level1,
            self.refinement,
        )
        replaced = 0
        for stage_index, stage in enumerate(stages):
            for block_index, block in enumerate(stage):
                if not isinstance(block, AttBlock):
                    raise AssertionError("unexpected non-AttBlock in VIVNet stage")
                channels = block.attn.dim
                block.attn = _make_mixer(
                    mixer_type,
                    channels,
                    base_dim=dim,
                    stage_index=stage_index,
                    block_index=block_index,
                    K=self.K,
                )
                replaced += 1
        expected = 2 * sum(num_blocks) + num_refinement_blocks
        if replaced != expected:
            raise AssertionError(f"replaced {replaced} mixers, expected {expected}")


def _self_test() -> None:
    x = torch.randn(1, 3, 28, 28)
    for mixer_type in ("dwconv", "mdta", "swin", "cspn_style"):
        model = VIVNetMatchedMixer(
            dim=8,
            num_blocks=[1, 1, 1],
            num_refinement_blocks=1,
            mixer_type=mixer_type,
            K=3,
        )
        y = model(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        print(mixer_type, tuple(y.shape), sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    _self_test()
