"""VIVNet carrier with complete source-native MaIR or Spatial-Mamba blocks.

This is the portable form of the audited ``VIVNativeBlocks`` implementation
used for the two 55,800-update Appendix D endpoints. It replaces all 22 carrier
AttBlocks and leaves the carrier widths, stages, skip fusions, and image
residual unchanged.
"""

import torch
from torch import nn

from .VIVNet_arch import AttBlock, VIVNet
from clpo2027 import mair_vmm_adapter
from clpo2027.spatial_native_adapter import module as spatial_official


_MAIR = mair_vmm_adapter._MAIR
_IDS = {}


def _mair_ids(height, width, scan_len, device):
    key = (height, width, scan_len, str(device))
    if key not in _IDS:
        base = _MAIR.mair_ids_generate((1, 1, height, width), scan_len=scan_len)
        shifted = _MAIR.mair_shift_ids_generate(
            (1, 1, height, width), scan_len=scan_len, shift_len=scan_len // 2
        )
        _IDS[key] = tuple(value.to(device=device) for value in (*base, *shifted))
    return _IDS[key]


class _MaIRVSSBlock(nn.Module):
    """Official MaIRUNet VSSBlock using the bundled official MaIR operators."""

    def __init__(self, channels, d_state, shift_size):
        super().__init__()
        self.ln_1 = nn.LayerNorm(channels)
        self.self_attention = _MAIR.VMM(
            d_model=channels, d_state=d_state, expand=1.2, dropout=0.0
        )
        self.drop_path = nn.Identity()
        self.skip_scale = nn.Parameter(torch.ones(channels))
        self.mlp = _MAIR.Mlp(
            in_features=channels, hidden_features=int(channels * 2.0)
        )
        self.ln_2 = nn.LayerNorm(channels)
        self.skip_scale2 = nn.Parameter(torch.ones(channels))
        self.shift_size = shift_size

    def forward(self, tokens, ids, size):
        batch, _, channels = tokens.shape
        height, width = size
        value = tokens.view(batch, height, width, channels).contiguous()
        chosen_ids = ids[2:] if self.shift_size else ids[:2]
        mixed = self.self_attention(self.ln_1(value), chosen_ids)
        value = value * self.skip_scale + self.drop_path(mixed)
        value = value * self.skip_scale2 + self.mlp(self.ln_2(value))
        return value.view(batch, -1, channels).contiguous()


class MaIRNativeBlock(nn.Module):
    def __init__(self, channels, *, d_state, block_index):
        super().__init__()
        self.scan_len = 8
        self.block = _MaIRVSSBlock(
            channels,
            d_state=d_state,
            shift_size=0 if block_index % 2 == 0 else self.scan_len // 2,
        )

    def forward(self, value):
        batch, channels, height, width = value.shape
        tokens = value.permute(0, 2, 3, 1).reshape(
            batch, height * width, channels
        ).contiguous()
        tokens = self.block(
            tokens, _mair_ids(height, width, self.scan_len, value.device), (height, width)
        )
        return tokens.reshape(batch, height, width, channels).permute(
            0, 3, 1, 2
        ).contiguous()


class SpatialNativeBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = spatial_official.SpatialMambaBlock(
            hidden_dim=channels,
            d_state=1,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            drop_path=0.0,
        )
        self.block.apply(lambda item: spatial_official.SpatialMamba._init_weights(None, item))

    def forward(self, value):
        value = self.block(value.permute(0, 2, 3, 1).contiguous())
        return value.permute(0, 3, 1, 2).contiguous()


class VIVNativeBlocks(VIVNet):
    """Fixed VIVNet carrier with complete source-native processing blocks."""

    def __init__(self, method):
        super().__init__(
            inp_channels=3,
            out_channels=3,
            dim=32,
            num_blocks=[2, 3, 4],
            num_refinement_blocks=4,
            ffn_expansion_factor=3,
            bias=False,
        )
        if method not in {"mair", "spatial"}:
            raise ValueError(method)
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
        for stage in stages:
            for block_index, block in enumerate(stage):
                if not isinstance(block, AttBlock):
                    raise TypeError("VIVNet carrier block changed before native replacement")
                channels = block.attn.dim
                if method == "mair":
                    stage[block_index] = MaIRNativeBlock(
                        channels,
                        d_state=4 * (channels // 32),
                        block_index=block_index,
                    )
                else:
                    stage[block_index] = SpatialNativeBlock(channels)
                replaced += 1
        if replaced != 22:
            raise ValueError(f"Expected 22 carrier blocks, found {replaced}")


def clear_spatial_eval_cache(model):
    """Drop only derived inference kernels after a state load or training."""
    for child in model.modules():
        if isinstance(child, spatial_official.StateFusion) and hasattr(
            child, "_merge_weight"
        ):
            del child._merge_weight
