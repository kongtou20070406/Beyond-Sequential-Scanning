from __future__ import annotations

from pathlib import Path
import sys

from torch import nn


VMAMBA_COMMIT = "2ed52ead062a51a64521ed3871d52914bf532876"
_MODELS = Path(__file__).resolve().parents[1] / "third_party" / "VMamba" / "classification" / "models"


def build_official_ss2d_mixer(
    channels: int, *, stage_index: int, block_index: int
) -> nn.Module:
    if str(_MODELS) not in sys.path:
        sys.path.insert(0, str(_MODELS))
    from vmamba import SS2D

    mixer = SS2D(
        d_model=channels,
        d_state=16,
        ssm_ratio=0.75,
        dt_rank="auto",
        d_conv=3,
        forward_type="v03",
        channel_first=True,
    )
    setattr(mixer, "\x65qop_stage_index", stage_index)
    setattr(mixer, "\x65qop_block_index", block_index)
    return mixer
