"""Controlled mixer implementations and shared-initialization contracts."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


VALID_MIXER_TYPES = frozenset(
    {
        "clpo_canonical",
        "mdta_matched",
        "dwconv_expanded",
        "wmsa",
        "wmsa_shifted",
        "ss2d",
        "cspn_style_matched",
        "cspn_original_style_matched",
    }
)
_MIXER_KEY_MARKERS = (
    ".clpo.",
    ".mdta.",
    ".dwconv.",
    ".wmsa.",
    ".ss2d.",
    ".cspn.",
)


class CSPNStyleMixer(nn.Module):
    """Matched-slot homogeneous CSPN-style propagation.

    One signed 3x3 affinity field is compiled from the initial real state and
    reused for ``steps`` homogeneous updates.  The center coefficient is the
    CSPN residual mass, while the eight off-center coefficients have unit
    absolute mass per state channel and spatial position.
    """

    _OFFSETS = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )

    def __init__(
        self,
        channels: int,
        *,
        steps: int = 11,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if channels <= 0 or steps < 1 or epsilon <= 0:
            raise ValueError("channels/steps/epsilon must be positive")
        self.channels = int(channels)
        self.steps = int(steps)
        self.epsilon = float(epsilon)
        self.state_in = nn.Conv2d(channels, channels, 1, bias=True)
        self.affinity_head = nn.Conv2d(
            channels,
            8 * channels,
            3,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.state_out = nn.Conv2d(channels, channels, 1, bias=True)

    def _compile_affinity(self, state: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = state.shape
        off_center = self.affinity_head(state).reshape(
            batch, channels, 8, height, width
        )
        off_center = off_center / (
            off_center.abs().sum(dim=2, keepdim=True) + self.epsilon
        )
        center = 1.0 - off_center.sum(dim=2)
        return center, off_center

    def _propagate(
        self,
        state: Tensor,
        center: Tensor,
        off_center: Tensor,
    ) -> Tensor:
        height, width = state.shape[-2:]
        padded = F.pad(state, (1, 1, 1, 1))
        result = center * state
        for index, (dy, dx) in enumerate(self._OFFSETS):
            neighbor = padded[
                :, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width
            ]
            result = result + off_center[:, :, index] * neighbor
        return result

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "CSPNStyleMixer expects BCHW input with its registered width"
            )
        state = self.state_in(x)
        center, off_center = self._compile_affinity(state)
        for _ in range(self.steps):
            state = self._propagate(state, center, off_center)
        return self.state_out(state)


class CSPNOriginalStyleMixer(CSPNStyleMixer):
    """CSPN-style anchored refinement with one shared real affinity field."""

    def _propagate_anchored(
        self,
        state: Tensor,
        initial: Tensor,
        center: Tensor,
        off_center: Tensor,
    ) -> Tensor:
        height, width = state.shape[-2:]
        padded = F.pad(state, (1, 1, 1, 1))
        result = center * initial
        for index, (dy, dx) in enumerate(self._OFFSETS):
            neighbor = padded[
                :, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width
            ]
            result = result + off_center[:, :, index] * neighbor
        return result

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "CSPNOriginalStyleMixer expects BCHW input with its registered width"
            )
        initial = self.state_in(x)
        center, off_center = self._compile_affinity(initial)
        if initial.is_cuda and os.name != "nt":
            from .cspn_anchor_cuda import propagate

            state = propagate(
                initial.float(), center.float(), off_center.float(), self.steps
            ).to(initial.dtype)
        else:
            state = initial
            for _ in range(self.steps):
                state = self._propagate_anchored(
                    state, initial, center, off_center
                )
        return self.state_out(state)


class MDTA(nn.Module):
    """Restormer-style multi-Dconv head transposed attention.

    This is the mixer branch only. The surrounding normalization, residual,
    GDFN, carrier stages, and task head are owned by the common restoration
    block so the controlled comparison changes one complete mixer branch.
    """

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        inner_multiplier: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_heads <= 0:
            raise ValueError("channels and num_heads must be positive")
        if not 1.0 <= inner_multiplier <= 4.0:
            raise ValueError("inner_multiplier must be finite and in [1,4]")
        inner_channels = (
            round(channels * inner_multiplier / num_heads) * num_heads
        )
        if channels % num_heads or inner_channels % num_heads:
            raise ValueError(
                "input and internal channels must be divisible by num_heads"
            )
        self.channels = int(channels)
        self.inner_channels = int(inner_channels)
        self.num_heads = int(num_heads)
        self.inner_multiplier = float(inner_multiplier)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(
            channels, 3 * self.inner_channels, 1, bias=bias
        )
        self.qkv_dwconv = nn.Conv2d(
            3 * self.inner_channels,
            3 * self.inner_channels,
            3,
            stride=1,
            padding=1,
            groups=3 * self.inner_channels,
            bias=bias,
        )
        self.project_out = nn.Conv2d(
            self.inner_channels, channels, 1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "MDTA expects BCHW input with its registered channel width"
            )
        batch, channels, height, width = x.shape
        head_channels = self.inner_channels // self.num_heads
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        def split_heads(value: Tensor) -> Tensor:
            return value.reshape(
                batch,
                self.num_heads,
                head_channels,
                height * width,
            )

        q = F.normalize(split_heads(q), dim=-1)
        k = F.normalize(split_heads(k), dim=-1)
        v = split_heads(v)
        attention = torch.matmul(q, k.transpose(-2, -1))
        attention = (attention * self.temperature).softmax(dim=-1)
        mixed = torch.matmul(attention, v).reshape(
            batch, self.inner_channels, height, width
        )
        return self.project_out(mixed)


class ExpandedDWConv(nn.Module):
    """ConvNeXt-style 7x7 depthwise mixer with an internal channel MLP.

    The common carrier owns pre-normalization, the block residual, and GDFN.
    This branch therefore keeps only the spatial 7x7 depthwise operation and
    its ConvNeXt-style pointwise expansion/projection.  A 3x expansion places
    the complete C32/no-H8 carrier in the preregistered parameter band.
    """

    def __init__(
        self,
        channels: int,
        *,
        inner_multiplier: float = 3.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if not 1.0 <= inner_multiplier <= 8.0:
            raise ValueError("inner_multiplier must be finite and in [1,8]")
        inner_channels = round(channels * inner_multiplier)
        self.channels = int(channels)
        self.inner_channels = int(inner_channels)
        self.inner_multiplier = float(inner_multiplier)
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=bias,
        )
        self.expand = nn.Conv2d(
            channels, inner_channels, kernel_size=1, bias=bias
        )
        self.project = nn.Conv2d(
            inner_channels, channels, kernel_size=1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "ExpandedDWConv expects BCHW input with its registered width"
            )
        return self.project(F.gelu(self.expand(self.dwconv(x))))


def _window_partition(value: Tensor, window_size: int) -> Tensor:
    batch, height, width, channels = value.shape
    return (
        value.reshape(
            batch,
            height // window_size,
            window_size,
            width // window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window_size * window_size, channels)
    )


def _window_reverse(
    windows: Tensor,
    *,
    batch: int,
    height: int,
    width: int,
    window_size: int,
) -> Tensor:
    channels = windows.shape[-1]
    return (
        windows.reshape(
            batch,
            height // window_size,
            width // window_size,
            window_size,
            window_size,
            channels,
        )
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(batch, height, width, channels)
    )


class ShiftedWindowAttention(nn.Module):
    """Swin/SwinIR-style window attention mixer for BCHW feature maps."""

    def __init__(
        self,
        channels: int,
        *,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        inner_multiplier: float = 1.75,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0 or num_heads <= 0:
            raise ValueError("channels and num_heads must be positive")
        if window_size <= 1:
            raise ValueError("window_size must be greater than one")
        if shift_size not in {0, window_size // 2}:
            raise ValueError("shift_size must be zero or half the window")
        if not 1.0 <= inner_multiplier <= 4.0:
            raise ValueError("inner_multiplier must be finite and in [1,4]")
        inner_channels = (
            round(channels * inner_multiplier / num_heads) * num_heads
        )
        if inner_channels % num_heads:
            raise ValueError("internal channels must divide into attention heads")
        self.channels = int(channels)
        self.inner_channels = int(inner_channels)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.inner_multiplier = float(inner_multiplier)
        self.head_channels = self.inner_channels // self.num_heads
        self.scale = self.head_channels**-0.5
        self.qkv = nn.Linear(
            channels, 3 * self.inner_channels, bias=qkv_bias
        )
        self.project = nn.Linear(self.inner_channels, channels, bias=True)

        relative_span = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_span * relative_span, num_heads)
        )
        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= relative_span
        self.register_buffer(
            "relative_position_index",
            relative.sum(-1),
            persistent=True,
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def _shift_attention_mask(
        self,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if self.shift_size == 0:
            return None
        mask = torch.zeros((1, height, width, 1), device=device, dtype=dtype)
        slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        region = 0
        for height_slice in slices:
            for width_slice in slices:
                mask[:, height_slice, width_slice, :] = region
                region += 1
        windows = _window_partition(mask, self.window_size).squeeze(-1)
        differences = windows.unsqueeze(1) - windows.unsqueeze(2)
        return differences.masked_fill(differences != 0, -100.0).masked_fill(
            differences == 0, 0.0
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                "ShiftedWindowAttention expects BCHW input with its "
                "registered width"
            )
        batch, _channels, source_height, source_width = x.shape
        pad_height = (-source_height) % self.window_size
        pad_width = (-source_width) % self.window_size
        if pad_height or pad_width:
            x = F.pad(x, (0, pad_width, 0, pad_height))
        height, width = x.shape[-2:]
        value = x.permute(0, 2, 3, 1)
        if self.shift_size:
            value = torch.roll(
                value,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        windows = _window_partition(value, self.window_size)
        window_batch, tokens, _ = windows.shape
        qkv = (
            self.qkv(windows)
            .reshape(
                window_batch,
                tokens,
                3,
                self.num_heads,
                self.head_channels,
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value_heads = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(tokens, tokens, self.num_heads)
        attention = attention + relative_bias.permute(2, 0, 1).unsqueeze(0)
        mask = self._shift_attention_mask(
            height=height,
            width=width,
            device=x.device,
            dtype=attention.dtype,
        )
        if mask is not None:
            window_count = mask.shape[0]
            attention = attention.reshape(
                batch, window_count, self.num_heads, tokens, tokens
            )
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.reshape(
                window_batch, self.num_heads, tokens, tokens
            )
        attention = attention.softmax(dim=-1)
        mixed = (
            (attention @ value_heads)
            .transpose(1, 2)
            .reshape(window_batch, tokens, self.inner_channels)
        )
        mixed = self.project(mixed)
        output = _window_reverse(
            mixed,
            batch=batch,
            height=height,
            width=width,
            window_size=self.window_size,
        )
        if self.shift_size:
            output = torch.roll(
                output,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        output = output.permute(0, 3, 1, 2).contiguous()
        return output[:, :, :source_height, :source_width]


def is_mixer_state_key(key: str) -> bool:
    """Return whether a state key belongs to the replaced mixer branch."""

    return any(marker in key for marker in _MIXER_KEY_MARKERS)


def shared_outer_state_contract(
    source: nn.Module,
    target: nn.Module,
    *,
    copy: bool,
) -> dict[str, Any]:
    """Validate, optionally copy, and directly compare common carrier state."""

    source_state: Mapping[str, Tensor] = source.state_dict()
    target_state: Mapping[str, Tensor] = target.state_dict()
    target_outer_keys = tuple(
        key for key in target_state if not is_mixer_state_key(key)
    )
    missing = tuple(key for key in target_outer_keys if key not in source_state)
    shape_mismatches = tuple(
        key
        for key in target_outer_keys
        if key in source_state
        and tuple(source_state[key].shape) != tuple(target_state[key].shape)
    )
    if missing or shape_mismatches:
        raise RuntimeError(
            "matched carrier state is incompatible: "
            f"missing={missing}, shape_mismatches={shape_mismatches}"
        )

    with torch.no_grad():
        if copy:
            for key in target_outer_keys:
                target_state[key].copy_(source_state[key])
        unequal = tuple(
            key
            for key in target_outer_keys
            if not torch.equal(source_state[key], target_state[key])
        )
    if unequal:
        raise RuntimeError(
            f"matched carrier tensors are not bitwise equal: {unequal[:8]}"
        )

    source_mixer_keys = tuple(
        key for key in source_state if is_mixer_state_key(key)
    )
    target_mixer_keys = tuple(
        key for key in target_state if is_mixer_state_key(key)
    )
    return {
        "schema": "clpo2027.shared_outer_state_contract.v1",
        "copy_performed": bool(copy),
        "shared_key_count": len(target_outer_keys),
        "shared_numel": sum(target_state[key].numel() for key in target_outer_keys),
        "shared_keys": list(target_outer_keys),
        "shared_shapes": {
            key: list(target_state[key].shape) for key in target_outer_keys
        },
        "source_mixer_key_count": len(source_mixer_keys),
        "source_mixer_numel": sum(
            source_state[key].numel() for key in source_mixer_keys
        ),
        "target_mixer_key_count": len(target_mixer_keys),
        "target_mixer_numel": sum(
            target_state[key].numel() for key in target_mixer_keys
        ),
        "bitwise_equal": True,
    }
