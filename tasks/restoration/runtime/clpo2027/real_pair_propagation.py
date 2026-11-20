"""Apply state propagation directly to real channel pairs."""
from __future__ import annotations

from functools import lru_cache
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


OFFSETS = ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1),
           (0, 1), (1, -1), (1, 0), (1, 1))


def _state_step(state, drive, kernel, strength, group_size):
    batch, channels, height, width = state.shape
    groups = channels // (2 * group_size)
    grouped = state.reshape(batch, groups, group_size, 2, height, width)
    energy = grouped.square().sum(3).mean(2, keepdim=True).unsqueeze(3)
    strength = strength.reshape(batch, groups, group_size, height, width)
    strength = strength.mean(2, keepdim=True).unsqueeze(3)
    retained = (grouped * torch.rsqrt(1 + strength * energy)).reshape_as(state)
    return apply_pair_kernel(drive + retained, kernel)


@lru_cache(maxsize=1)
def _compiled_state_step():
    return torch.compile(_state_step, dynamic=False, fullgraph=True)


def apply_pair_kernel(value: Tensor, kernel: Tensor) -> Tensor:
    """Apply nine two-channel transforms. Store only their two free values.

    value: [batch, channels, height, width]. Adjacent channels form a pair.
    kernel: [batch, 9, pairs, 2, height, width]. Each pair stores a and b.
    The transform is [[a, -b], [b, a]].
    """
    batch, channels, height, width = value.shape
    pairs = value.reshape(batch, channels // 2, 2, height, width)
    padded = F.pad(pairs, (1, 1, 1, 1))
    result = torch.zeros_like(pairs)
    for tap, (dy, dx) in enumerate(OFFSETS):
        neighbor = padded[:, :, :, 1+dy:1+dy+height, 1+dx:1+dx+width]
        a, b = kernel[:, tap, :, 0], kernel[:, tap, :, 1]
        # These are views of the two coordinates used by each transform.
        first, second = neighbor[:, :, 0], neighbor[:, :, 1]
        update = torch.stack((a*first-b*second, b*first+a*second), dim=2)
        result = result + update
    return result.reshape_as(value)


class RealPairStatePropagation(nn.Module):
    """Generate real features in pair order and keep that order through L_f."""

    def __init__(self, channels: int, *, K: int = 5, rho_max: float = .95,
                 group_size: int = 4, linear_tilt_init: float = .20,
                 curved_tilt_init: float = .01) -> None:
        super().__init__()
        if group_size < 1 or K < 1 or not 0 < rho_max < 1:
            raise ValueError("invalid group size, update count, or gain limit")
        if channels <= 0 or channels % (2*group_size):
            raise ValueError("channels must be a positive multiple of 2*group_size")
        if min(linear_tilt_init, curved_tilt_init) < 0:
            raise ValueError("depth weights require nonnegative initial slopes")
        self.channels = channels
        self.state_channels = channels // 2
        self.K = K
        self.rho_max = rho_max
        self.group_size = group_size
        self.group_count = self.state_channels // group_size
        self.gate_range = .25
        self.tap_count = 9
        self.selective_group_size = group_size
        self.selective_group_count = self.group_count
        self.selective_gate_range = self.gate_range
        self.state_gate = "radial"
        self.use_checkpoint = True
        self._loaded_legacy_layout = False
        d = self.state_channels
        self.coeff_gamma = nn.Parameter(torch.full((d,), .1))
        self.Bin = nn.Conv2d(channels, channels, 1, bias=False)
        self.coeff_in = nn.Conv2d(channels, d, 1, groups=self.group_count, bias=False)
        self.coeff_dw = nn.Conv2d(d, d, 3, padding=1, groups=d)
        self.coeff_dw2 = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.coeff_out = nn.Conv2d(d, 18*d, 1, groups=self.group_count)
        self.Cout = nn.Conv2d(channels, channels, 1, bias=False)
        self.selective_head = nn.Conv2d(d, 3*d, 1, groups=self.group_count)
        self.linear_tilt_raw = nn.Parameter(torch.tensor(math.sqrt(linear_tilt_init)))
        self.curved_tilt_raw = nn.Parameter(torch.tensor(math.sqrt(curved_tilt_init)))
        # Old files have no layout key. Convert their projection axes on load.
        self.register_buffer("pair_layout_version", torch.tensor(1, dtype=torch.int64))
        nn.init.normal_(self.coeff_out.weight, std=1e-3)
        nn.init.zeros_(self.Cout.weight)
        nn.init.zeros_(self.selective_head.weight)
        with torch.no_grad():
            bias = self.coeff_out.bias.view(self.group_count, 2, 9, group_size)
            bias.zero_()
            bias[:, 0, 0].fill_(1.)
            bias[:, 0, 1:].fill_(.5)
            control = self.selective_head.bias.view(self.group_count, 3, group_size)
            control.zero_()
            control[:, 2].fill_(-4.)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + "pair_layout_version")
        if version is None:
            keys = (prefix + "Bin.weight", prefix + "Cout.weight")
            if not all(key in state_dict for key in keys):
                raise RuntimeError("Old weights must contain both projection weights.")
            pair = torch.arange(self.state_channels, device=state_dict[keys[0]].device)
            order = torch.stack((pair, pair + self.state_channels), dim=1).flatten()
            state_dict[keys[0]] = state_dict[keys[0]].index_select(0, order)
            state_dict[keys[1]] = state_dict[keys[1]].index_select(1, order.to(state_dict[keys[1]].device))
            state_dict[prefix + "pair_layout_version"] = self.pair_layout_version.detach().clone()
            self._loaded_legacy_layout = True
        elif version.numel() != 1 or version.item() != 1:
            raise RuntimeError("Unknown state channel layout.")
        else:
            self._loaded_legacy_layout = False
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def coefficient_hidden(self, feature: Tensor) -> Tensor:
        hidden0 = self.coeff_in(feature)
        hidden = F.gelu(self.coeff_dw(hidden0))
        secondary = F.gelu(self.coeff_dw2(hidden0))
        return hidden + self.coeff_gamma.view(1, -1, 1, 1) * secondary

    def input_fields(self, feature: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Generate the shared kernel and controls from the feature directly."""
        hidden = self.coefficient_hidden(feature)
        batch, _, height, width = hidden.shape
        raw = self.coeff_out(hidden).reshape(
            batch, self.group_count, 2, 9, self.group_size, height, width)
        raw = raw.permute(0, 3, 1, 4, 2, 5, 6).reshape(
            batch, 9, self.state_channels, 2, height, width)
        mass = raw.square().sum(3).sqrt().sum(1, keepdim=True).unsqueeze(3)
        kernel = raw * (self.rho_max / (1+mass))
        control = self.selective_head(hidden).reshape(
            batch, self.group_count, 3, self.group_size, height, width)
        control = control.permute(0, 2, 1, 3, 4, 5).reshape(
            batch, 3, self.state_channels, height, width)
        write = 1 + self.gate_range * control[:, 0].tanh()
        read = 1 + self.gate_range * control[:, 1].tanh()
        strength = F.softplus(control[:, 2])
        return kernel, write, read, strength

    def scale_pairs(self, value: Tensor, scale: Tensor) -> Tensor:
        batch, channels, height, width = value.shape
        pairs = value.reshape(batch, channels//2, 2, height, width)
        return (pairs * scale.unsqueeze(2)).reshape_as(value)

    def retain_state(self, state: Tensor, strength: Tensor) -> Tensor:
        batch, _, height, width = state.shape
        grouped = state.reshape(batch, self.group_count, self.group_size, 2, height, width)
        energy = grouped.square().sum(3).mean(2, keepdim=True).unsqueeze(3)
        strength = strength.reshape(batch, self.group_count, self.group_size, height, width)
        strength = strength.mean(2, keepdim=True).unsqueeze(3)
        return (grouped * torch.rsqrt(1+strength*energy)).reshape_as(state)

    def depth_weights(self, reference: Tensor) -> Tensor:
        position = torch.linspace(0, 1, self.K, device=reference.device, dtype=reference.dtype)
        linear = self.linear_tilt_raw.to(reference.dtype).square()
        curved = self.curved_tilt_raw.to(reference.dtype).square()
        return (linear*position + curved*position.square()).softmax(0)*self.K

    def forward(self, x: Tensor) -> Tensor:
        feature = self.Bin(x)
        kernel, write, read, strength = self.input_fields(feature)
        drive = self.scale_pairs(feature, write)
        state = torch.zeros_like(feature)
        readout = torch.zeros_like(feature)
        step = _compiled_state_step() if x.is_cuda and torch.is_grad_enabled() else _state_step
        for weight in self.depth_weights(feature):
            if self.use_checkpoint and torch.is_grad_enabled():
                next_state = checkpoint(step, state, drive, kernel, strength, self.group_size,
                                        use_reentrant=False, preserve_rng_state=False)
            else:
                next_state = step(state, drive, kernel, strength, self.group_size)
            readout = readout + weight*(next_state-state)
            state = next_state
        return self.Cout(self.scale_pairs(readout, read))

    def convert_loaded_optimizer_state(self, optimizers):
        """Apply the same axis changes to a resumed old optimizer state."""
        if not self._loaded_legacy_layout:
            return
        pair = torch.arange(self.state_channels)
        order = torch.stack((pair, pair + self.state_channels), dim=1).flatten()
        for optimizer in optimizers:
            for parameter, axis in ((self.Bin.weight, 0), (self.Cout.weight, 1)):
                for name, value in optimizer.state.get(parameter, {}).items():
                    if isinstance(value, Tensor) and value.shape == parameter.shape:
                        optimizer.state[parameter][name] = value.index_select(axis, order.to(value.device))
        self._loaded_legacy_layout = False

    @classmethod
    def from_legacy(cls, operator):
        """Keep the seeded initial model and move only its channel order."""
        with torch.random.fork_rng(devices=[]):
            result = cls(operator.channels, K=operator.K, rho_max=operator.rho_max,
                         group_size=operator.selective_group_size)
        result.load_state_dict(operator.state_dict(), strict=True)
        result._loaded_legacy_layout = False
        return result
