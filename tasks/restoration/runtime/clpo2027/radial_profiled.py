"""Radial-profiled F9 mixer for the matched MetaFormer classifier."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .model import ComplexStateOperator, N9_OFFSETS

_UNFOLD_TO_N9 = (4, 0, 1, 2, 3, 5, 6, 7, 8)


def _radial_rollout_kernel(
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    ar: Tensor,
    ai: Tensor,
    weights: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """Fusable K-step radial recurrence used by the optional CUDA path."""
    batch, channels, height, width = drive_r.shape
    group_count = channels // group_size
    shape = (batch, group_count, group_size, height, width)
    propagated_r = torch.zeros_like(drive_r)
    propagated_i = torch.zeros_like(drive_i)
    profiled_r = torch.zeros_like(drive_r)
    profiled_i = torch.zeros_like(drive_i)
    for weight in weights.unbind():
        real_g = propagated_r.reshape(shape)
        imag_g = propagated_i.reshape(shape)
        strength_g = strength.reshape(shape).mean(dim=2, keepdim=True)
        energy = (real_g.square() + imag_g.square()).mean(dim=2, keepdim=True)
        scale = torch.rsqrt(1.0 + strength_g * energy)
        memory_r = (real_g * scale).reshape_as(drive_r)
        memory_i = (imag_g * scale).reshape_as(drive_i)
        zr = drive_r + memory_r
        zi = drive_i + memory_i
        padded_r = F.pad(zr, (1, 1, 1, 1), mode="constant", value=0.0)
        padded_i = F.pad(zi, (1, 1, 1, 1), mode="constant", value=0.0)
        # ``unfold`` keeps the 3x3 neighborhoods as a view; reshape them to
        # the row-major tap axis and reorder to the canonical N9 tap order.
        patches_r = padded_r.unfold(2, 3, 1).unfold(3, 3, 1)
        patches_i = padded_i.unfold(2, 3, 1).unfold(3, 3, 1)
        shifted_r = patches_r.permute(0, 4, 5, 1, 2, 3).reshape(
            zr.shape[0], 9, zr.shape[1], height, width
        )[:, _UNFOLD_TO_N9]
        shifted_i = patches_i.permute(0, 4, 5, 1, 2, 3).reshape(
            zr.shape[0], 9, zr.shape[1], height, width
        )[:, _UNFOLD_TO_N9]
        next_r = (ar * shifted_r - ai * shifted_i).sum(dim=1)
        next_i = (ar * shifted_i + ai * shifted_r).sum(dim=1)
        profiled_r = profiled_r + weight * (next_r - propagated_r)
        profiled_i = profiled_i + weight * (next_i - propagated_i)
        propagated_r, propagated_i = next_r, next_i
    return profiled_r, profiled_i


class RadialProfiledOperator(ComplexStateOperator):
    """Input-conditioned F9 propagation with radial state forgetting.

    The operator keeps the common complex F9 interface, but gates the state
    radius at every propagation step and applies a learned monotone profile to
    the realized depth increments.  The nonlinear recurrence intentionally
    uses the exact-unrolled/eager transition path.
    """

    def __init__(
        self,
        channels: int,
        *,
        K: int = 5,
        rho_max: float = 0.95,
        group_size: int = 4,
        linear_tilt_init: float = 0.20,
        curved_tilt_init: float = 0.01,
        init_seed: int = 0,
        transition_backend: str = "eager",
    ) -> None:
        if channels % 2:
            raise ValueError("RadialProfiled requires an even channel count")
        state_channels = channels // 2
        if group_size < 1 or state_channels % group_size:
            raise ValueError("state channels must be divisible by group_size")
        if K < 1:
            raise ValueError("RadialProfiled requires K >= 1")
        if linear_tilt_init < 0.0 or curved_tilt_init < 0.0:
            raise ValueError("profile tilt initializers must be nonnegative")
        if transition_backend not in {"eager", "blackwell_radial_k6", "blackwell_radial"}:
            raise ValueError(f"unsupported radial transition backend={transition_backend!r}")
        if transition_backend == "blackwell_radial_k6" and K != 5:
            raise ValueError("blackwell_radial_k6 requires five propagation steps")
        if transition_backend == "blackwell_radial" and K > 16:
            raise ValueError("blackwell_radial supports at most 16 transitions")

        super().__init__(
            channels,
            "f9_propagated",
            K=K + 1,
            rho_max=rho_max,
            state_channels=state_channels,
            coeff_generator_depth=2,
            coefficient_compiler_mode="canonical_hidden",
        )
        # K denotes actual synchronous updates and does not include the initial state h_0.
        self.K = int(K)
        self.radial_group_size = int(group_size)
        self.radial_group_count = state_channels // group_size
        self.radial_gate_range = 0.25
        self.radial_transition_backend = transition_backend

        groups = self.radial_group_count
        d = state_channels
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.coeff_in = nn.Conv2d(2 * d, d, 1, groups=groups, bias=False)
            self.coeff_dw = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=True)
            self.coeff_dw2 = nn.Conv2d(d, d, 7, padding=3, groups=d, bias=True)
            self.coeff_gamma = nn.Parameter(torch.full((d,), 0.1))
            self.coeff_out = nn.Conv2d(
                d, 2 * self.tap_count * d, 1, groups=groups, bias=True
            )
            self.selective_head = nn.Conv2d(
                d, 3 * d, 1, groups=groups, bias=True
            )
        self._reset_lightweight_compiler(selection_init=-4.0)
        self.linear_tilt_raw = nn.Parameter(torch.tensor(math.sqrt(linear_tilt_init)))
        self.curved_tilt_raw = nn.Parameter(torch.tensor(math.sqrt(curved_tilt_init)))
        # Runtime-only memory/speed switch.  The exact recurrence is unchanged;
        # disabling checkpointing lets a sufficiently large micro-batch avoid
        # recomputation on GPUs with enough memory.
        self.use_checkpoint = True
        self._compiled_rollout = None
        self.set_backward_mode("exact_unrolled")

    def _reset_lightweight_compiler(self, selection_init: float) -> None:
        nn.init.normal_(self.coeff_out.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.selective_head.weight)
        with torch.no_grad():
            bias = self.coeff_out.bias.view(
                self.radial_group_count,
                2,
                self.tap_count,
                self.radial_group_size,
            )
            bias.zero_()
            bias[:, 0, 0].fill_(1.0)
            bias[:, 0, 1:].fill_(0.5)
            gate_bias = self.selective_head.bias.view(
                self.radial_group_count, 3, self.radial_group_size
            )
            gate_bias.zero_()
            gate_bias[:, 2].fill_(float(selection_init))

    @staticmethod
    def _interleave_complex(packed_f: Tensor) -> Tensor:
        fr, fi = packed_f.chunk(2, dim=1)
        return torch.stack((fr, fi), dim=2).flatten(1, 2)

    def _coefficient_hidden(
        self, x_or_packed_f: Tensor, packed_f: Tensor | None = None
    ) -> Tensor:
        # Keep the parent helper signature usable for inherited diagnostics.
        source = self._interleave_complex(
            x_or_packed_f if packed_f is None else packed_f
        )
        z0 = self.coeff_in(source)
        primary = F.gelu(self.coeff_dw(z0))
        secondary = F.gelu(self.coeff_dw2(z0))
        # Keep the compiler on the active AMP path.  A FP32 parameter multiplied
        # by BF16 activations otherwise promotes the entire hidden field back to
        # FP32, including both large coefficient heads and their backward graph.
        gamma = self.coeff_gamma.to(dtype=secondary.dtype).view(1, -1, 1, 1)
        return primary + gamma * secondary

    def _lightweight_coefficients(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.coeff_out(hidden)
        batch, _, height, width = raw.shape
        raw = raw.view(
            batch,
            self.radial_group_count,
            2,
            self.tap_count,
            self.radial_group_size,
            height,
            width,
        ).permute(0, 2, 3, 1, 4, 5, 6)
        return self._diagonal_coefficients_from_raw(
            raw.reshape(batch, 2 * self.tap_count * self.state_channels, height, width)
        )

    def _radial_controls(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        raw = self.selective_head(hidden)
        batch, _, height, width = raw.shape
        raw = raw.view(
            batch,
            self.radial_group_count,
            3,
            self.radial_group_size,
            height,
            width,
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            batch, 3 * self.state_channels, height, width
        )
        write_logit, read_logit, selection_logit = raw.chunk(3, dim=1)
        strength = F.softplus(selection_logit)
        alpha = self.radial_gate_range
        write = 1.0 + alpha * torch.tanh(write_logit)
        read = 1.0 + alpha * torch.tanh(read_logit)
        return write, read, strength

    def _select_state(
        self, real: Tensor, imag: Tensor, strength: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, _, height, width = real.shape
        shape = (
            batch,
            self.radial_group_count,
            self.radial_group_size,
            height,
            width,
        )
        real_g = real.reshape(shape)
        imag_g = imag.reshape(shape)
        strength_g = strength.reshape(shape).mean(dim=2, keepdim=True)
        group_energy = (real_g.square() + imag_g.square()).mean(
            dim=2, keepdim=True
        )
        scale = torch.rsqrt(1.0 + strength_g * group_energy)
        return (real_g * scale).reshape_as(real), (imag_g * scale).reshape_as(imag)

    def _state_depth_step(
        self,
        propagated_r: Tensor,
        propagated_i: Tensor,
        drive_r: Tensor,
        drive_i: Tensor,
        strength: Tensor,
        ar: Tensor,
        ai: Tensor,
    ) -> tuple[Tensor, Tensor]:
        memory_r, memory_i = self._select_state(
            propagated_r, propagated_i, strength
        )
        return self.apply_transition(drive_r + memory_r, drive_i + memory_i, ar, ai)

    def _apply_stencil(
        self, zr: Tensor, zi: Tensor, ar: Tensor, ai: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Apply all nine taps as one reduction instead of nine serial updates."""
        height, width = zr.shape[-2:]
        padded_r = F.pad(zr, (1, 1, 1, 1), mode="constant", value=0.0)
        padded_i = F.pad(zi, (1, 1, 1, 1), mode="constant", value=0.0)
        shifted_r = torch.stack(
            [
                padded_r[:, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
                for dy, dx in self.offsets
            ],
            dim=1,
        )
        shifted_i = torch.stack(
            [
                padded_i[:, :, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
                for dy, dx in self.offsets
            ],
            dim=1,
        )
        return (
            (ar * shifted_r - ai * shifted_i).sum(dim=1),
            (ar * shifted_i + ai * shifted_r).sum(dim=1),
        )

    def depth_weights(self, reference: Tensor) -> Tensor:
        steps = self.K
        position = torch.linspace(
            0.0, 1.0, steps, device=reference.device, dtype=reference.dtype
        )
        linear = self.linear_tilt_raw.to(dtype=reference.dtype).square()
        curved = self.curved_tilt_raw.to(dtype=reference.dtype).square()
        logits = linear * position + curved * position.square()
        return torch.softmax(logits, dim=0) * steps

    def _profiled_rollout(
        self,
        drive_r: Tensor,
        drive_i: Tensor,
        strength: Tensor,
        ar: Tensor,
        ai: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run the exact K-step recurrence inside one checkpoint region.

        Keeping the whole fixed-length rollout in one region removes eleven
        per-step checkpoint dispatches while preserving the same explicit
        recurrence and exact autograd derivatives.
        """
        propagated_r = torch.zeros_like(drive_r)
        propagated_i = torch.zeros_like(drive_i)
        profiled_r = torch.zeros_like(drive_r)
        profiled_i = torch.zeros_like(drive_i)
        for weight in weights.unbind():
            next_r, next_i = self._state_depth_step(
                propagated_r,
                propagated_i,
                drive_r,
                drive_i,
                strength,
                ar,
                ai,
            )
            profiled_r = profiled_r + weight * (next_r - propagated_r)
            profiled_i = profiled_i + weight * (next_i - propagated_i)
            propagated_r, propagated_i = next_r, next_i
        return profiled_r, profiled_i

    def forward(self, x: Tensor) -> Tensor:
        packed_f, fr, fi, attention = self._drive_with_attention(x)
        hidden = self._coefficient_hidden(packed_f)
        if self.radial_transition_backend in {"blackwell_radial_k6", "blackwell_radial"} and hidden.is_cuda:
            from .radial_profiled_cuda import radial_coefficients_cuda

            ar, ai = radial_coefficients_cuda(
                hidden,
                self.coeff_out.weight,
                self.coeff_out.bias,
                self.rho_max,
                self.radial_group_size,
            )
        else:
            ar, ai = self._lightweight_coefficients(hidden)
        write, read, strength = self._radial_controls(hidden)
        drive_r, drive_i = write * fr, write * fi

        weights = self.depth_weights(fr)
        if self.radial_transition_backend in {"blackwell_radial_k6", "blackwell_radial"} and drive_r.is_cuda:
            from .radial_profiled_cuda import radial_profiled_cuda

            profiled_r, profiled_i = radial_profiled_cuda(
                drive_r,
                drive_i,
                strength,
                ar,
                ai,
                weights,
                self.radial_group_size,
            )
            value = torch.cat((read * profiled_r, read * profiled_i), dim=1)
            return self._interface_readout(value, attention)

        rollout = self._compiled_rollout or self._profiled_rollout
        if torch.is_grad_enabled() and self.use_checkpoint:
            profiled_r, profiled_i = checkpoint(
                rollout,
                drive_r,
                drive_i,
                strength,
                ar,
                ai,
                weights,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            if self._compiled_rollout is None:
                profiled_r, profiled_i = self._profiled_rollout(
                    drive_r, drive_i, strength, ar, ai, weights
                )
            else:
                profiled_r, profiled_i = self._compiled_rollout(
                    drive_r,
                    drive_i,
                    strength,
                    ar,
                    ai,
                    weights,
                )

        value = torch.cat((read * profiled_r, read * profiled_i), dim=1)
        return self._interface_readout(value, attention)


class RadialProfiledMixer(nn.Module):
    """BCHW token-mixer wrapper used by the MetaFormer carrier."""

    def __init__(
        self,
        channels: int,
        *,
        K: int = 5,
        rho_max: float = 0.95,
        group_size: int = 4,
        linear_tilt_init: float = 0.20,
        curved_tilt_init: float = 0.01,
        init_seed: int = 0,
        transition_backend: str = "eager",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.operator = RadialProfiledOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            init_seed=init_seed,
            transition_backend=transition_backend,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError("RadialProfiledMixer expects BCHW input")
        return self.operator(x)


__all__ = ["RadialProfiledMixer", "RadialProfiledOperator"]
