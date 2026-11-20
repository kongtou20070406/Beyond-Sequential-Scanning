"""VIVNet-S with a lightweight Selective State-Depth CLPO mixer."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .VIVNet_arch import AttBlock, VIVNet
from clpo2027.model import ComplexStateOperator


class SelectiveStateDepthOperator(ComplexStateOperator):
    """Finite 2-D propagation with selective write, state, and read gates.

    The transition and all input-conditioned controls are compiled once.  At
    every state-depth step, the current complex state can either contract by
    group radius or gate the nine memory-transition taps direction by direction.
    """

    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int = 4,
        state_gate: str = "radial",
        gate_range: float = 0.25,
        selection_init: float = -4.0,
        init_seed: int = 0,
    ) -> None:
        if channels % 2:
            raise ValueError("Selective State-Depth CLPO requires even channels")
        state_channels = channels // 2
        if group_size < 1 or state_channels % group_size:
            raise ValueError("state channels must be divisible by group_size")
        if not 0.0 < gate_range < 1.0:
            raise ValueError("gate_range must lie in (0, 1)")
        if state_gate not in {"radial", "tap_discrepancy"}:
            raise ValueError("state_gate must be radial or tap_discrepancy")

        super().__init__(
            channels,
            "f9_propagated",
            K=K + 1,
            rho_max=rho_max,
            state_channels=state_channels,
            coeff_generator_depth=2,
            coefficient_compiler_mode="canonical_hidden",
        )
        # Public convention: K is the number of actual updates; the parent counts state slots.
        self.K = int(K)
        self.selective_group_size = int(group_size)
        self.selective_group_count = state_channels // group_size
        self.selective_gate_range = float(gate_range)
        self.state_gate = state_gate
        self.selective_backend = "eager"

        # One grouped pointwise field feeds full-width parallel 3x3/7x7
        # branches.  All coefficient and control heads still compile only once.
        groups = self.selective_group_count
        d = self.state_channels
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            self.coeff_in = nn.Conv2d(
                2 * d, d, 1, groups=groups, bias=False
            )
            self.coeff_dw = nn.Conv2d(
                d, d, 3, padding=1, groups=d, bias=True
            )
            self.coeff_dw2 = nn.Conv2d(
                d,
                d,
                7,
                padding=3,
                groups=d,
                bias=True,
            )
            self.coeff_gamma = nn.Parameter(torch.full((d,), 0.1))
            self.coeff_out = nn.Conv2d(
                d, 2 * self.tap_count * d, 1, groups=groups, bias=True
            )
            control_channels = (
                3 * d
                if state_gate == "radial"
                else 2 * d + self.tap_count * groups
            )
            self.selective_head = nn.Conv2d(
                d, control_channels, 1, groups=groups, bias=True
            )
        self._reset_lightweight_compiler(selection_init)
        # The nonlinear state selector changes the recurrence Jacobian, so the
        # historical linear finite-adjoint shortcut is intentionally bypassed.
        self.set_backward_mode("exact_unrolled")

    def _reset_lightweight_compiler(self, selection_init: float) -> None:
        if self.coeff_out.bias is None or self.selective_head.bias is None:
            raise AssertionError("selective compiler requires bias terms")
        nn.init.normal_(self.coeff_out.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.selective_head.weight)
        with torch.no_grad():
            # Group-major ABI: [group, real/imag, tap, state-within-group].
            bias = self.coeff_out.bias.view(
                self.selective_group_count,
                2,
                self.tap_count,
                self.selective_group_size,
            )
            bias.zero_()
            bias[:, 0, 0].fill_(1.0)
            bias[:, 0, 1:].fill_(0.5)

            if self.state_gate == "radial":
                # [write, read, state-selection-strength].
                gate_bias = self.selective_head.bias.view(
                    self.selective_group_count,
                    3,
                    self.selective_group_size,
                )
                gate_bias.zero_()
                gate_bias[:, 2].fill_(float(selection_init))
            else:
                # Per group: write/read state logits followed by nine positive
                # tap-decay strengths.  softplus(-4) starts close to no gating.
                gate_bias = self.selective_head.bias.view(
                    self.selective_group_count,
                    2 * self.selective_group_size + self.tap_count,
                )
                gate_bias.zero_()
                gate_bias[:, 2 * self.selective_group_size :].fill_(
                    float(selection_init)
                )

    @staticmethod
    def _interleave_complex(packed_f: torch.Tensor) -> torch.Tensor:
        fr, fi = packed_f.chunk(2, dim=1)
        return torch.stack((fr, fi), dim=2).flatten(1, 2)

    def _coefficient_hidden(
        self, x: torch.Tensor, packed_f: torch.Tensor
    ) -> torch.Tensor:
        del x
        source = self._interleave_complex(packed_f)
        z0 = self.coeff_in(source)
        primary = F.gelu(self.coeff_dw(z0))
        secondary = F.gelu(self.coeff_dw2(z0))
        return primary + self.coeff_gamma.view(1, -1, 1, 1) * secondary

    def _lightweight_coefficients(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.coeff_out(hidden)
        b, _, h, w = raw.shape
        raw = raw.view(
            b,
            self.selective_group_count,
            2,
            self.tap_count,
            self.selective_group_size,
            h,
            w,
        ).permute(0, 2, 3, 1, 4, 5, 6)
        return self._diagonal_coefficients_from_raw(
            raw.reshape(b, 2 * self.tap_count * self.state_channels, h, w)
        )

    def _selective_controls(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.selective_head(hidden)
        b, _, h, w = raw.shape
        if self.state_gate == "radial":
            raw = raw.view(
                b,
                self.selective_group_count,
                3,
                self.selective_group_size,
                h,
                w,
            ).permute(0, 2, 1, 3, 4, 5).reshape(
                b, 3 * self.state_channels, h, w
            )
            write_logit, read_logit, selection_logit = raw.chunk(3, dim=1)
            strength = F.softplus(selection_logit)
        else:
            raw = raw.view(
                b,
                self.selective_group_count,
                2 * self.selective_group_size + self.tap_count,
                h,
                w,
            )
            state_logits = raw[:, :, : 2 * self.selective_group_size]
            state_logits = state_logits.view(
                b,
                self.selective_group_count,
                2,
                self.selective_group_size,
                h,
                w,
            ).permute(0, 2, 1, 3, 4, 5).reshape(
                b, 2 * self.state_channels, h, w
            )
            write_logit, read_logit = state_logits.chunk(2, dim=1)
            tap_logits = raw[:, :, 2 * self.selective_group_size :]
            strength = F.softplus(tap_logits.permute(0, 2, 1, 3, 4))
        alpha = self.selective_gate_range
        write = 1.0 + alpha * torch.tanh(write_logit)
        read = 1.0 + alpha * torch.tanh(read_logit)
        return write, read, strength

    def _select_state(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
        strength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = real.shape
        shape = (
            b,
            self.selective_group_count,
            self.selective_group_size,
            h,
            w,
        )
        real_g = real.reshape(shape)
        imag_g = imag.reshape(shape)
        strength_g = strength.reshape(shape).mean(dim=2, keepdim=True)
        group_energy = (
            real_g.square() + imag_g.square()
        ).mean(dim=2, keepdim=True)
        scale = torch.rsqrt(1.0 + strength_g * group_energy)
        return (real_g * scale).reshape_as(real), (imag_g * scale).reshape_as(imag)

    def _state_depth_step(
        self,
        propagated_r: torch.Tensor,
        propagated_i: torch.Tensor,
        drive_r: torch.Tensor,
        drive_i: torch.Tensor,
        strength: torch.Tensor,
        ar: torch.Tensor,
        ai: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_gate == "tap_discrepancy":
            from clpo2027.selective_f9_backend import (
                compiled_tap_discrepancy_f9_step,
                tap_discrepancy_f9_step_graph,
            )

            step = (
                compiled_tap_discrepancy_f9_step
                if self.selective_backend == "compiled_selective_f9"
                else tap_discrepancy_f9_step_graph
            )
            return step(
                propagated_r,
                propagated_i,
                drive_r,
                drive_i,
                strength,
                ar,
                ai,
                self.selective_group_size,
            )
        if self.selective_backend == "compiled_selective_f9":
            from clpo2027.selective_f9_backend import compiled_selective_f9_step

            return compiled_selective_f9_step(
                propagated_r, propagated_i, drive_r, drive_i,
                strength, ar, ai, self.selective_group_size,
            )
        if self.selective_backend != "eager":
            raise RuntimeError(
                f"unknown selective backend {self.selective_backend!r}"
            )
        memory_r, memory_i = self._select_state(
            propagated_r, propagated_i, strength
        )
        return self.apply_transition(
            drive_r + memory_r, drive_i + memory_i, ar, ai
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed_f, fr, fi, attention = self._drive_with_attention(x)
        hidden = self._coefficient_hidden(x, packed_f)
        ar, ai = self._lightweight_coefficients(hidden)
        write, read, strength = self._selective_controls(hidden)
        drive_r, drive_i = write * fr, write * fi
        if self.state_gate == "tap_discrepancy":
            drive_r, drive_i = self.apply_transition(
                drive_r, drive_i, ar, ai
            )

        propagated_r = torch.zeros_like(fr)
        propagated_i = torch.zeros_like(fi)
        for _ in range(self.K):
            step = lambda pr, pi: self._state_depth_step(
                pr, pi, drive_r, drive_i, strength, ar, ai
            )
            propagated_r, propagated_i = (
                checkpoint(
                    step,
                    propagated_r,
                    propagated_i,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if torch.is_grad_enabled()
                else step(propagated_r, propagated_i)
            )
        value = torch.cat(
            (read * propagated_r, read * propagated_i), dim=1
        )
        return self._interface_readout(value, attention)


class VIVNetSelectiveCLPOMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int,
        state_gate: str,
        block_index: int,
    ) -> None:
        super().__init__()
        self.operator = SelectiveStateDepthOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            state_gate=state_gate,
            init_seed=20278000 + 101 * (channels // 2) + block_index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = (
            "compiled_selective_f9"
            if x.is_cuda and torch.is_grad_enabled()
            else "eager"
        )
        self.operator.selective_backend = backend
        if self.operator.transition_backend != "eager":
            self.operator.set_transition_backend("eager")
        return self.operator(x)


class VIVNetSelectiveStateDepthCLPO(VIVNet):
    """VIVNet-S carrier with Selective State-Depth CLPO in every mixer slot."""

    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 32,
        num_blocks: list[int] = [2, 3, 4],
        num_refinement_blocks: int = 4,
        ffn_expansion_factor: float = 3,
        bias: bool = False,
        K: int = 5,
        rho_max: float = 0.95,
        group_size: int = 4,
        state_gate: str = "radial",
    ) -> None:
        if K < 1:
            raise ValueError("Selective State-Depth CLPO requires K >= 1")
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
        )
        self.K = int(K)
        self.rho_max = float(rho_max)
        self.group_size = int(group_size)
        self.state_gate = state_gate

        blocks = [module for module in self.modules() if isinstance(module, AttBlock)]
        expected = 2 * sum(num_blocks) + num_refinement_blocks
        if len(blocks) != expected:
            raise AssertionError(f"unexpected VIVNet AttBlock count: {len(blocks)}")
        for block_index, block in enumerate(blocks):
            block.attn = VIVNetSelectiveCLPOMixer(
                block.attn.dim,
                K=self.K,
                rho_max=self.rho_max,
                group_size=self.group_size,
                state_gate=self.state_gate,
                block_index=block_index,
            )

    def clpos(self) -> tuple[SelectiveStateDepthOperator, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, SelectiveStateDepthOperator)
        )

    def set_rollout_k(self, K: int) -> None:
        if K < 2:
            raise ValueError("rollout K must be >= 2")
        self.K = int(K)
        for operator in self.clpos():
            operator.K = self.K


def _self_test() -> None:
    torch.manual_seed(7)
    operator = SelectiveStateDepthOperator(
        8, K=3, rho_max=0.95, group_size=2, init_seed=11
    )
    nn.init.normal_(operator.Cout.weight, mean=0.0, std=1.0e-2)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    y = operator(x)
    y.square().mean().backward()
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise AssertionError("selective operator forward failed")
    gradient = operator.selective_head.weight.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise AssertionError("selective controls have no finite gradient")
    print(
        "SELF_TEST_OK",
        f"shape={tuple(y.shape)}",
        f"params={sum(p.numel() for p in operator.parameters())}",
        f"gate_grad={float(gradient.norm()):.6e}",
    )


if __name__ == "__main__":
    _self_test()
