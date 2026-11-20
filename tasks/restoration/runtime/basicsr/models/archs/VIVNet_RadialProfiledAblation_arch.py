"""Controlled RadialProfiled CLPO ablations on the compiled F9 path."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .VIVNet_arch import AttBlock, VIVNet
from .VIVNet_ProfiledCLPO_arch import RadialStateDepthProfiledOperator
from clpo2027.selective_f9_backend import (
    compiled_diagonal_lag1_f9_step,
    compiled_identity_memory_f9_step,
    compiled_matched_feedforward_rollout,
    compiled_real_pair_radial_f9_step,
    compiled_state_dynamic_rollout,
    compiled_step_conditioned_rollout,
    diagonal_lag1_f9_step_graph,
    identity_memory_f9_step_graph,
    real_pair_radial_f9_step_graph,
)


ABLATIONS = {
    "full",
    "first_drive",
    "static_transition",
    "identity_memory",
    "uniform_readout",
    "identity_memory_uniform_readout",
    "step_conditioned_transition",
    "matched_feedforward",
    "real_f9",
    "no_projection",
    "compiler_3x3x2",
    "diagonal_lag1_transition",
    "state_dynamic_transition",
}


class RadialProfiledAblationOperator(RadialStateDepthProfiledOperator):
    """Change exactly one named field of the RadialProfiled A contract."""

    def __init__(self, channels: int, *, ablation: str, init_seed: int, **kwargs) -> None:
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown RadialProfiled ablation {ablation!r}")
        propagation_steps = int(kwargs.get("K"))
        if propagation_steps < 1:
            raise ValueError("K must be the positive number of propagation steps")
        super().__init__(channels, init_seed=init_seed, **kwargs)
        self.K = propagation_steps
        self.ablation = ablation

        if ablation in {"step_conditioned_transition", "matched_feedforward"}:
            self.step_embedding = nn.Parameter(
                torch.zeros(self.K, self.state_channels, 1, 1)
            )

        if ablation == "static_transition":
            # The existing coefficient bias is the spatially shared learned F9;
            # the unused dynamic coefficient weights are frozen, not executed.
            self.coeff_out.weight.requires_grad_(False)

        if ablation == "compiler_3x3x2":
            d = self.state_channels
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed + 1)
                self.coeff_dw2 = nn.Conv2d(
                    d, d, 3, padding=1, groups=d, bias=True
                )

        if ablation == "real_f9":
            # Six controls for 2D real states. Doubling groups keeps the head
            # weight count equal to A; one shared 3D bias restores exact count.
            d = self.state_channels
            groups = self.selective_group_count
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed + 2)
                self.selective_head = nn.Conv2d(
                    d, 6 * d, 1, groups=2 * groups, bias=False
                )
            self.real_control_bias = nn.Parameter(torch.zeros(1, 3 * d, 1, 1))
            with torch.no_grad():
                self.selective_head.weight.zero_()
                self.real_control_bias[:, 2 * d :].fill_(-4.0)

    def depth_weights(self, reference: torch.Tensor) -> torch.Tensor:
        position = torch.linspace(
            0.0, 1.0, self.K, device=reference.device, dtype=reference.dtype
        )
        linear = self.linear_tilt_raw.to(dtype=reference.dtype).square()
        curved = self.curved_tilt_raw.to(dtype=reference.dtype).square()
        logits = linear * position + curved * position.square()
        return torch.softmax(logits, dim=0) * self.K

    def _coefficient_hidden(self, x: torch.Tensor, packed_f: torch.Tensor) -> torch.Tensor:
        if self.ablation != "compiler_3x3x2":
            return super()._coefficient_hidden(x, packed_f)
        del x
        source = self._interleave_complex(packed_f)
        primary = F.gelu(self.coeff_dw(self.coeff_in(source)))
        secondary = F.gelu(self.coeff_dw2(primary))
        return primary + self.coeff_gamma.view(1, -1, 1, 1) * secondary

    def _group_major_raw(self, hidden: torch.Tensor) -> torch.Tensor:
        b, _, h, w = hidden.shape
        if self.ablation == "static_transition":
            raw = self.coeff_out.bias.view(
                self.selective_group_count,
                2,
                self.tap_count,
                self.selective_group_size,
            ).permute(1, 2, 0, 3).reshape(
                1, 2 * self.tap_count * self.state_channels, 1, 1
            )
            return raw.expand(b, -1, h, w)
        return self._group_major_from_native(self.coeff_out(hidden))

    def _group_major_from_native(self, raw: torch.Tensor) -> torch.Tensor:
        b, _, h, w = raw.shape
        return raw.view(
            b,
            self.selective_group_count,
            2,
            self.tap_count,
            self.selective_group_size,
            h,
            w,
        ).permute(0, 2, 3, 1, 4, 5, 6).reshape(
            b, 2 * self.tap_count * self.state_channels, h, w
        )

    def _coefficients_from_group_major_raw(
        self, raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = raw.shape
        if self.ablation == "no_projection":
            raw = raw.view(b, 2, self.tap_count, self.state_channels, h, w)
            return raw[:, 0], raw[:, 1]
        return self._diagonal_coefficients_from_raw(raw)

    def _ablation_coefficients(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._coefficients_from_group_major_raw(self._group_major_raw(hidden))

    def _step_raw_offsets(self, reference: torch.Tensor) -> torch.Tensor:
        steps = self.step_embedding.shape[0]
        groups = self.coeff_out.groups
        inputs_per_group = self.coeff_out.in_channels // groups
        outputs_per_group = self.coeff_out.out_channels // groups
        embedded = self.step_embedding.to(dtype=reference.dtype).reshape(
            steps, groups, inputs_per_group
        )
        weight = self.coeff_out.weight.to(dtype=reference.dtype).reshape(
            groups, outputs_per_group, inputs_per_group
        )
        native = torch.einsum("sgi,goi->sgo", embedded, weight).reshape(
            steps, self.coeff_out.out_channels, 1, 1
        )
        return self._group_major_from_native(native)

    def _real_coefficients(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._group_major_raw(hidden).view(
            hidden.shape[0], 2, self.tap_count, self.state_channels,
            hidden.shape[-2], hidden.shape[-1],
        )
        coeff_a, coeff_b = raw[:, 0], raw[:, 1]
        scale_a = self.rho_max / (1.0 + coeff_a.abs().sum(dim=1, keepdim=True))
        scale_b = self.rho_max / (1.0 + coeff_b.abs().sum(dim=1, keepdim=True))
        return coeff_a * scale_a, coeff_b * scale_b

    def _real_controls(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        raw = self.selective_head(hidden)
        bias = torch.cat((self.real_control_bias, self.real_control_bias), dim=1)
        first, second = (raw + bias).chunk(2, dim=1)
        controls = []
        for field in (first, second):
            write, read, strength = field.chunk(3, dim=1)
            alpha = self.selective_gate_range
            controls.extend((
                1.0 + alpha * torch.tanh(write),
                1.0 + alpha * torch.tanh(read),
                F.softplus(strength),
            ))
        return tuple(controls)

    def _step(
        self,
        state_r: torch.Tensor,
        state_i: torch.Tensor,
        drive_r: torch.Tensor,
        drive_i: torch.Tensor,
        strength: torch.Tensor,
        ar: torch.Tensor,
        ai: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ablation in {"identity_memory", "identity_memory_uniform_readout"}:
            fn = (
                compiled_identity_memory_f9_step
                if self.selective_backend == "compiled_selective_f9"
                else identity_memory_f9_step_graph
            )
            return fn(state_r, state_i, drive_r, drive_i, ar, ai)
        return self._state_depth_step(
            state_r, state_i, drive_r, drive_i, strength, ar, ai
        )

    def _step_coefficients(
        self, hidden: torch.Tensor, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ablation not in {"step_conditioned_transition", "matched_feedforward"}:
            return self._ablation_coefficients(hidden)
        raw = self._group_major_raw(hidden)
        return self._coefficients_from_group_major_raw(
            raw + self._step_raw_offsets(hidden)[index : index + 1]
        )

    def _forward_real(self, x: torch.Tensor) -> torch.Tensor:
        packed_f, drive_a, drive_b, attention = self._drive_with_attention(x)
        hidden = self._coefficient_hidden(x, packed_f)
        coeff_a, coeff_b = self._real_coefficients(hidden)
        write_a, read_a, strength_a, write_b, read_b, strength_b = self._real_controls(hidden)
        drive_a, drive_b = write_a * drive_a, write_b * drive_b
        weights = self.depth_weights(drive_a)
        state_a, state_b = torch.zeros_like(drive_a), torch.zeros_like(drive_b)
        profile_a, profile_b = torch.zeros_like(drive_a), torch.zeros_like(drive_b)
        for weight in weights.unbind():
            fn = (
                compiled_real_pair_radial_f9_step
                if self.selective_backend == "compiled_selective_f9"
                else real_pair_radial_f9_step_graph
            )
            step = lambda a, b: fn(
                a, b, drive_a, drive_b, strength_a, strength_b,
                coeff_a, coeff_b, self.selective_group_size,
            )
            next_a, next_b = (
                checkpoint(step, state_a, state_b, use_reentrant=False, preserve_rng_state=False)
                if torch.is_grad_enabled() else step(state_a, state_b)
            )
            profile_a = profile_a + weight * (next_a - state_a)
            profile_b = profile_b + weight * (next_b - state_b)
            state_a, state_b = next_a, next_b
        return self._interface_readout(
            torch.cat((read_a * profile_a, read_b * profile_b), dim=1), attention
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ablation == "full":
            return super().forward(x)
        if self.ablation == "real_f9":
            return self._forward_real(x)

        packed_f, fr, fi, attention = self._drive_with_attention(x)
        hidden = self._coefficient_hidden(x, packed_f)
        write, read, strength = self._selective_controls(hidden)
        drive_r, drive_i = write * fr, write * fi

        if self.ablation == "identity_memory_uniform_readout":
            ar, ai = self._step_coefficients(hidden, 0)
            fn = (
                compiled_identity_memory_f9_step
                if self.selective_backend == "compiled_selective_f9"
                else identity_memory_f9_step_graph
            )

            def rollout(dr, di, coeff_r, coeff_i):
                state_r, state_i = torch.zeros_like(dr), torch.zeros_like(di)
                for _ in range(self.K):
                    state_r, state_i = fn(
                        state_r, state_i, dr, di, coeff_r, coeff_i
                    )
                return state_r, state_i

            state_r, state_i = (
                checkpoint(
                    rollout,
                    drive_r,
                    drive_i,
                    ar,
                    ai,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if torch.is_grad_enabled()
                else rollout(drive_r, drive_i, ar, ai)
            )
            return self._interface_readout(
                torch.cat((read * state_r, read * state_i), dim=1), attention
            )

        weights = (
            torch.ones(self.K, device=fr.device, dtype=fr.dtype)
            if self.ablation in {"uniform_readout", "identity_memory_uniform_readout"}
            else self.depth_weights(fr)
        )

        if self.ablation == "diagonal_lag1_transition":
            ar, ai = self._ablation_coefficients(hidden)
            state_r, state_i = torch.zeros_like(fr), torch.zeros_like(fi)
            previous_r, previous_i = torch.zeros_like(fr), torch.zeros_like(fi)
            profile_r, profile_i = torch.zeros_like(fr), torch.zeros_like(fi)
            fn = (
                compiled_diagonal_lag1_f9_step
                if self.selective_backend == "compiled_selective_f9"
                else diagonal_lag1_f9_step_graph
            )
            for weight in weights.unbind():
                step = lambda r, i, hr, hi: fn(
                    r,
                    i,
                    hr,
                    hi,
                    drive_r,
                    drive_i,
                    strength,
                    ar,
                    ai,
                    self.selective_group_size,
                )
                next_r, next_i, source_r, source_i = (
                    checkpoint(
                        step,
                        state_r,
                        state_i,
                        previous_r,
                        previous_i,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if torch.is_grad_enabled()
                    else step(state_r, state_i, previous_r, previous_i)
                )
                profile_r = profile_r + weight * (next_r - state_r)
                profile_i = profile_i + weight * (next_i - state_i)
                state_r, state_i = next_r, next_i
                previous_r, previous_i = source_r, source_i
            return self._interface_readout(
                torch.cat((read * profile_r, read * profile_i), dim=1), attention
            )

        if self.ablation == "state_dynamic_transition":
            if self.selective_backend == "compiled_selective_f9":
                profile_r, profile_i = compiled_state_dynamic_rollout(
                    packed_f,
                    drive_r,
                    drive_i,
                    strength,
                    self.coeff_in.weight,
                    self.coeff_dw.weight,
                    self.coeff_dw.bias,
                    self.coeff_dw2.weight,
                    self.coeff_dw2.bias,
                    self.coeff_gamma,
                    self.coeff_out.weight,
                    self.coeff_out.bias,
                    weights,
                    self.rho_max,
                    self.selective_group_size,
                )
                return self._interface_readout(
                    torch.cat((read * profile_r, read * profile_i), dim=1),
                    attention,
                )
            state_r, state_i = torch.zeros_like(fr), torch.zeros_like(fi)
            profile_r, profile_i = torch.zeros_like(fr), torch.zeros_like(fi)
            for weight in weights.unbind():
                def step(r, i):
                    state_packed = torch.cat((r, i), dim=1)
                    step_hidden = self._coefficient_hidden(x, packed_f + state_packed)
                    ar, ai = self._ablation_coefficients(step_hidden)
                    return self._state_depth_step(
                        r, i, drive_r, drive_i, strength, ar, ai
                    )

                next_r, next_i = (
                    checkpoint(
                        step,
                        state_r,
                        state_i,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if torch.is_grad_enabled()
                    else step(state_r, state_i)
                )
                profile_r = profile_r + weight * (next_r - state_r)
                profile_i = profile_i + weight * (next_i - state_i)
                state_r, state_i = next_r, next_i
            return self._interface_readout(
                torch.cat((read * profile_r, read * profile_i), dim=1), attention
            )

        if self.ablation == "matched_feedforward":
            if self.selective_backend == "compiled_selective_f9":
                state_r, state_i = compiled_matched_feedforward_rollout(
                    drive_r,
                    drive_i,
                    hidden,
                    self.coeff_out.weight,
                    self.coeff_out.bias,
                    self.step_embedding,
                    self.rho_max,
                    self.selective_group_size,
                )
                return self._interface_readout(
                    torch.cat((read * state_r, read * state_i), dim=1), attention
                )
            fn = (
                compiled_identity_memory_f9_step
                if self.selective_backend == "compiled_selective_f9"
                else identity_memory_f9_step_graph
            )

            def feedforward_rollout(dr, di, coefficient_hidden):
                state_r, state_i = dr, di
                zero_r, zero_i = torch.zeros_like(dr), torch.zeros_like(di)
                raw_base = self._group_major_raw(coefficient_hidden)
                raw_offsets = self._step_raw_offsets(coefficient_hidden)
                for index in range(self.K):
                    ar, ai = self._coefficients_from_group_major_raw(
                        raw_base + raw_offsets[index : index + 1]
                    )
                    state_r, state_i = fn(
                        state_r, state_i, zero_r, zero_i, ar, ai
                    )
                return state_r, state_i

            state_r, state_i = feedforward_rollout(drive_r, drive_i, hidden)
            return self._interface_readout(
                torch.cat((read * state_r, read * state_i), dim=1), attention
            )

        if self.ablation == "step_conditioned_transition":
            if self.selective_backend == "compiled_selective_f9":
                profile_r, profile_i = compiled_step_conditioned_rollout(
                    drive_r,
                    drive_i,
                    strength,
                    hidden,
                    self.coeff_out.weight,
                    self.coeff_out.bias,
                    self.step_embedding,
                    weights,
                    self.rho_max,
                    self.selective_group_size,
                )
                return self._interface_readout(
                    torch.cat((read * profile_r, read * profile_i), dim=1), attention
                )

            def state_rollout(dr, di, state_strength, coefficient_hidden, depth_weights):
                state_r, state_i = torch.zeros_like(dr), torch.zeros_like(di)
                profile_r, profile_i = torch.zeros_like(dr), torch.zeros_like(di)
                raw_base = self._group_major_raw(coefficient_hidden)
                raw_offsets = self._step_raw_offsets(coefficient_hidden)
                for index, weight in enumerate(depth_weights.unbind()):
                    ar, ai = self._coefficients_from_group_major_raw(
                        raw_base + raw_offsets[index : index + 1]
                    )
                    next_r, next_i = self._state_depth_step(
                        state_r, state_i, dr, di, state_strength, ar, ai
                    )
                    profile_r = profile_r + weight * (next_r - state_r)
                    profile_i = profile_i + weight * (next_i - state_i)
                    state_r, state_i = next_r, next_i
                return profile_r, profile_i

            profile_r, profile_i = state_rollout(
                drive_r, drive_i, strength, hidden, weights
            )
            return self._interface_readout(
                torch.cat((read * profile_r, read * profile_i), dim=1), attention
            )

        shared_coefficients = self._ablation_coefficients(hidden)

        state_r, state_i = torch.zeros_like(fr), torch.zeros_like(fi)
        profile_r, profile_i = torch.zeros_like(fr), torch.zeros_like(fi)
        zero_r, zero_i = torch.zeros_like(drive_r), torch.zeros_like(drive_i)
        for index, weight in enumerate(weights.unbind()):
            ar, ai = shared_coefficients
            current_drive_r = drive_r if self.ablation != "first_drive" or index == 0 else zero_r
            current_drive_i = drive_i if self.ablation != "first_drive" or index == 0 else zero_i
            step = lambda r, i, dr=current_drive_r, di=current_drive_i, ar=ar, ai=ai: self._step(
                r, i, dr, di, strength, ar, ai
            )
            next_r, next_i = (
                checkpoint(step, state_r, state_i, use_reentrant=False, preserve_rng_state=False)
                if torch.is_grad_enabled() else step(state_r, state_i)
            )
            profile_r = profile_r + weight * (next_r - state_r)
            profile_i = profile_i + weight * (next_i - state_i)
            state_r, state_i = next_r, next_i
        return self._interface_readout(
            torch.cat((read * profile_r, read * profile_i), dim=1), attention
        )


class VIVNetRadialProfiledAblationMixer(nn.Module):
    def __init__(
        self, channels: int, *, block_index: int, activation_checkpoint: bool, **kwargs
    ) -> None:
        super().__init__()
        self.activation_checkpoint = activation_checkpoint
        self.operator = RadialProfiledAblationOperator(
            channels,
            init_seed=20282000 + 101 * (channels // 2) + block_index,
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.operator.selective_backend = (
            "compiled_selective_f9" if x.is_cuda and torch.is_grad_enabled() else "eager"
        )
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(
                self.operator,
                x,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return self.operator(x)


class VIVNetRadialProfiledAblation(VIVNet):
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
        linear_tilt_init: float = 0.20,
        curved_tilt_init: float = 0.01,
        ablation: str = "full",
        activation_checkpoint_blocks: int = 0,
    ) -> None:
        super().__init__(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
        )
        blocks = [module for module in self.modules() if isinstance(module, AttBlock)]
        high_resolution_index = 0
        for block_index, block in enumerate(blocks):
            is_high_resolution = block.attn.dim == dim
            block.attn = VIVNetRadialProfiledAblationMixer(
                block.attn.dim,
                K=K,
                rho_max=rho_max,
                group_size=group_size,
                linear_tilt_init=linear_tilt_init,
                curved_tilt_init=curved_tilt_init,
                ablation=ablation,
                block_index=block_index,
                activation_checkpoint=(
                    is_high_resolution
                    and high_resolution_index < activation_checkpoint_blocks
                ),
            )
            high_resolution_index += int(is_high_resolution)

    def clpos(self) -> tuple[RadialProfiledAblationOperator, ...]:
        return tuple(m for m in self.modules() if isinstance(m, RadialProfiledAblationOperator))
