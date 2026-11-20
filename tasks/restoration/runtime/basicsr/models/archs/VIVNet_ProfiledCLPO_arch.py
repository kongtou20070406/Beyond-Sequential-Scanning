"""VIVNet-S with a two-parameter monotone state-depth F9 profile."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .VIVNet_arch import AttBlock, VIVNet
from .VIVNet_SelectiveCLPO_arch import SelectiveStateDepthOperator
from .VIVNet_TiltedCLPO_arch import StateDepthTiltedOperator
from clpo2027.real_pair_propagation import RealPairStatePropagation
from clpo2027.selective_f9_backend import (
    compiled_group_channel_radial_f9_step,
    group_channel_radial_f9_step_graph,
)


class StateDepthProfiledOperator(StateDepthTiltedOperator):
    """Learn a monotone linear-plus-curved profile over existing F9 orders."""

    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int = 4,
        linear_tilt_init: float = 0.20,
        curved_tilt_init: float = 0.01,
        init_seed: int = 0,
    ) -> None:
        if linear_tilt_init < 0.0 or curved_tilt_init < 0.0:
            raise ValueError("state-depth tilt initializers must be nonnegative")
        super().__init__(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            depth_tilt_init=0.0,
            init_seed=init_seed,
        )
        self.K = int(K)
        self.depth_tilt_raw = None
        self.state_gate = "state_depth_profile"
        self.linear_tilt_raw = nn.Parameter(
            torch.tensor(math.sqrt(linear_tilt_init))
        )
        self.curved_tilt_raw = nn.Parameter(
            torch.tensor(math.sqrt(curved_tilt_init))
        )

    def depth_weights(self, reference: torch.Tensor) -> torch.Tensor:
        steps = self.K
        if steps < 1:
            raise ValueError("State-Depth Profiled F9 requires K >= 1")
        position = torch.linspace(
            0.0, 1.0, steps, device=reference.device, dtype=reference.dtype
        )
        linear = self.linear_tilt_raw.to(dtype=reference.dtype).square()
        curved = self.curved_tilt_raw.to(dtype=reference.dtype).square()
        logits = linear * position + curved * position.square()
        return torch.softmax(logits, dim=0) * steps


class RadialStateDepthProfiledOperator(SelectiveStateDepthOperator):
    """Combine radial state forgetting with a monotone late-depth readout."""

    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int = 4,
        linear_tilt_init: float = 0.20,
        curved_tilt_init: float = 0.01,
        init_seed: int = 0,
    ) -> None:
        if linear_tilt_init < 0.0 or curved_tilt_init < 0.0:
            raise ValueError("state-depth tilt initializers must be nonnegative")
        super().__init__(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            state_gate="radial",
            init_seed=init_seed,
        )
        # In the release API, K counts only actual synchronous updates.
        self.K = int(K)
        self.linear_tilt_raw = nn.Parameter(
            torch.tensor(math.sqrt(linear_tilt_init))
        )
        self.curved_tilt_raw = nn.Parameter(
            torch.tensor(math.sqrt(curved_tilt_init))
        )

    def depth_weights(self, reference: torch.Tensor) -> torch.Tensor:
        steps = self.K
        if steps < 1:
            raise ValueError("Radial Profiled F9 requires K >= 1")
        position = torch.linspace(
            0.0, 1.0, steps, device=reference.device, dtype=reference.dtype
        )
        linear = self.linear_tilt_raw.to(dtype=reference.dtype).square()
        curved = self.curved_tilt_raw.to(dtype=reference.dtype).square()
        logits = linear * position + curved * position.square()
        return torch.softmax(logits, dim=0) * steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed_f, fr, fi, attention = self._drive_with_attention(x)
        hidden = self._coefficient_hidden(x, packed_f)
        ar, ai = self._lightweight_coefficients(hidden)
        write, read, strength = self._selective_controls(hidden)
        drive_r, drive_i = write * fr, write * fi
        weights = self.depth_weights(fr)

        propagated_r = torch.zeros_like(fr)
        propagated_i = torch.zeros_like(fi)
        profiled_r = torch.zeros_like(fr)
        profiled_i = torch.zeros_like(fi)
        for weight in weights.unbind():
            step = lambda pr, pi: self._state_depth_step(
                pr, pi, drive_r, drive_i, strength, ar, ai
            )
            next_r, next_i = (
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
            profiled_r = profiled_r + weight * (next_r - propagated_r)
            profiled_i = profiled_i + weight * (next_i - propagated_i)
            propagated_r, propagated_i = next_r, next_i

        return self._interface_readout(
            torch.cat((read * profiled_r, read * profiled_i), dim=1), attention
        )


class CTMHistoryRadialStateDepthProfiledOperator(
    RadialStateDepthProfiledOperator
):
    """RadialProfiled with a CTM-style three-token state-depth trace.

    Two learned complex prehistory tokens seed each state channel.  The current
    input-conditioned drive fills the third slot; every propagation step then
    shifts the window and appends the realized complex state increment.  A
    block-shared MLP reads each channel's own trace and covariance and emits one
    complex correction coefficient.  The existing F9 transition/compiler is
    reused unchanged.
    """

    history_length = 3
    history_feature_count = 12
    history_hidden = 4

    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int = 4,
        linear_tilt_init: float = 0.20,
        curved_tilt_init: float = 0.01,
        init_seed: int = 0,
    ) -> None:
        super().__init__(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            init_seed=init_seed,
        )
        d = self.state_channels
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed + 37)
            bound = 1.0 / math.sqrt(d)
            self.start_trace = nn.Parameter(
                torch.empty(2, 2, d).uniform_(-bound, bound)
            )
            self.history_mlp = nn.Sequential(
                nn.Linear(self.history_feature_count, self.history_hidden),
                nn.SiLU(),
                nn.Linear(self.history_hidden, 2),
            )
            nn.init.xavier_uniform_(self.history_mlp[0].weight)
            nn.init.zeros_(self.history_mlp[0].bias)
            nn.init.normal_(self.history_mlp[2].weight, mean=0.0, std=2.0e-2)
            nn.init.zeros_(self.history_mlp[2].bias)

    @staticmethod
    def _shift_history(
        history: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        newest: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        return history[1], history[2], newest

    def history_parameter_count(self) -> int:
        return self.start_trace.numel() + sum(
            parameter.numel() for parameter in self.history_mlp.parameters()
        )

    def _history_features(
        self,
        history: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        write: torch.Tensor,
        read: torch.Tensor,
        strength: torch.Tensor,
    ) -> torch.Tensor:
        real = torch.stack([token[0] for token in history], dim=-1)
        imag = torch.stack([token[1] for token in history], dim=-1)
        rms = torch.sqrt((real.square() + imag.square()).mean(-1, keepdim=True) + 1.0e-6)
        real, imag = real / rms, imag / rms
        centered_r = real - real.mean(-1, keepdim=True)
        centered_i = imag - imag.mean(-1, keepdim=True)
        cov_rr = centered_r.square().mean(-1, keepdim=True)
        cov_ri = (centered_r * centered_i).mean(-1, keepdim=True)
        cov_ii = centered_i.square().mean(-1, keepdim=True)
        controls = torch.stack(
            (write - 1.0, read - 1.0, torch.log1p(strength)), dim=-1
        )
        return torch.cat((real, imag, cov_rr, cov_ri, cov_ii, controls), dim=-1)

    def _history_correction(
        self,
        history: tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
        ],
        write: torch.Tensor,
        read: torch.Tensor,
        strength: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficient = self.history_mlp(
            self._history_features(history, write, read, strength)
        )
        kr, ki = coefficient.unbind(-1)
        latest_r, latest_i = history[-1]
        return kr * latest_r - ki * latest_i, kr * latest_i + ki * latest_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed_f, fr, fi, attention = self._drive_with_attention(x)
        hidden = self._coefficient_hidden(x, packed_f)
        ar, ai = self._lightweight_coefficients(hidden)
        write, read, strength = self._selective_controls(hidden)
        drive_r, drive_i = write * fr, write * fi
        weights = self.depth_weights(fr)

        drive_magnitude = torch.sqrt(drive_r.square() + drive_i.square() + 1.0e-6)
        trace = self.start_trace.to(dtype=drive_r.dtype)
        seed_r = trace[:, 0].view(2, 1, self.state_channels, 1, 1) * drive_magnitude.unsqueeze(0)
        seed_i = trace[:, 1].view(2, 1, self.state_channels, 1, 1) * drive_magnitude.unsqueeze(0)
        history = (
            (seed_r[0], seed_i[0]),
            (seed_r[1], seed_i[1]),
            (drive_r, drive_i),
        )

        propagated_r = torch.zeros_like(fr)
        propagated_i = torch.zeros_like(fi)
        profiled_r = torch.zeros_like(fr)
        profiled_i = torch.zeros_like(fi)
        for weight in weights.unbind():
            correction_r, correction_i = self._history_correction(
                history, write, read, strength
            )
            corrected_drive_r = drive_r + correction_r
            corrected_drive_i = drive_i + correction_i

            def step(pr, pi, dr, di):
                return self._state_depth_step(
                    pr, pi, dr, di, strength, ar, ai
                )

            next_r, next_i = (
                checkpoint(
                    step,
                    propagated_r,
                    propagated_i,
                    corrected_drive_r,
                    corrected_drive_i,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
                if torch.is_grad_enabled()
                else step(
                    propagated_r,
                    propagated_i,
                    corrected_drive_r,
                    corrected_drive_i,
                )
            )
            delta = (next_r - propagated_r, next_i - propagated_i)
            profiled_r = profiled_r + weight * delta[0]
            profiled_i = profiled_i + weight * delta[1]
            propagated_r, propagated_i = next_r, next_i
            history = self._shift_history(history, delta)

        return self._interface_readout(
            torch.cat((read * profiled_r, read * profiled_i), dim=1), attention
        )


class GroupChannelRadialStateDepthProfiledOperator(
    RadialStateDepthProfiledOperator
):
    """Interpolate group and channel radial memory without changing F9."""

    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int = 4,
        linear_tilt_init: float = 0.20,
        curved_tilt_init: float = 0.01,
        channel_radial_eta_init: float = 0.0,
        init_seed: int = 0,
    ) -> None:
        if not 0.0 <= channel_radial_eta_init <= 1.0:
            raise ValueError("channel_radial_eta_init must lie in [0, 1]")
        super().__init__(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            init_seed=init_seed,
        )
        self.channel_radial_eta_raw = nn.Parameter(
            torch.tensor(float(channel_radial_eta_init))
        )

    def channel_radial_eta(self, reference: torch.Tensor) -> torch.Tensor:
        raw = self.channel_radial_eta_raw.to(dtype=reference.dtype)
        bounded = raw.clamp(0.0, 1.0)
        return raw + (bounded - raw).detach()

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
        eta = self.channel_radial_eta(propagated_r)
        if self.selective_backend == "compiled_selective_f9":
            return compiled_group_channel_radial_f9_step(
                propagated_r,
                propagated_i,
                drive_r,
                drive_i,
                strength,
                eta,
                ar,
                ai,
                self.selective_group_size,
            )
        if self.selective_backend != "eager":
            raise RuntimeError(
                f"unknown selective backend {self.selective_backend!r}"
            )
        return group_channel_radial_f9_step_graph(
            propagated_r,
            propagated_i,
            drive_r,
            drive_i,
            strength,
            eta,
            ar,
            ai,
            self.selective_group_size,
        )


class VIVNetRadialStateDepthProfiledCLPOMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int,
        linear_tilt_init: float,
        curved_tilt_init: float,
        block_index: int,
        real_pair_state: bool = True,
    ) -> None:
        super().__init__()
        self.operator = RadialStateDepthProfiledOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            init_seed=20282000 + 101 * (channels // 2) + block_index,
        )
        if real_pair_state:
            self.operator = RealPairStatePropagation.from_legacy(self.operator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from clpo2027.fp32_inference import use_fp32_inference, fp32_inference
        if type(self) is VIVNetRadialStateDepthProfiledCLPOMixer and use_fp32_inference(self.operator, x):
            return fp32_inference(self.operator, x)
        if isinstance(self.operator, RealPairStatePropagation):
            return self.operator(x)
        self.operator.selective_backend = (
            "compiled_selective_f9"
            if x.is_cuda and torch.is_grad_enabled()
            else "eager"
        )
        if self.operator.transition_backend != "eager":
            self.operator.set_transition_backend("eager")
        return self.operator(x)


class VIVNetCTMHistoryRadialStateDepthProfiledCLPOMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int,
        linear_tilt_init: float,
        curved_tilt_init: float,
        block_index: int,
    ) -> None:
        super().__init__()
        self.operator = CTMHistoryRadialStateDepthProfiledOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            init_seed=20284000 + 101 * (channels // 2) + block_index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.operator.selective_backend = (
            "compiled_selective_f9"
            if x.is_cuda and torch.is_grad_enabled()
            else "eager"
        )
        if self.operator.transition_backend != "eager":
            self.operator.set_transition_backend("eager")
        return self.operator(x)


class VIVNetGroupChannelRadialStateDepthProfiledCLPOMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int,
        linear_tilt_init: float,
        curved_tilt_init: float,
        channel_radial_eta_init: float,
        block_index: int,
    ) -> None:
        super().__init__()
        self.operator = GroupChannelRadialStateDepthProfiledOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            channel_radial_eta_init=channel_radial_eta_init,
            init_seed=20283000 + 101 * (channels // 2) + block_index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.operator.selective_backend = (
            "compiled_selective_f9"
            if x.is_cuda and torch.is_grad_enabled()
            else "eager"
        )
        if self.operator.transition_backend != "eager":
            self.operator.set_transition_backend("eager")
        return self.operator(x)


class VIVNetStateDepthProfiledCLPOMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int,
        linear_tilt_init: float,
        curved_tilt_init: float,
        block_index: int,
    ) -> None:
        super().__init__()
        self.operator = StateDepthProfiledOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            init_seed=20281000 + 101 * (channels // 2) + block_index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = "compiled_f9" if x.is_cuda and torch.is_grad_enabled() else "eager"
        if self.operator.transition_backend != backend:
            self.operator.set_transition_backend(backend)
        return self.operator(x)


class VIVNetStateDepthProfiledCLPO(VIVNet):
    """VIVNet-S with the general two-parameter State-Depth Tilted F9 arm."""

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
    ) -> None:
        if K < 1:
            raise ValueError("State-Depth Profiled F9 requires K >= 1")
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

        blocks = [module for module in self.modules() if isinstance(module, AttBlock)]
        expected = 2 * sum(num_blocks) + num_refinement_blocks
        if len(blocks) != expected:
            raise AssertionError(f"unexpected VIVNet AttBlock count: {len(blocks)}")
        for block_index, block in enumerate(blocks):
            block.attn = VIVNetStateDepthProfiledCLPOMixer(
                block.attn.dim,
                K=self.K,
                rho_max=self.rho_max,
                group_size=group_size,
                linear_tilt_init=linear_tilt_init,
                curved_tilt_init=curved_tilt_init,
                block_index=block_index,
            )

    def clpos(self) -> tuple[StateDepthProfiledOperator, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, StateDepthProfiledOperator)
        )

    def set_rollout_k(self, K: int) -> None:
        if K < 1:
            raise ValueError("rollout K must be >= 1")
        self.K = int(K)
        for operator in self.clpos():
            operator.K = self.K


class VIVNetRadialStateDepthProfiledCLPO(VIVNet):
    """VIVNet-S with radial forgetting and a two-parameter depth profile."""

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
        real_pair_state: bool = True,
    ) -> None:
        if K < 1:
            raise ValueError("Radial Profiled F9 requires K >= 1")
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

        blocks = [module for module in self.modules() if isinstance(module, AttBlock)]
        expected = 2 * sum(num_blocks) + num_refinement_blocks
        if len(blocks) != expected:
            raise AssertionError(f"unexpected VIVNet AttBlock count: {len(blocks)}")
        for block_index, block in enumerate(blocks):
            block.attn = VIVNetRadialStateDepthProfiledCLPOMixer(
                block.attn.dim,
                K=self.K,
                rho_max=self.rho_max,
                group_size=group_size,
                linear_tilt_init=linear_tilt_init,
                curved_tilt_init=curved_tilt_init,
                block_index=block_index,
                real_pair_state=real_pair_state,
            )

    def clpos(self) -> tuple[RadialStateDepthProfiledOperator | RealPairStatePropagation, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, (RadialStateDepthProfiledOperator, RealPairStatePropagation))
        )

    def set_rollout_k(self, K: int) -> None:
        if K < 1:
            raise ValueError("rollout K must be >= 1")
        self.K = int(K)
        for operator in self.clpos():
            operator.K = self.K


class VIVNetCTMHistoryRadialStateDepthProfiledCLPO(VIVNet):
    """RadialProfiled with learned prehistory and a sliding history MLP."""

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
    ) -> None:
        if K < 1:
            raise ValueError("CTM-History Radial Profiled F9 requires K >= 1")
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

        blocks = [module for module in self.modules() if isinstance(module, AttBlock)]
        expected = 2 * sum(num_blocks) + num_refinement_blocks
        if len(blocks) != expected:
            raise AssertionError(f"unexpected VIVNet AttBlock count: {len(blocks)}")
        for block_index, block in enumerate(blocks):
            block.attn = VIVNetCTMHistoryRadialStateDepthProfiledCLPOMixer(
                block.attn.dim,
                K=self.K,
                rho_max=self.rho_max,
                group_size=group_size,
                linear_tilt_init=linear_tilt_init,
                curved_tilt_init=curved_tilt_init,
                block_index=block_index,
            )

    def clpos(self) -> tuple[CTMHistoryRadialStateDepthProfiledOperator, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, CTMHistoryRadialStateDepthProfiledOperator)
        )

    def history_parameter_count(self) -> int:
        return sum(operator.history_parameter_count() for operator in self.clpos())

    def set_rollout_k(self, K: int) -> None:
        if K < 1:
            raise ValueError("rollout K must be >= 1")
        self.K = int(K)
        for operator in self.clpos():
            operator.K = self.K


class VIVNetGroupChannelRadialStateDepthProfiledCLPO(VIVNet):
    """RadialProfiled with group-to-channel selective memory interpolation."""

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
        channel_radial_eta_init: float = 0.0,
    ) -> None:
        if K < 1:
            raise ValueError("Group-Channel Radial Profiled F9 requires K >= 1")
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

        blocks = [module for module in self.modules() if isinstance(module, AttBlock)]
        expected = 2 * sum(num_blocks) + num_refinement_blocks
        if len(blocks) != expected:
            raise AssertionError(f"unexpected VIVNet AttBlock count: {len(blocks)}")
        for block_index, block in enumerate(blocks):
            block.attn = VIVNetGroupChannelRadialStateDepthProfiledCLPOMixer(
                block.attn.dim,
                K=self.K,
                rho_max=self.rho_max,
                group_size=group_size,
                linear_tilt_init=linear_tilt_init,
                curved_tilt_init=curved_tilt_init,
                channel_radial_eta_init=channel_radial_eta_init,
                block_index=block_index,
            )

    def clpos(self) -> tuple[GroupChannelRadialStateDepthProfiledOperator, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, GroupChannelRadialStateDepthProfiledOperator)
        )

    def set_rollout_k(self, K: int) -> None:
        if K < 1:
            raise ValueError("rollout K must be >= 1")
        self.K = int(K)
        for operator in self.clpos():
            operator.K = self.K


def _self_test() -> None:
    torch.manual_seed(7)
    operator = StateDepthProfiledOperator(
        8, K=4, rho_max=0.95, group_size=2, init_seed=11
    )
    reference = torch.zeros((), dtype=torch.float32)
    weights = operator.depth_weights(reference)
    if not torch.all(weights[1:] >= weights[:-1]):
        raise AssertionError("profiled depth weights must be nondecreasing")
    if not torch.allclose(weights.sum(), torch.tensor(3.0), atol=1.0e-6):
        raise AssertionError("profiled depth weights must preserve total mass")
    with torch.no_grad():
        saved_linear = operator.linear_tilt_raw.clone()
        saved_curved = operator.curved_tilt_raw.clone()
        operator.linear_tilt_raw.zero_()
        operator.curved_tilt_raw.zero_()
        uniform = operator.depth_weights(reference)
        operator.linear_tilt_raw.copy_(saved_linear)
        operator.curved_tilt_raw.copy_(saved_curved)
    if not torch.equal(uniform, torch.ones_like(uniform)):
        raise AssertionError("zero profile must recover the original finite F9 readout")

    nn.init.normal_(operator.Cout.weight, mean=0.0, std=1.0e-2)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    y = operator(x)
    y.square().mean().backward()
    gradients = (operator.linear_tilt_raw.grad, operator.curved_tilt_raw.grad)
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise AssertionError("profiled operator forward failed")
    if any(g is None or not torch.isfinite(g) for g in gradients):
        raise AssertionError("state-depth profile has no finite gradient")
    print(
        "PROFILED_SELF_TEST_OK",
        f"shape={tuple(y.shape)}",
        f"weights={[round(float(v), 6) for v in weights.detach()]}",
        f"weight_sum={float(weights.detach().sum()):.6f}",
        f"linear_grad={float(gradients[0]):.6e}",
        f"curved_grad={float(gradients[1]):.6e}",
    )

    radial_profiled = RadialStateDepthProfiledOperator(
        8, K=4, rho_max=0.95, group_size=2, init_seed=13
    )
    radial_weights = radial_profiled.depth_weights(reference)
    if not torch.all(radial_weights[1:] >= radial_weights[:-1]):
        raise AssertionError("radial-profiled weights must be nondecreasing")
    if not torch.allclose(radial_weights.sum(), torch.tensor(3.0), atol=1.0e-6):
        raise AssertionError("radial-profiled weights must preserve total mass")
    nn.init.normal_(radial_profiled.Cout.weight, mean=0.0, std=1.0e-2)
    radial_x = torch.randn(1, 8, 16, 16, requires_grad=True)
    radial_y = radial_profiled(radial_x)
    radial_y.square().mean().backward()
    radial_gradients = (
        radial_profiled.linear_tilt_raw.grad,
        radial_profiled.curved_tilt_raw.grad,
    )
    if radial_y.shape != radial_x.shape or not torch.isfinite(radial_y).all():
        raise AssertionError("radial-profiled operator forward failed")
    if any(g is None or not torch.isfinite(g) for g in radial_gradients):
        raise AssertionError("radial-profiled depth profile has no finite gradient")
    print(
        "RADIAL_PROFILED_SELF_TEST_OK",
        f"shape={tuple(radial_y.shape)}",
        f"weights={[round(float(v), 6) for v in radial_weights.detach()]}",
        f"weight_sum={float(radial_weights.detach().sum()):.6f}",
    )

    group_channel = GroupChannelRadialStateDepthProfiledOperator(
        8,
        K=4,
        rho_max=0.95,
        group_size=2,
        channel_radial_eta_init=0.0,
        init_seed=13,
    )
    group_channel.load_state_dict(radial_profiled.state_dict(), strict=False)
    nn.init.normal_(group_channel.Cout.weight, mean=0.0, std=1.0e-2)
    radial_profiled.Cout.weight.data.copy_(group_channel.Cout.weight.data)
    compare_x = torch.randn(1, 8, 12, 12)
    baseline_y = radial_profiled(compare_x)
    recovered_y = group_channel(compare_x)
    if not torch.equal(baseline_y, recovered_y):
        raise AssertionError("eta=0 must exactly recover RadialProfiled")
    group_channel.channel_radial_eta_raw.data.fill_(0.5)
    channel_x = compare_x.detach().clone().requires_grad_(True)
    channel_y = group_channel(channel_x)
    channel_y.square().mean().backward()
    eta_gradient = group_channel.channel_radial_eta_raw.grad
    if channel_y.shape != channel_x.shape or not torch.isfinite(channel_y).all():
        raise AssertionError("group-channel radial operator forward failed")
    if eta_gradient is None or not torch.isfinite(eta_gradient):
        raise AssertionError("group-channel radial eta has no finite gradient")
    print(
        "GROUP_CHANNEL_RADIAL_SELF_TEST_OK",
        f"eta0_max_abs={float((baseline_y - recovered_y).abs().max().detach()):.3e}",
        f"eta={float(group_channel.channel_radial_eta(compare_x).detach()):.3f}",
        f"eta_grad={float(eta_gradient):.6e}",
    )

    history_operator = CTMHistoryRadialStateDepthProfiledOperator(
        8, K=4, rho_max=0.95, group_size=2, init_seed=17
    )
    nn.init.normal_(history_operator.Cout.weight, mean=0.0, std=1.0e-2)
    history_x = torch.randn(1, 8, 12, 12, requires_grad=True)
    history_y = history_operator(history_x)
    history_y.square().mean().backward()
    expected_operator_history_params = 4 * history_operator.state_channels + 62
    if history_operator.history_parameter_count() != expected_operator_history_params:
        raise AssertionError("unexpected CTM-history operator parameter count")
    history_gradients = (
        history_operator.start_trace.grad,
        history_operator.history_mlp[0].weight.grad,
        history_operator.history_mlp[2].weight.grad,
    )
    if history_y.shape != history_x.shape or not torch.isfinite(history_y).all():
        raise AssertionError("CTM-history operator forward failed")
    if any(
        gradient is None
        or not torch.isfinite(gradient).all()
        or float(gradient.abs().sum()) == 0.0
        for gradient in history_gradients
    ):
        raise AssertionError("CTM-history parameters have no finite nonzero gradient")

    token = lambda value: (torch.tensor(float(value)), torch.tensor(0.0))
    trace = (token(-2), token(-1), token(0))
    trace = history_operator._shift_history(trace, token(1))
    trace = history_operator._shift_history(trace, token(2))
    if [int(pair[0]) for pair in trace] != [0, 1, 2]:
        raise AssertionError("learned prehistory was not flushed after two shifts")
    parameters_at_k4 = history_operator.history_parameter_count()
    history_operator.K = 6
    with torch.no_grad():
        history_y_k6 = history_operator(history_x.detach())
    if history_y_k6.shape != history_x.shape:
        raise AssertionError("CTM-history K compatibility failed")
    if history_operator.history_parameter_count() != parameters_at_k4:
        raise AssertionError("CTM-history parameters must not depend on K")

    baseline_model = VIVNetRadialStateDepthProfiledCLPO()
    history_model = VIVNetCTMHistoryRadialStateDepthProfiledCLPO()
    total_delta = sum(p.numel() for p in history_model.parameters()) - sum(
        p.numel() for p in baseline_model.parameters()
    )
    if history_model.history_parameter_count() != 4692 or total_delta != 4692:
        raise AssertionError(
            f"expected 4692 whole-network history parameters, got "
            f"history={history_model.history_parameter_count()} delta={total_delta}"
        )
    if len(history_model.clpos()) != 22:
        raise AssertionError("expected 22 CTM-history operators")
    print(
        "CTM_HISTORY_PROFILED_SELF_TEST_OK",
        f"shape={tuple(history_y.shape)}",
        f"operator_history_params={history_operator.history_parameter_count()}",
        f"network_history_params={history_model.history_parameter_count()}",
        f"network_param_delta={total_delta}",
        f"start_trace_grad={float(history_operator.start_trace.grad.abs().sum()):.6e}",
        "K=4,6",
    )


if __name__ == "__main__":
    _self_test()
