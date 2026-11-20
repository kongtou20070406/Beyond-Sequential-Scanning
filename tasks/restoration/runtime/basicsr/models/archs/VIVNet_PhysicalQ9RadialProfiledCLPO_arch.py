"""RadialProfiled CLPO with a conjugate-symmetric nine-real Q9 chart."""

from __future__ import annotations

import torch
from torch import nn

from .VIVNet_arch import AttBlock, VIVNet
from .VIVNet_ProfiledCLPO_arch import RadialStateDepthProfiledOperator


class PhysicalQ9RadialProfiledOperator(RadialStateDepthProfiledOperator):
    """Compile center-real plus four complex pairs, then use the original projection."""

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(channels, **kwargs)
        d = self.state_channels
        groups = self.selective_group_count
        self.coeff_out = nn.Conv2d(
            d, 9 * d, 1, groups=groups, bias=True
        )
        nn.init.normal_(self.coeff_out.weight, mean=0.0, std=1.0e-3)
        if self.coeff_out.bias is None:
            raise AssertionError("PhysicalQ9 coefficient head requires bias")
        with torch.no_grad():
            # Group-major ABI: [group, c, E.re/im, S.re/im, SE.re/im, SW.re/im, lane].
            bias = self.coeff_out.bias.view(
                groups, 9, self.selective_group_size
            )
            bias.zero_()
            bias[:, 0].fill_(1.0)
            bias[:, (1, 3, 5, 7)].fill_(0.5)

    def _lightweight_coefficients(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.coeff_out(hidden)
        b, _, h, w = raw.shape
        coordinates = raw.view(
            b,
            self.selective_group_count,
            9,
            self.selective_group_size,
            h,
            w,
        ).permute(0, 2, 1, 3, 4, 5).reshape(
            b, 9, self.state_channels, h, w
        )
        center = coordinates[:, 0]
        e_r, e_i = coordinates[:, 1], coordinates[:, 2]
        s_r, s_i = coordinates[:, 3], coordinates[:, 4]
        se_r, se_i = coordinates[:, 5], coordinates[:, 6]
        sw_r, sw_i = coordinates[:, 7], coordinates[:, 8]
        zero = torch.zeros_like(center)

        # N9 order: center, NW, N, NE, W, E, SW, S, SE.
        raw_real = torch.stack(
            (center, se_r, s_r, sw_r, e_r, e_r, sw_r, s_r, se_r), dim=1
        )
        raw_imag = torch.stack(
            (zero, -se_i, -s_i, -sw_i, -e_i, e_i, sw_i, s_i, se_i), dim=1
        )
        packed = torch.stack((raw_real, raw_imag), dim=1).reshape(
            b, 18 * self.state_channels, h, w
        )
        return self._diagonal_coefficients_from_raw(packed)


class VIVNetPhysicalQ9RadialProfiledCLPOMixer(nn.Module):
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
        self.operator = PhysicalQ9RadialProfiledOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            linear_tilt_init=linear_tilt_init,
            curved_tilt_init=curved_tilt_init,
            init_seed=20285000 + 101 * (channels // 2) + block_index,
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


class VIVNetPhysicalQ9RadialProfiledCLPO(VIVNet):
    """VIVNet-S with the matched RadialProfiled PhysicalQ9 transition."""

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
            raise ValueError("PhysicalQ9 RadialProfiled requires K >= 1")
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
            block.attn = VIVNetPhysicalQ9RadialProfiledCLPOMixer(
                block.attn.dim,
                K=self.K,
                rho_max=self.rho_max,
                group_size=group_size,
                linear_tilt_init=linear_tilt_init,
                curved_tilt_init=curved_tilt_init,
                block_index=block_index,
            )

    def clpos(self) -> tuple[PhysicalQ9RadialProfiledOperator, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, PhysicalQ9RadialProfiledOperator)
        )

    def set_rollout_k(self, K: int) -> None:
        if K < 1:
            raise ValueError("rollout K must be >= 1")
        self.K = int(K)
        for operator in self.clpos():
            operator.K = self.K
