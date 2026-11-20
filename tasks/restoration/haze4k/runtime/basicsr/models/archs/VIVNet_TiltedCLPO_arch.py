"""VIVNet-S with a state-depth tilted finite F9 readout."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .VIVNet_arch import AttBlock, VIVNet
from .VIVNet_SelectiveCLPO_arch import SelectiveStateDepthOperator
from clpo2027.model import (
    _finite_trajectory,
    _stencil_coefficient_gradients,
    _transition_adjoint,
)


def _depth_increments(
    packed_f: torch.Tensor,
    endpoint: torch.Tensor,
    trajectory: torch.Tensor,
) -> torch.Tensor:
    if trajectory.shape[0] == 1:
        return endpoint.unsqueeze(0)
    return torch.cat(
        (
            trajectory[1:] - trajectory[:-1],
            (endpoint - trajectory[-1] + packed_f).unsqueeze(0),
        ),
        dim=0,
    )


class _FiniteTiltedAdjointProxy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        weighted_value: torch.Tensor,
        packed_f: torch.Tensor,
        ar: torch.Tensor,
        ai: torch.Tensor,
        weights: torch.Tensor,
        endpoint: torch.Tensor,
        trajectory: torch.Tensor,
        operator: SelectiveStateDepthOperator,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            packed_f.detach(),
            ar.detach(),
            ai.detach(),
            weights.detach(),
            endpoint.detach(),
            trajectory.detach(),
        )
        ctx.operator = operator
        return weighted_value.clone()

    @staticmethod
    def backward(ctx: Any, q: torch.Tensor) -> tuple[Any, ...]:
        packed_f, ar, ai, weights, endpoint, trajectory = ctx.saved_tensors
        operator = ctx.operator
        with torch.no_grad():
            increments = _depth_increments(packed_f, endpoint, trajectory)
            grad_weights = (increments * q.unsqueeze(0)).flatten(1).sum(dim=1)

            state_coefficients = torch.empty_like(weights)
            state_coefficients[:-1] = weights[:-1] - weights[1:]
            state_coefficients[-1] = weights[-1]

            future = torch.zeros_like(q)
            grad_f = torch.zeros_like(q)
            aligned_adjoint = [None] * weights.numel()
            for index in range(weights.numel() - 1, -1, -1):
                propagated = (
                    torch.zeros_like(q)
                    if index == weights.numel() - 1
                    else _transition_adjoint(operator, future, ar, ai)
                )
                if index != weights.numel() - 1:
                    grad_f = grad_f + propagated
                future = state_coefficients[index] * q + propagated
                aligned_adjoint[index] = future
            grad_f = grad_f + _transition_adjoint(operator, future, ar, ai)
            grad_ar, grad_ai = _stencil_coefficient_gradients(
                operator,
                trajectory,
                torch.stack(aligned_adjoint, dim=0),
                ar,
                ai,
            )
        return None, grad_f, grad_ar, grad_ai, grad_weights, None, None, None


class StateDepthTiltedOperator(SelectiveStateDepthOperator):
    """Reuse one fixed F9 transition while reweighting its finite powers."""

    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int = 4,
        depth_tilt_init: float = 0.05,
        init_seed: int = 0,
    ) -> None:
        if depth_tilt_init < 0.0:
            raise ValueError("depth_tilt_init must be nonnegative")
        super().__init__(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            state_gate="radial",
            init_seed=init_seed,
        )
        # The tilted arm has no write/state/read gate.  It reuses only the
        # parallel 3x3/7x7 compiler and the unchanged contractive F9 map.
        self.selective_head = None
        self.state_gate = "depth_tilt"
        self.depth_tilt_raw = nn.Parameter(
            torch.tensor(math.sqrt(float(depth_tilt_init)))
        )
        self.set_backward_mode("finite_exact_adjoint")

    def depth_weights(self, reference: torch.Tensor) -> torch.Tensor:
        steps = self.K
        if steps < 1:
            raise ValueError("State-Depth Tilted F9 requires K >= 1")
        order = torch.arange(steps, device=reference.device, dtype=reference.dtype)
        tau = self.depth_tilt_raw.to(dtype=reference.dtype).square()
        return torch.softmax(tau * order, dim=0) * steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed_f, fr, fi, attention = self._drive_with_attention(x)
        hidden = self._coefficient_hidden(x, packed_f)
        ar, ai = self._lightweight_coefficients(hidden)
        weights = self.depth_weights(fr)
        endpoint, trajectory = _finite_trajectory(
            self,
            packed_f.detach(),
            ar.detach(),
            ai.detach(),
            self.K,
        )
        weighted_value = (
            _depth_increments(packed_f.detach(), endpoint, trajectory)
            * weights.detach().view(-1, 1, 1, 1, 1)
        ).sum(dim=0)
        value = _FiniteTiltedAdjointProxy.apply(
            weighted_value,
            packed_f,
            ar,
            ai,
            weights,
            endpoint,
            trajectory,
            self,
        )
        return self._interface_readout(value, attention)


class VIVNetStateDepthTiltedCLPOMixer(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        K: int,
        rho_max: float,
        group_size: int,
        depth_tilt_init: float,
        block_index: int,
    ) -> None:
        super().__init__()
        self.operator = StateDepthTiltedOperator(
            channels,
            K=K,
            rho_max=rho_max,
            group_size=group_size,
            depth_tilt_init=depth_tilt_init,
            init_seed=20279000 + 101 * (channels // 2) + block_index,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = "compiled_f9" if x.is_cuda and torch.is_grad_enabled() else "eager"
        if self.operator.transition_backend != backend:
            self.operator.set_transition_backend(backend)
        return self.operator(x)


class VIVNetStateDepthTiltedCLPO(VIVNet):
    """VIVNet-S with a fixed F9 and learnable state-depth tilt per mixer."""

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
        depth_tilt_init: float = 0.05,
    ) -> None:
        if K < 1:
            raise ValueError("State-Depth Tilted F9 requires K >= 1")
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
            block.attn = VIVNetStateDepthTiltedCLPOMixer(
                block.attn.dim,
                K=self.K,
                rho_max=self.rho_max,
                group_size=group_size,
                depth_tilt_init=depth_tilt_init,
                block_index=block_index,
            )

    def clpos(self) -> tuple[StateDepthTiltedOperator, ...]:
        return tuple(
            module
            for module in self.modules()
            if isinstance(module, StateDepthTiltedOperator)
        )

    def set_rollout_k(self, K: int) -> None:
        if K < 2:
            raise ValueError("rollout K must be >= 2")
        self.K = int(K)
        for operator in self.clpos():
            operator.K = self.K


def _self_test() -> None:
    torch.manual_seed(7)
    operator = StateDepthTiltedOperator(
        8,
        K=4,
        rho_max=0.95,
        group_size=2,
        depth_tilt_init=0.05,
        init_seed=11,
    )
    weights = operator.depth_weights(torch.zeros((), dtype=torch.float32))
    if not torch.all(weights[1:] >= weights[:-1]):
        raise AssertionError("depth weights must be nondecreasing")
    if not torch.allclose(weights.sum(), torch.tensor(3.0), atol=1.0e-6):
        raise AssertionError("depth weights must preserve total coefficient mass")
    with torch.no_grad():
        saved_tilt = operator.depth_tilt_raw.clone()
        operator.depth_tilt_raw.zero_()
        uniform = operator.depth_weights(torch.zeros((), dtype=torch.float32))
        operator.depth_tilt_raw.copy_(saved_tilt)
    if not torch.equal(uniform, torch.ones_like(uniform)):
        raise AssertionError("zero tilt must recover the original finite F9 readout")

    nn.init.normal_(operator.Cout.weight, mean=0.0, std=1.0e-2)
    x = torch.randn(1, 8, 16, 16, requires_grad=True)
    y = operator(x)
    y.square().mean().backward()
    gradient = operator.depth_tilt_raw.grad
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise AssertionError("tilted operator forward failed")
    if gradient is None or not torch.isfinite(gradient):
        raise AssertionError("depth tilt has no finite gradient")
    print(
        "TILTED_SELF_TEST_OK",
        f"shape={tuple(y.shape)}",
        f"weights={[round(float(v), 6) for v in weights.detach()]}",
        f"weight_sum={float(weights.detach().sum()):.6f}",
        f"tilt_grad={float(gradient):.6e}",
    )


if __name__ == "__main__":
    _self_test()
