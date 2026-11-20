"""Canonical complex state operator shared by the published carriers."""
from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F
from torch import Tensor, nn

N8_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)
N9_OFFSETS: tuple[tuple[int, int], ...] = ((0, 0),) + N8_OFFSETS
VALID_VARIANTS = frozenset({"x8_full", "x8_propagated", "f9_propagated"})


def _q9_spatial_operator_for_mode(mode: str) -> str:
    return "dct_hybrid" if mode == "q9_shared_head_dct_hybrid_operator" else "local_depthwise"


def _new_q9_generator(state_channels: int, coefficient_compiler_mode: str):
    from .coefficient_compilers import ComplexF9Q9Generator
    return ComplexF9Q9Generator(
        state_channels,
        widely_linear=True,
        spatial_operator=_q9_spatial_operator_for_mode(coefficient_compiler_mode),
        gain_chart="direct",
    )


class ComplexStateOperator(nn.Module):
    """Finite propagated-only complex F9 operator.

    The base K value counts state slots and therefore performs K-1 transitions.
    Radial/selective subclasses pass K+1 and expose K actual updates.
    """

    def __init__(
        self,
        channels: int,
        variant: str = "f9_propagated",
        *,
        K: int = 12,
        rho_max: float = 0.95,
        state_channels: int | None = None,
        exchange_rank: int = 0,
        exchange_eta: float = 0.5,
        exchange_seed: int = 0,
        coeff_generator_depth: int = 1,
        coefficient_compiler_mode: str = "canonical_hidden",
        coefficient_coordinate_mode: str = "cartesian",
        q9_generator: nn.Module | None = None,
        q9_head: nn.Module | None = None,
        q9_init_seed: int = 0,
        **_: Any,
    ) -> None:
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"unsupported variant={variant!r}")
        if K < 1:
            raise ValueError("K must be >= 1")
        if not 0.0 < rho_max < 1.0:
            raise ValueError("rho_max must lie strictly between zero and one")
        if type(exchange_rank) is not int or exchange_rank < 0:
            raise ValueError("exchange_rank must be a non-negative integer")
        if exchange_rank:
            raise ValueError("channel exchange is not part of the release operator")
        if coeff_generator_depth not in {1, 2}:
            raise ValueError("coeff_generator_depth must be 1 or 2")
        if coefficient_coordinate_mode != "cartesian":
            raise ValueError("the release operator uses Cartesian F9 coordinates")
        if coefficient_compiler_mode not in {
            "canonical_hidden",
            "q9_shared_head_dct_hybrid_operator",
        }:
            raise ValueError(f"unsupported release compiler mode={coefficient_compiler_mode!r}")

        self.channels = int(channels)
        self.variant = variant
        self.K = int(K)
        self.rho_max = float(rho_max)
        self.state_channels = (
            int(state_channels) if state_channels is not None else max(1, channels // 2)
        )
        if self.state_channels < 1:
            raise ValueError("state_channels must be positive")
        self.coeff_generator_depth = int(coeff_generator_depth)
        self.coefficient_compiler_mode = coefficient_compiler_mode
        self.coefficient_coordinate_mode = coefficient_coordinate_mode
        self.offsets = N9_OFFSETS if variant == "f9_propagated" else N8_OFFSETS
        self.tap_count = len(self.offsets)
        self.backward_mode = "exact_unrolled"
        self.arnoldi_m: int | None = None
        self.transition_backend = "eager"
        self.interface_mode = "static"
        self.interface_temperature = 1.0
        self.exchange_rank = 0
        self.exchange_eta = float(exchange_eta)
        self.exchange_active = True
        self.exchange_constant_input = False
        self.q9_mode = coefficient_compiler_mode != "canonical_hidden"
        self.content_address_kind = "none"
        self.content_address_rank = 0
        self.local_cross_fastpath = True
        self.internal_scene_mass_controller = None
        self.per_state_direction_mass_controller = None
        self.content_alpha_context = None
        self.content_alpha_scale_context = None
        self.content_rank_scene_context = None
        self.coefficient_gain_delta_context = None

        # This assignment gives the historical actual-parameter ordering.
        self.coeff_gamma = (
            nn.Parameter(torch.zeros(self.state_channels))
            if coeff_generator_depth == 2 else None
        )
        if coeff_generator_depth == 1:
            self.register_parameter("coeff_gamma", None)

        d = self.state_channels
        self.Bin = nn.Conv2d(channels, 2 * d, 1, bias=False)
        if self.q9_mode:
            self.coeff_in = None
            self.coeff_dw = None
            self.coeff_dw2 = None
            self.coeff_out = None
            from .coefficient_compilers import ComplexF9Q9Compiler, Q9RealReadoutHead
            if q9_generator is None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(q9_init_seed)
                    self.q9_generator = _new_q9_generator(d, coefficient_compiler_mode)
                bound_generator = self.q9_generator
            else:
                object.__setattr__(self, "_q9_generator_ref", q9_generator)
                bound_generator = q9_generator
            if q9_head is None:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(q9_init_seed + 1)
                    self.q9_head = Q9RealReadoutHead(d)
                bound_head = self.q9_head
            else:
                object.__setattr__(self, "_q9_head_ref", q9_head)
                bound_head = q9_head
            self.coefficient_compiler = ComplexF9Q9Compiler(
                bound_generator, bound_head, rho_max=self.rho_max
            )
        else:
            coeff_input_channels = 2 * d if variant == "f9_propagated" else channels
            self.coeff_in = nn.Conv2d(coeff_input_channels, d, 1, bias=False)
            self.coeff_dw = nn.Conv2d(d, d, 3, padding=1, groups=d, bias=True)
            self.coeff_dw2 = (
                nn.Conv2d(d, d, 3, padding=1, groups=d, bias=True)
                if coeff_generator_depth == 2 else None
            )
            self.coeff_out = nn.Conv2d(d, 2 * self.tap_count * d, 1, bias=True)
            self.coefficient_compiler = None
        self.Cout = nn.Conv2d(2 * d, channels, 1, bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.coefficient_compiler is not None:
            return
        if self.coeff_out is None:
            raise AssertionError("canonical coefficient output is missing")
        nn.init.normal_(self.coeff_out.weight, mean=0.0, std=1.0e-3)
        with torch.no_grad():
            bias = self.coeff_out.bias.view(2, self.tap_count, self.state_channels)
            bias.zero_()
            if self.variant == "f9_propagated":
                bias[0, 0].fill_(1.0)
                bias[0, 1:].fill_(0.5)
            else:
                bias[0].fill_(0.5)
        nn.init.zeros_(self.Cout.weight)

    def set_K(self, K: int) -> None:
        if K < 1:
            raise ValueError("K must be >= 1")
        self.K = int(K)

    def set_backward_mode(self, mode: str, arnoldi_m: int | None = None) -> None:
        if mode not in {"exact_unrolled", "finite_exact_adjoint", "finite_arnoldi"}:
            raise ValueError(f"unsupported backward mode={mode!r}")
        if mode != "finite_arnoldi" and arnoldi_m is not None:
            raise ValueError(f"arnoldi_m must be omitted for {mode} backward")
        if mode == "finite_arnoldi" and (arnoldi_m is None or arnoldi_m < 1):
            raise ValueError("finite_arnoldi requires a positive arnoldi_m")
        self.backward_mode = mode
        self.arnoldi_m = int(arnoldi_m) if arnoldi_m is not None else None

    def set_transition_backend(self, backend: str) -> None:
        if backend not in {"eager", "compiled_f9"}:
            raise ValueError(f"unsupported transition backend={backend!r}")
        self.transition_backend = backend

    def set_local_cross_fastpath(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        self.local_cross_fastpath = enabled

    def _interface_attention(self, x: Tensor) -> Tensor | None:
        if self.interface_mode != "static":
            raise RuntimeError("dynamic interfaces are not part of the release operator")
        return None

    def _drive_with_attention(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        packed = self.Bin(x)
        fr, fi = packed.chunk(2, dim=1)
        return packed, fr, fi, self._interface_attention(x)

    def _interface_readout(self, value: Tensor, attention: Tensor | None) -> Tensor:
        if attention is not None:
            raise RuntimeError("dynamic interfaces are not part of the release operator")
        return self.Cout(value)

    def drive(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        packed, fr, fi, _ = self._drive_with_attention(x)
        return packed, fr, fi

    def _coefficient_hidden(self, x: Tensor, packed_f: Tensor) -> Tensor:
        if self.coeff_in is None or self.coeff_dw is None:
            raise RuntimeError("hidden coefficient path is unavailable")
        source = packed_f if self.variant == "f9_propagated" else x
        z1 = F.gelu(self.coeff_dw(self.coeff_in(source)))
        if self.coeff_generator_depth == 1:
            return z1
        if self.coeff_dw2 is None or self.coeff_gamma is None:
            raise AssertionError("depth-2 coefficient generator is incomplete")
        z2 = F.gelu(self.coeff_dw2(z1))
        return z1 + self.coeff_gamma.view(1, -1, 1, 1) * z2

    @staticmethod
    def _complex_magnitude(real: Tensor, imag: Tensor) -> Tensor:
        squared = real.square() + imag.square()
        zero = squared == 0
        # Avoid sqrt'(0) followed by 0 * inf in backward. The forward
        # magnitude is unchanged; choose the zero subgradient at the origin.
        protected = torch.where(zero, torch.ones_like(squared), squared)
        return torch.where(zero, torch.zeros_like(squared), torch.sqrt(protected))

    def _diagonal_coefficients_from_raw(self, raw: Tensor, **_: Any) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = raw.shape
        expected = 2 * self.tap_count * self.state_channels
        if channels != expected:
            raise ValueError(f"raw coefficient channels must equal {expected}, got {channels}")
        raw = raw.view(batch, 2, self.tap_count, self.state_channels, height, width)
        raw_real, raw_imag = raw[:, 0], raw[:, 1]
        magnitude = self._complex_magnitude(raw_real, raw_imag)
        raw_mass = magnitude.sum(dim=1, keepdim=True)
        scale = self.rho_max / (1.0 + raw_mass)
        return raw_real * scale, raw_imag * scale

    def _diagonal_coefficients_from_hidden(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if self.coeff_out is None:
            raise RuntimeError("hidden coefficient output is unavailable")
        return self._diagonal_coefficients_from_raw(self.coeff_out(hidden))

    def _raw_coefficients(self, x: Tensor, packed_f: Tensor) -> Tensor:
        if self.coefficient_compiler is not None:
            return self.coefficient_compiler(packed_f)
        hidden = self._coefficient_hidden(x, packed_f)
        if self.coeff_out is None:
            raise AssertionError("canonical coefficient output is missing")
        return self.coeff_out(hidden)

    def _q9_coefficients_from_raw(self, raw: Tensor, *, return_row_mass: bool = False):
        if raw.ndim != 4 or raw.shape[1] != 9 * self.state_channels:
            raise ValueError("q9 raw coordinates must have shape [B,9D,H,W]")
        batch, _, height, width = raw.shape
        coordinates = raw.view(batch, 9, self.state_channels, height, width)
        center_logit = coordinates[:, 0]
        pair_real = torch.stack((coordinates[:, 1], coordinates[:, 3], coordinates[:, 5], coordinates[:, 7]), dim=1)
        pair_imag = torch.stack((coordinates[:, 2], coordinates[:, 4], coordinates[:, 6], coordinates[:, 8]), dim=1)
        pair_magnitude = torch.sqrt(pair_real.square() + pair_imag.square() + torch.finfo(raw.dtype).eps)
        raw_pair_mass = 2.0 * pair_magnitude.sum(dim=1)
        center_floor = 0.10
        numerical_margin = 32.0 * torch.finfo(raw.dtype).eps
        safe_rho = self.rho_max - numerical_margin
        scale = 0.5 * (safe_rho - center_floor) / (1.0 + raw_pair_mass)
        pair_real = pair_real * scale.unsqueeze(1)
        pair_imag = pair_imag * scale.unsqueeze(1)
        off_mass = scale * raw_pair_mass
        center = center_floor + off_mass + (safe_rho - center_floor - 2.0 * off_mass) * torch.sigmoid(center_logit).clamp(
            min=torch.finfo(raw.dtype).eps, max=1.0 - torch.finfo(raw.dtype).eps
        )
        zero = torch.zeros_like(center)
        ax_real, ay_real, d1_real, d2_real = pair_real.unbind(dim=1)
        ax_imag, ay_imag, d1_imag, d2_imag = pair_imag.unbind(dim=1)
        ar = torch.stack((center, d1_real, ay_real, d2_real, ax_real, ax_real, d2_real, ay_real, d1_real), dim=1)
        ai = torch.stack((zero, -d1_imag, -ay_imag, -d2_imag, -ax_imag, ax_imag, d2_imag, ay_imag, d1_imag), dim=1)
        if return_row_mass:
            return ar, ai, center + off_mass
        return ar, ai

    def _apply_stencil(self, zr: Tensor, zi: Tensor, ar: Tensor, ai: Tensor) -> tuple[Tensor, Tensor]:
        height, width = zr.shape[-2:]
        padded_r = F.pad(zr, (1, 1, 1, 1), mode="constant", value=0.0)
        padded_i = F.pad(zi, (1, 1, 1, 1), mode="constant", value=0.0)
        out_r = torch.zeros_like(zr)
        out_i = torch.zeros_like(zi)
        for tap, (dy, dx) in enumerate(self.offsets):
            y0, x0 = 1 + dy, 1 + dx
            sr = padded_r[:, :, y0:y0 + height, x0:x0 + width]
            si = padded_i[:, :, y0:y0 + height, x0:x0 + width]
            tap_r, tap_i = ar[:, tap], ai[:, tap]
            out_r = out_r + tap_r * sr - tap_i * si
            out_i = out_i + tap_r * si + tap_i * sr
        return out_r, out_i

    def apply_transition(self, zr: Tensor, zi: Tensor, ar: Tensor, ai: Tensor, *_: Any, **__: Any) -> tuple[Tensor, Tensor]:
        return self._apply_stencil(zr, zi, ar, ai)

    def _transition_components(self, x: Tensor, packed_f: Tensor):
        if self.coefficient_compiler is None:
            ar, ai = self._diagonal_coefficients_from_hidden(self._coefficient_hidden(x, packed_f))
        else:
            ar, ai = self._q9_coefficients_from_raw(self._raw_coefficients(x, packed_f))
        return ar, ai, None, None, None, None, None

    def _propagated_state_with_attention(self, x: Tensor):
        packed_f, fr, fi, attention = self._drive_with_attention(x)
        ar, ai, *_ = self._transition_components(x, packed_f)
        pr, pi = torch.zeros_like(fr), torch.zeros_like(fi)
        for _ in range(self.K - 1):
            pr, pi = self.apply_transition(fr + pr, fi + pi, ar, ai)
        return fr, fi, pr, pi, torch.sqrt(ar.square() + ai.square()).sum(dim=1), attention

    def propagated_state(self, x: Tensor):
        return self._propagated_state_with_attention(x)[:5]

    def forward(self, x: Tensor) -> Tensor:
        if self.backward_mode != "exact_unrolled":
            raise RuntimeError(f"backward mode {self.backward_mode!r} requires its carrier forward")
        fr, fi, pr, pi, _, attention = self._propagated_state_with_attention(x)
        if self.variant == "x8_full":
            read_r, read_i = fr + pr, fi + pi
        else:
            read_r, read_i = pr, pi
        return self._interface_readout(torch.cat((read_r, read_i), dim=1), attention)


def _packed_transition(op: ComplexStateOperator, packed_state: Tensor, ar: Tensor, ai: Tensor, *args: Any, **kwargs: Any) -> Tensor:
    zr, zi = packed_state.chunk(2, dim=1)
    out_r, out_i = op.apply_transition(zr, zi, ar, ai, *args, **kwargs)
    return torch.cat((out_r, out_i), dim=1)


def _stencil_adjoint(op: ComplexStateOperator, packed_vector: Tensor, ar: Tensor, ai: Tensor) -> Tensor:
    vr, vi = packed_vector.chunk(2, dim=1)
    out_r, out_i = torch.zeros_like(vr), torch.zeros_like(vi)
    for tap, (dy, dx) in enumerate(op.offsets):
        tap_r, tap_i = ar[:, tap], ai[:, tap]
        local_r = tap_r * vr + tap_i * vi
        local_i = -tap_i * vr + tap_r * vi
        out_r = out_r + op._shift_zero(local_r, -dy, -dx)
        out_i = out_i + op._shift_zero(local_i, -dy, -dx)
    return torch.cat((out_r, out_i), dim=1)


def _transition_adjoint(op: ComplexStateOperator, packed_vector: Tensor, ar: Tensor, ai: Tensor, *_: Any, **__: Any) -> Tensor:
    return _stencil_adjoint(op, packed_vector, ar, ai)


def _shift_zero_impl(x: Tensor, dy: int, dx: int) -> Tensor:
    height, width = x.shape[-2:]
    padded = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)
    y0, x0 = 1 + dy, 1 + dx
    return padded[:, :, y0:y0 + height, x0:x0 + width]


ComplexStateOperator._shift_zero = staticmethod(_shift_zero_impl)


@torch.no_grad()
def _finite_trajectory(op: ComplexStateOperator, packed_f: Tensor, ar: Tensor, ai: Tensor, transition_count: int, *_: Any, **__: Any):
    fr, fi = packed_f.chunk(2, dim=1)
    pr, pi = torch.zeros_like(fr), torch.zeros_like(fi)
    states: list[Tensor] = []
    for _ in range(transition_count):
        zr, zi = fr + pr, fi + pi
        states.append(torch.cat((zr, zi), dim=1))
        pr, pi = op.apply_transition(zr, zi, ar, ai)
    trajectory = torch.stack(states, dim=0) if states else packed_f.new_empty((0, *packed_f.shape))
    return torch.cat((pr, pi), dim=1), trajectory


@torch.no_grad()
def _stencil_coefficient_gradients(op: ComplexStateOperator, trajectory: Tensor, aligned_adjoint: Tensor, ar: Tensor, ai: Tensor):
    if trajectory.shape != aligned_adjoint.shape:
        raise ValueError("finite exact adjoint state/adjoint shape mismatch")
    steps, batch, packed_channels, height, width = trajectory.shape
    states = trajectory.reshape(steps * batch, packed_channels, height, width)
    adjoints = aligned_adjoint.reshape(steps * batch, packed_channels, height, width)
    hr, hi = states.chunk(2, dim=1)
    lr, li = adjoints.chunk(2, dim=1)
    d = hr.shape[1]
    lr = lr.reshape(steps, batch, d, height, width)
    li = li.reshape(steps, batch, d, height, width)
    padded_hr = F.pad(hr, (1, 1, 1, 1), mode="constant", value=0.0)
    padded_hi = F.pad(hi, (1, 1, 1, 1), mode="constant", value=0.0)
    grad_ar = torch.zeros_like(ar)
    grad_ai = torch.zeros_like(ai)
    for tap, (dy, dx) in enumerate(op.offsets):
        y0, x0 = 1 + dy, 1 + dx
        shifted_r = padded_hr[:, :, y0:y0 + height, x0:x0 + width].reshape(steps, batch, d, height, width)
        shifted_i = padded_hi[:, :, y0:y0 + height, x0:x0 + width].reshape(steps, batch, d, height, width)
        grad_ar[:, tap] = (lr * shifted_r + li * shifted_i).sum(dim=0)
        grad_ai[:, tap] = (-lr * shifted_i + li * shifted_r).sum(dim=0)
    return grad_ar, grad_ai


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "ComplexStateOperator",
    "N8_OFFSETS",
    "N9_OFFSETS",
    "VALID_VARIANTS",
    "_finite_trajectory",
    "_new_q9_generator",
    "_stencil_coefficient_gradients",
    "_transition_adjoint",
    "count_parameters",
]

