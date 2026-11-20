"""Complex-aware coefficient compilers for the fixed CLPO F9 transition.

This module intentionally contains no transition or recurrence logic.  A
compiler is evaluated once from the packed complex drive ``f`` and returns the
raw nine-tap complex rows consumed by the existing canonical projection.
"""

from __future__ import annotations

from contextlib import contextmanager
import math
import types
from typing import Any, Callable, Iterator, Sequence

import torch
from torch import Tensor, nn

from .blackwell_low3_dct import direct_low3_dct, direct_low3_idct
import torch.nn.functional as F

from .givens_transport import givens_strang_transport
from .glrc9_role_film import Q9RoleFiLMResidual
from .glrc9_routing_continuation import glrc9_routing_conditioning
from .glrc9_local_family_continuation import scale_local_family_residual
from .glrc9_mn_offdiag_continuation import (
    GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR,
    mn_offdiag_diagonal_pointwise,
    mn_offdiag_materialized_reference_pointwise,
    mn_offdiag_continuation_pointwise,
)


# Read-only response-geometry probe.  ``None`` is the production default and
# therefore leaves the GLRC9 compiler bit-for-bit unchanged.  The CPU audit sets
# this context temporarily to scale the four non-anchor global shell responses
# and the four local directional responses before the unchanged H readout and
# q9 legalizer.  Keeping the hook here (rather than editing checkpoint weights)
# makes the probe an explicit semantic interpolation and lets the downstream
# legal action projection remain part of every measured output.
_GLRC9_RESPONSE_SCALE_CONTEXT: tuple[float, float] | None = None

# Diagnostic-only source-alignment ablation for GCHAF-CCAR9.  Production and
# training leave this at ``None``.  When enabled in an eager frozen-validation
# audit, it permutes only the already decoded content code across the batch;
# the live-f routing, nine actions, parent aggregate, and private parameters
# remain aligned with their original samples.
_CCAR9_CONTENT_BATCH_PERMUTATION_CONTEXT: tuple[int, ...] | None = None


def set_glrc9_response_scale_context(
    global_scale: float, local_scale: float
) -> tuple[float, float] | None:
    """Set the temporary GLRC9 family-scale probe and return its prior value.

    The function is intentionally a tiny audit hook: callers must restore the
    returned value in a ``finally`` block.  The default ``None`` path has no
    effect on the production compiler.
    """

    global _GLRC9_RESPONSE_SCALE_CONTEXT
    values = (float(global_scale), float(local_scale))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("GLRC9 response scales must be finite")
    previous = _GLRC9_RESPONSE_SCALE_CONTEXT
    _GLRC9_RESPONSE_SCALE_CONTEXT = values
    return previous


def restore_glrc9_response_scale_context(
    previous: tuple[float, float] | None,
) -> None:
    """Restore a value returned by :func:`set_glrc9_response_scale_context`."""

    global _GLRC9_RESPONSE_SCALE_CONTEXT
    if previous is not None and not all(math.isfinite(value) for value in previous):
        raise ValueError("previous GLRC9 response scales must be finite")
    _GLRC9_RESPONSE_SCALE_CONTEXT = previous


@contextmanager
def ccar9_content_batch_permutation(
    permutation: Sequence[int],
) -> Iterator[None]:
    """Temporarily mismatch only CCAR9 content codes along the batch axis.

    This hook is diagnostic-only and defaults to disabled.  ``permutation``
    must be a complete zero-based permutation; its length is checked again
    against the live batch inside the compiler, so a stale audit setting fails
    closed rather than silently broadcasting or truncating.
    """

    global _CCAR9_CONTENT_BATCH_PERMUTATION_CONTEXT
    values = tuple(permutation)
    if any(type(value) is not int for value in values):
        raise TypeError("CCAR9 content permutation entries must be integers")
    if tuple(sorted(values)) != tuple(range(len(values))):
        raise ValueError(
            "CCAR9 content permutation must contain each batch index exactly once"
        )
    previous = _CCAR9_CONTENT_BATCH_PERMUTATION_CONTEXT
    _CCAR9_CONTENT_BATCH_PERMUTATION_CONTEXT = values
    try:
        yield
    finally:
        _CCAR9_CONTENT_BATCH_PERMUTATION_CONTEXT = previous


def _q9_grouped_head_to_tap_major(raw: Tensor, state_channels: int) -> Tensor:
    """Convert grouped-H ``[state,tap]`` rows to q9 ``[tap,state]`` rows."""

    if raw.ndim != 4:
        raise ValueError("q9 grouped head output must have shape [B,9D,H,W]")
    batch, channels, height, width = raw.shape
    if channels != 9 * state_channels:
        raise ValueError("q9 grouped head output channels must equal 9 * D")
    return (
        raw.view(batch, state_channels, 9, height, width)
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, 9 * state_channels, height, width)
    )


def _q9_grouped_head_to_coordinate_major(
    raw: Tensor, state_channels: int
) -> Tensor:
    """Return grouped H output as explicit ``[B,9,D,H,W]`` coordinates."""

    if raw.ndim != 4:
        raise ValueError("q9 grouped head output must have shape [B,9D,H,W]")
    batch, channels, height, width = raw.shape
    if channels != 9 * state_channels:
        raise ValueError("q9 grouped head output channels must equal 9 * D")
    return raw.view(batch, state_channels, 9, height, width).permute(
        0, 2, 1, 3, 4
    )


class _MomentExponentialInverseScale(torch.autograd.Function):
    """Production scale inverse with the exact implicit VJP.

    This Function intentionally has no custom JVP: the production M2/CM-I
    chart is traced by ``torch.compile(fullgraph=True)``.  The forward-mode
    rule lives in the explicitly audit-only sibling below.
    """

    @staticmethod
    def forward(
        z_real: Tensor,
        z_imag: Tensor,
        contraction_budget: float,
        epsilon: float,
    ) -> Tensor:
        with torch.no_grad():
            lo = torch.zeros_like(z_real[:, :1])
            hi = torch.full_like(lo, float(contraction_budget))
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                w = torch.sqrt(
                    z_real.square()
                    + z_imag.square()
                    + float(epsilon) * mid.square()
                )
                residual = (
                    mid
                    + 2.0 * w.sum(dim=1, keepdim=True)
                    - float(contraction_budget)
                )
                positive = residual > 0.0
                hi = torch.where(positive, mid, hi)
                lo = torch.where(positive, lo, mid)
            scale = 0.5 * (lo + hi)
        return scale

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Tensor, Tensor, float, float],
        output: Tensor,
    ) -> None:
        z_real, z_imag, _contraction_budget, epsilon = inputs
        # Keep the historical first-order VJP independent of higher-order
        # derivatives.
        ctx.save_for_backward(z_real.detach(), z_imag.detach(), output)
        ctx.epsilon = float(epsilon)

    @staticmethod
    def backward(
        ctx: Any, grad_scale: Tensor
    ) -> tuple[Tensor, Tensor, None, None]:
        z_real, z_imag, scale = ctx.saved_tensors
        epsilon = ctx.epsilon
        w = torch.sqrt(
            z_real.square() + z_imag.square() + epsilon * scale.square()
        )
        denominator = 1.0 + 2.0 * (epsilon * scale / w).sum(
            dim=1, keepdim=True
        )
        factor = -2.0 * grad_scale / denominator
        return factor * z_real / w, factor * z_imag / w, None, None


class _MomentExponentialInverseScaleAudit(torch.autograd.Function):
    """Audit-only scale inverse with the exact implicit VJP and JVP."""

    @staticmethod
    def forward(
        z_real: Tensor,
        z_imag: Tensor,
        contraction_budget: float,
        epsilon: float,
    ) -> Tensor:
        # Calling the production forward directly keeps the two variants
        # mathematically identical while registering a distinct Function
        # whose JVP is never exposed to the production compiled graph.
        return _MomentExponentialInverseScale.forward(
            z_real, z_imag, contraction_budget, epsilon
        )

    @staticmethod
    def setup_context(
        ctx: Any,
        inputs: tuple[Tensor, Tensor, float, float],
        output: Tensor,
    ) -> None:
        z_real, z_imag, _contraction_budget, epsilon = inputs
        ctx.save_for_backward(z_real.detach(), z_imag.detach(), output)
        ctx.save_for_forward(z_real, z_imag, output)
        ctx.epsilon = float(epsilon)

    @staticmethod
    def backward(
        ctx: Any, grad_scale: Tensor
    ) -> tuple[Tensor, Tensor, None, None]:
        return _MomentExponentialInverseScale.backward(ctx, grad_scale)

    @staticmethod
    def jvp(
        ctx: Any,
        z_real_tangent: Tensor,
        z_imag_tangent: Tensor,
        _contraction_budget_tangent: None,
        _epsilon_tangent: None,
    ) -> Tensor:
        z_real, z_imag, scale = ctx.saved_tensors
        epsilon = ctx.epsilon
        w = torch.sqrt(
            z_real.square() + z_imag.square() + epsilon * scale.square()
        )
        denominator = 1.0 + 2.0 * (epsilon * scale / w).sum(
            dim=1, keepdim=True
        )
        numerator = -2.0 * (
            (z_real * z_real_tangent + z_imag * z_imag_tangent) / w
        ).sum(dim=1, keepdim=True)
        return numerator / denominator


def _q9_moment_exponential_raw_impl(
    coordinate_major: Tensor,
    rho_max: float = 0.95,
    *,
    inverse_scale: Any,
) -> Tensor:
    """Map M2-9 natural coordinates to raw q9 tap-major coordinates."""

    if coordinate_major.ndim != 5 or coordinate_major.shape[1] != 9:
        raise ValueError("M2-9 coordinates must have shape [B,9,D,H,W]")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    epsilon = torch.finfo(coordinate_major.dtype).eps
    safe_rho = float(rho_max) - 32.0 * epsilon
    center_floor = 0.10
    contraction_budget = 0.5 * (safe_rho - center_floor)
    (
        ell_mass,
        alpha,
        beta,
        gamma,
        omega_x,
        omega_y,
        nu,
        upsilon,
        kappa,
    ) = coordinate_major.unbind(dim=1)
    delta_x = coordinate_major.new_tensor((1.0, 0.0, 1.0, 1.0)).view(
        1, 4, 1, 1, 1
    )
    delta_y = coordinate_major.new_tensor((0.0, 1.0, 1.0, -1.0)).view(
        1, 4, 1, 1, 1
    )
    delta_norm_sq = delta_x.square() + delta_y.square()
    delta_quadrupole = delta_x.square() - delta_y.square()
    delta_cross = delta_x * delta_y
    score = (
        alpha.unsqueeze(1) * delta_norm_sq
        + beta.unsqueeze(1) * delta_quadrupole
        + 2.0 * gamma.unsqueeze(1) * delta_cross
    )
    probability = torch.softmax(score, dim=1)
    phase = (
        omega_x.unsqueeze(1) * delta_x
        + omega_y.unsqueeze(1) * delta_y
        + nu.unsqueeze(1) * delta_quadrupole
        + 2.0 * upsilon.unsqueeze(1) * delta_cross
    )
    total_mass = 0.5 * contraction_budget * torch.sigmoid(ell_mass)
    z_real = total_mass.unsqueeze(1) * probability * torch.cos(phase)
    z_imag = total_mass.unsqueeze(1) * probability * torch.sin(phase)
    scale = inverse_scale.apply(
        z_real,
        z_imag,
        contraction_budget,
        epsilon,
    )[:, 0]
    # The physical center is defined by c=mu+mtilde+2s*sigmoid(kappa).
    # Therefore the exact physical-to-raw inverse is r_c=kappa; rebuilding c
    # and subtracting mtilde/s only introduces avoidable endpoint rounding.
    raw_center = kappa
    return torch.stack(
        (
            raw_center,
            z_real[:, 0] / scale,
            z_imag[:, 0] / scale,
            z_real[:, 1] / scale,
            z_imag[:, 1] / scale,
            z_real[:, 2] / scale,
            z_imag[:, 2] / scale,
            z_real[:, 3] / scale,
            z_imag[:, 3] / scale,
        ),
        dim=1,
    )


def _q9_moment_exponential_raw(
    coordinate_major: Tensor,
    rho_max: float = 0.95,
) -> Tensor:
    """Map M2-9 coordinates through the production compiled chart."""

    return _q9_moment_exponential_raw_impl(
        coordinate_major,
        rho_max,
        inverse_scale=_MomentExponentialInverseScale,
    )


def _q9_moment_exponential_raw_audit(
    coordinate_major: Tensor,
    rho_max: float = 0.95,
) -> Tensor:
    """Map M2-9 coordinates through the exact audit replay chart."""

    return _q9_moment_exponential_raw_impl(
        coordinate_major,
        rho_max,
        inverse_scale=_MomentExponentialInverseScaleAudit,
    )


def _q9_logpolar_raw(coordinate_major: Tensor) -> Tensor:
    """Map LP-9 log-radius/phase coordinates to raw q9 coordinates.

    The chart is intentionally raw-first: the center is the unconstrained
    real coordinate ``kappa`` and each off-center pair is ``exp(u + i phi)``.
    The caller then applies the unchanged production q9 legalizer.
    """

    if coordinate_major.ndim != 5 or coordinate_major.shape[1] != 9:
        raise ValueError("LP-9 coordinates must have shape [B,9,D,H,W]")
    kappa, ux, uy, ud1, ud2, phix, phiy, phid1, phid2 = (
        coordinate_major.unbind(dim=1)
    )
    radii = torch.exp(torch.stack((ux, uy, ud1, ud2), dim=1))
    phases = torch.stack((phix, phiy, phid1, phid2), dim=1)
    pair_real = radii * torch.cos(phases)
    pair_imag = radii * torch.sin(phases)
    return torch.stack(
        (
            kappa,
            pair_real[:, 0],
            pair_imag[:, 0],
            pair_real[:, 1],
            pair_imag[:, 1],
            pair_real[:, 2],
            pair_imag[:, 2],
            pair_real[:, 3],
            pair_imag[:, 3],
        ),
        dim=1,
    )


def _q9_mass_simplex_phase_raw(coordinate_major: Tensor) -> Tensor:
    """Map AP-S9 mass/simplex/phase coordinates to raw q9 coordinates."""

    if coordinate_major.ndim != 5 or coordinate_major.shape[1] != 9:
        raise ValueError("AP-S9 coordinates must have shape [B,9,D,H,W]")
    (
        ell_tau,
        a_x,
        a_y,
        a_d1,
        phi_x,
        phi_y,
        phi_d1,
        phi_d2,
        kappa,
    ) = coordinate_major.unbind(dim=1)
    scores = torch.stack(
        (a_x, a_y, a_d1, torch.zeros_like(a_d1)),
        dim=1,
    )
    probability = torch.softmax(scores, dim=1)
    phase = torch.stack((phi_x, phi_y, phi_d1, phi_d2), dim=1)
    mass = torch.sigmoid(ell_tau).unsqueeze(1) * probability
    pair_real = mass * torch.cos(phase)
    pair_imag = mass * torch.sin(phase)
    return torch.stack(
        (
            kappa,
            pair_real[:, 0],
            pair_imag[:, 0],
            pair_real[:, 1],
            pair_imag[:, 1],
            pair_real[:, 2],
            pair_imag[:, 2],
            pair_real[:, 3],
            pair_imag[:, 3],
        ),
        dim=1,
    )


def _q9_center_mass_inverse_raw_impl(
    coordinate_major: Tensor,
    rho_max: float = 0.95,
    *,
    inverse_scale: Any,
) -> Tensor:
    """Map CM-I9 physical coordinates through the epsilon-aware right inverse."""

    if coordinate_major.ndim != 5 or coordinate_major.shape[1] != 9:
        raise ValueError("CM-I9 coordinates must have shape [B,9,D,H,W]")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    epsilon = torch.finfo(coordinate_major.dtype).eps
    safe_rho = float(rho_max) - 32.0 * epsilon
    center_floor = 0.10
    contraction_budget = 0.5 * (safe_rho - center_floor)
    (
        ell_mass,
        alpha_x,
        alpha_y,
        alpha_d1,
        phi_x,
        phi_y,
        phi_d1,
        phi_d2,
        kappa,
    ) = coordinate_major.unbind(dim=1)
    probability = torch.softmax(
        torch.stack(
            (alpha_x, alpha_y, alpha_d1, torch.zeros_like(alpha_d1)), dim=1
        ),
        dim=1,
    )
    phase = torch.stack((phi_x, phi_y, phi_d1, phi_d2), dim=1)
    total_mass = contraction_budget * torch.sigmoid(ell_mass)
    z_real = 0.5 * total_mass.unsqueeze(1) * probability * torch.cos(phase)
    z_imag = 0.5 * total_mass.unsqueeze(1) * probability * torch.sin(phase)
    scale = inverse_scale.apply(
        z_real,
        z_imag,
        contraction_budget,
        epsilon,
    )[:, 0]
    return torch.stack(
        (
            kappa,
            z_real[:, 0] / scale,
            z_imag[:, 0] / scale,
            z_real[:, 1] / scale,
            z_imag[:, 1] / scale,
            z_real[:, 2] / scale,
            z_imag[:, 2] / scale,
            z_real[:, 3] / scale,
            z_imag[:, 3] / scale,
        ),
        dim=1,
    )


def _q9_center_mass_inverse_raw(
    coordinate_major: Tensor,
    rho_max: float = 0.95,
) -> Tensor:
    """Map CM-I9 coordinates through the production compiled chart."""

    return _q9_center_mass_inverse_raw_impl(
        coordinate_major,
        rho_max,
        inverse_scale=_MomentExponentialInverseScale,
    )


def _q9_center_mass_inverse_raw_audit(
    coordinate_major: Tensor,
    rho_max: float = 0.95,
) -> Tensor:
    """Map CM-I9 coordinates through the exact audit replay chart."""

    return _q9_center_mass_inverse_raw_impl(
        coordinate_major,
        rho_max,
        inverse_scale=_MomentExponentialInverseScaleAudit,
    )


def paired_real_to_complex(packed: Tensor) -> Tensor:
    """Convert ``[real channels, imaginary channels]`` to a complex tensor."""

    if packed.ndim < 2 or packed.shape[1] % 2:
        raise ValueError(
            "packed complex tensor must have an even channel dimension"
        )
    if not packed.is_floating_point():
        raise TypeError("packed complex tensor must use a real floating dtype")
    real, imag = packed.chunk(2, dim=1)
    return torch.complex(real, imag)


def complex_to_paired_real(value: Tensor) -> Tensor:
    """Convert a complex tensor to ``[real channels, imaginary channels]``."""

    if not value.is_complex():
        raise TypeError("value must use a complex dtype")
    return torch.cat((value.real, value.imag), dim=1)


def _fused_complex_jet_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    expand_real: Tensor,
    expand_imag: Tensor,
    output_bias: Tensor,
) -> Tensor:
    """Functional three-convolution Jet with no full activation repacking."""

    d = mix_m_real.shape[0]
    tap_count = expand_real.shape[0] // d
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag),
        dim=1,
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real),
        dim=1,
    )
    pointwise_weight = torch.stack(
        (real_rows, imag_rows),
        dim=1,
    ).reshape(2 * d, 2 * d, 1, 1)
    interleaved = F.conv2d(packed_f, pointwise_weight)

    depthwise_wr = depthwise_real[:, 0]
    depthwise_wi = depthwise_imag[:, 0]
    depthwise_real_rows = torch.stack(
        (depthwise_wr, -depthwise_wi),
        dim=1,
    )
    depthwise_imag_rows = torch.stack(
        (depthwise_wi, depthwise_wr),
        dim=1,
    )
    depthwise_weight = torch.stack(
        (depthwise_real_rows, depthwise_imag_rows),
        dim=1,
    ).reshape(2 * d, 2, 3, 3)
    interleaved = F.conv2d(
        interleaved,
        depthwise_weight,
        padding=1,
        groups=d,
    )

    batch, _, height, width = interleaved.shape
    paired = interleaved.view(batch, d, 2, height, width)
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    gain = F.silu(magnitude) / magnitude
    interleaved = (paired * gain.unsqueeze(2)).reshape(
        batch,
        2 * d,
        height,
        width,
    )

    expand_wr = expand_real.view(d, tap_count, 3, 3)
    expand_wi = expand_imag.view(d, tap_count, 3, 3)
    expand_real_rows = torch.stack((expand_wr, -expand_wi), dim=2)
    expand_imag_rows = torch.stack((expand_wi, expand_wr), dim=2)
    expand_weight = torch.stack(
        (expand_real_rows, expand_imag_rows),
        dim=2,
    ).reshape(2 * tap_count * d, 2, 3, 3)
    expanded = F.conv2d(
        interleaved,
        expand_weight,
        padding=1,
        groups=d,
    ).view(batch, d, tap_count, 2, height, width)
    real = expanded[:, :, :, 0].permute(0, 2, 1, 3, 4)
    imag = expanded[:, :, :, 1].permute(0, 2, 1, 3, 4)
    bias = output_bias.to(dtype=real.dtype)
    real = real + bias[0][None, :, :, None, None]
    imag = imag + bias[1][None, :, :, None, None]
    return torch.cat(
        (
            real.reshape(batch, tap_count * d, height, width),
            imag.reshape(batch, tap_count * d, height, width),
        ),
        dim=1,
    )


_COMPILED_FUSED_COMPLEX_JET_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}


def _fused_q9_compiler_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """Fused real graph for the q9 compiler's ``G`` followed by ``H``.

    ``G`` is the same real-linear complex pointwise map, complex depthwise
    3x3 jet and phase-preserving radial SiLU used by the direct Jet.  ``H``
    is a grouped real 3x3 readout from each complex latent state to nine
    real q9 coordinates.  Keeping this graph explicit avoids materializing
    the intermediate real/imaginary tensors between the compiler stages.
    """

    d = mix_m_real.shape[0]
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag),
        dim=1,
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real),
        dim=1,
    )
    pointwise_weight = torch.stack(
        (real_rows, imag_rows),
        dim=1,
    ).reshape(2 * d, 2 * d, 1, 1)
    interleaved = F.conv2d(packed_f, pointwise_weight)

    depthwise_wr = depthwise_real[:, 0]
    depthwise_wi = depthwise_imag[:, 0]
    depthwise_real_rows = torch.stack(
        (depthwise_wr, -depthwise_wi),
        dim=1,
    )
    depthwise_imag_rows = torch.stack(
        (depthwise_wi, depthwise_wr),
        dim=1,
    )
    depthwise_weight = torch.stack(
        (depthwise_real_rows, depthwise_imag_rows),
        dim=1,
    ).reshape(2 * d, 2, 3, 3)
    interleaved = F.conv2d(
        interleaved,
        depthwise_weight,
        padding=1,
        groups=d,
    )

    batch, _, height, width = interleaved.shape
    paired = interleaved.view(batch, d, 2, height, width)
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    gain = F.silu(magnitude) / magnitude
    interleaved = (paired * gain.unsqueeze(2)).reshape(
        batch,
        2 * d,
        height,
        width,
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _nk9_landmark_weights(q: Tensor) -> Tensor:
    """Return the bounded four-landmark NK-9 interpolation weights.

    ``q`` is the two-channel real/imaginary channel-mean field.  The solve is
    deliberately kept in FP32 and uses ``solve_ex`` rather than forming an
    inverse.  The returned matrix has shape ``[B, HW, 4]`` and strict
    per-location L1 norm below one.
    """

    if q.ndim != 4 or q.shape[1] != 2:
        raise ValueError("NK-9 q must have shape [B,2,H,W]")
    q32 = q.float()
    batch, _, height, width = q32.shape
    query = q32.permute(0, 2, 3, 1).reshape(batch, height * width, 2)
    landmarks = F.adaptive_avg_pool2d(q32, (2, 2)).reshape(
        batch, 2, 4
    ).transpose(1, 2).contiguous()

    sigma2 = 1.0
    kpm = torch.exp(
        -0.5
        * (query[:, :, None, :] - landmarks[:, None, :, :])
        .square()
        .sum(dim=-1)
        / sigma2
    )
    kmm = torch.exp(
        -0.5
        * (landmarks[:, :, None, :] - landmarks[:, None, :, :])
        .square()
        .sum(dim=-1)
        / sigma2
    )
    eye = torch.eye(4, device=q.device, dtype=torch.float32).expand(
        batch, -1, -1
    )
    system = kmm + 0.1 * eye
    solved, info = torch.linalg.solve_ex(
        system.transpose(-1, -2), kpm.transpose(-1, -2)
    )
    if bool(torch.any(info != 0)):
        details = info.detach().cpu().tolist()
        raise RuntimeError(f"NK-9 solve_ex failed: info={details}")
    a0 = solved.transpose(-1, -2)
    return a0 / (1.0 + a0.abs().sum(dim=-1, keepdim=True))


def _nk9_content_correction(real: Tensor, imag: Tensor) -> tuple[Tensor, Tensor]:
    """Interpolate four pooled complex ``u`` landmarks using NK-9 weights."""

    q = torch.tanh(
        torch.cat(
            (
                real.mean(dim=1, keepdim=True),
                imag.mean(dim=1, keepdim=True),
            ),
            dim=1,
        )
    )
    weights = _nk9_landmark_weights(q)
    batch, d, _, _ = real.shape
    real_mu = F.adaptive_avg_pool2d(real.float(), (2, 2)).reshape(
        batch, d, 4
    )
    imag_mu = F.adaptive_avg_pool2d(imag.float(), (2, 2)).reshape(
        batch, d, 4
    )
    correction_real = torch.einsum("bpk,bck->bcp", weights, real_mu)
    correction_imag = torch.einsum("bpk,bck->bcp", weights, imag_mu)
    height, width = real.shape[-2:]
    return (
        correction_real.reshape(batch, d, height, width).to(real.dtype),
        correction_imag.reshape(batch, d, height, width).to(imag.dtype),
    )


def _fused_q9_nk9_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """NK-9 compiler graph with the existing q9 H and depthwise nine slots."""

    d = mix_m_real.shape[0]
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag), dim=1
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real), dim=1
    )
    pointwise_weight = torch.stack(
        (real_rows, imag_rows), dim=1
    ).reshape(2 * d, 2 * d, 1, 1)
    mixed = F.conv2d(packed_f, pointwise_weight)
    batch, _, height, width = mixed.shape
    paired = mixed.view(batch, d, 2, height, width)
    real = paired[:, :, 0]
    imag = paired[:, :, 1]

    correction_real, correction_imag = _nk9_content_correction(real, imag)
    value_real = real + 0.5 * correction_real
    value_imag = imag + 0.5 * correction_imag

    depthwise_value_real = (
        F.conv2d(
            value_real,
            depthwise_real,
            padding=1,
            groups=d,
        )
        - F.conv2d(
            value_imag,
            depthwise_imag,
            padding=1,
            groups=d,
        )
    )
    depthwise_value_imag = (
        F.conv2d(
            value_real,
            depthwise_imag,
            padding=1,
            groups=d,
        )
        + F.conv2d(
            value_imag,
            depthwise_real,
            padding=1,
            groups=d,
        )
    )
    # Subtract the center identity slot: T = u + (Gamma - I) v.
    transition_real = real + depthwise_value_real - value_real
    transition_imag = imag + depthwise_value_imag - value_imag

    magnitude = torch.sqrt(
        transition_real.square()
        + transition_imag.square()
        + torch.finfo(transition_real.dtype).eps
    )
    gain = F.silu(magnitude) / magnitude
    interleaved = torch.stack(
        (transition_real * gain, transition_imag * gain), dim=2
    ).reshape(batch, 2 * d, height, width)
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_phase_gauge_compiler_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """q9 compiler in an input-canonical, parameter-free complex gauge.

    Each complex state channel is rotated so its spatial mean is real and
    non-negative before the shared widely-linear pointwise map, then rotated
    back before the unchanged depthwise jet and block-local q9 head.  The
    phase is computed only from the live ``f`` supplied to this compiler.  It
    is unit-modulus (or exactly one below the fixed low-amplitude threshold),
    so ``M = I, N = 0`` remains the identity map for every input.

    This is not magnitude normalization or whitening: amplitudes, channel
    covariance, optimizer state, and task gradients are never used to scale
    the feature field.
    """

    d = mix_m_real.shape[0]
    real = packed_f[:, :d]
    imag = packed_f[:, d:]
    mean_real = real.mean(dim=(-2, -1), keepdim=True)
    mean_imag = imag.mean(dim=(-2, -1), keepdim=True)
    mean_square = mean_real.square() + mean_imag.square()
    phase_floor = 1.0e-4
    active = mean_square >= phase_floor * phase_floor
    mean_norm = torch.sqrt(
        torch.clamp(mean_square, min=torch.finfo(packed_f.dtype).tiny)
    )
    # q = conj(mean) / |mean| maps the channel mean to the positive real axis.
    gauge_real = torch.where(active, mean_real / mean_norm, torch.ones_like(mean_real))
    gauge_imag = torch.where(active, -mean_imag / mean_norm, torch.zeros_like(mean_imag))
    canonical_real = gauge_real * real - gauge_imag * imag
    canonical_imag = gauge_imag * real + gauge_real * imag
    canonical_f = torch.cat((canonical_real, canonical_imag), dim=1)

    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag),
        dim=1,
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real),
        dim=1,
    )
    pointwise_weight = torch.stack(
        (real_rows, imag_rows),
        dim=1,
    ).reshape(2 * d, 2 * d, 1, 1)
    canonical_output = F.conv2d(canonical_f, pointwise_weight)

    batch, _, height, width = canonical_output.shape
    canonical_pair = canonical_output.view(batch, d, 2, height, width)
    output_real = canonical_pair[:, :, 0]
    output_imag = canonical_pair[:, :, 1]
    # Multiply by conj(q), returning the shared map to the original frame.
    restored_real = gauge_real * output_real + gauge_imag * output_imag
    restored_imag = gauge_real * output_imag - gauge_imag * output_real
    interleaved = torch.stack((restored_real, restored_imag), dim=2).reshape(
        batch, 2 * d, height, width
    )

    depthwise_wr = depthwise_real[:, 0]
    depthwise_wi = depthwise_imag[:, 0]
    depthwise_real_rows = torch.stack(
        (depthwise_wr, -depthwise_wi),
        dim=1,
    )
    depthwise_imag_rows = torch.stack(
        (depthwise_wi, depthwise_wr),
        dim=1,
    )
    depthwise_weight = torch.stack(
        (depthwise_real_rows, depthwise_imag_rows),
        dim=1,
    ).reshape(2 * d, 2, 3, 3)
    interleaved = F.conv2d(
        interleaved,
        depthwise_weight,
        padding=1,
        groups=d,
    )

    paired = interleaved.view(batch, d, 2, height, width)
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    gain = F.silu(magnitude) / magnitude
    interleaved = (paired * gain.unsqueeze(2)).reshape(
        batch,
        2 * d,
        height,
        width,
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_cosine_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """Parameter-matched low-rank cosine integral q9 compiler.

    The parameter tensors and their initialization are identical to
    :func:`_fused_q9_compiler_graph`.  The complex ``D x 1 x 3 x 3`` spatial
    weights are instead interpreted as gains on the first nine separable,
    orthonormal DCT-II modes.  The center real coefficient is the identity
    anchor, so the registered initialization (center=1, all other entries=0)
    gives exactly the same latent identity map as the local depthwise graph:

        z -> z + sum_(m,n<3) (r_mn - delta_m0 delta_n0) <z,phi_mn> phi_mn.

    This is a finite-rank integral operator with resolution-independent
    parameters and non-periodic cosine basis functions.  It changes spatial
    parameter sharing without adding capacity, whitening coordinates, or
    touching the downstream q9 legalizer/F9 recurrence.
    """

    d = mix_m_real.shape[0]
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag),
        dim=1,
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real),
        dim=1,
    )
    pointwise_weight = torch.stack(
        (real_rows, imag_rows),
        dim=1,
    ).reshape(2 * d, 2 * d, 1, 1)
    mixed = F.conv2d(packed_f, pointwise_weight)
    batch, _, height, width = mixed.shape
    if height < 3 or width < 3:
        raise ValueError(
            "cosine_integral requires both spatial dimensions to be >= 3"
        )
    paired = mixed.view(batch, d, 2, height, width)
    mixed_real = paired[:, :, 0]
    mixed_imag = paired[:, :, 1]

    mode = torch.arange(3, device=packed_f.device, dtype=packed_f.dtype)
    y = torch.arange(height, device=packed_f.device, dtype=packed_f.dtype)
    x = torch.arange(width, device=packed_f.device, dtype=packed_f.dtype)
    scale = torch.sqrt(
        torch.where(mode == 0, torch.ones_like(mode), 2.0 * torch.ones_like(mode))
    )
    basis_y = (
        scale[:, None]
        * torch.cos(
            torch.pi * mode[:, None] * (y[None, :] + 0.5) / float(height)
        )
        / float(height) ** 0.5
    )
    basis_x = (
        scale[:, None]
        * torch.cos(
            torch.pi * mode[:, None] * (x[None, :] + 0.5) / float(width)
        )
        / float(width) ** 0.5
    )

    # Roll the registered spatial center to DCT mode (0,0).  Subtracting the
    # fixed anchor makes the default center=1 tensor an exact zero integral
    # residual while keeping every checkpoint key and scalar parameter.
    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    )
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    )
    dc_anchor = F.pad(
        torch.ones_like(gain_real[:, :1, :1]),
        (0, 2, 0, 2),
    )
    gain_real = gain_real - dc_anchor

    coefficient_real = torch.einsum(
        "bdhw,mh,nw->bdmn", mixed_real, basis_y, basis_x
    )
    coefficient_imag = torch.einsum(
        "bdhw,mh,nw->bdmn", mixed_imag, basis_y, basis_x
    )
    filtered_real = (
        gain_real[None] * coefficient_real
        - gain_imag[None] * coefficient_imag
    )
    filtered_imag = (
        gain_imag[None] * coefficient_real
        + gain_real[None] * coefficient_imag
    )
    integral_real = torch.einsum(
        "bdmn,mh,nw->bdhw", filtered_real, basis_y, basis_x
    )
    integral_imag = torch.einsum(
        "bdmn,mh,nw->bdhw", filtered_imag, basis_y, basis_x
    )
    paired = torch.stack(
        (mixed_real + integral_real, mixed_imag + integral_imag), dim=2
    )
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    gain = F.silu(magnitude) / magnitude
    interleaved = (paired * gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


_GREEN_RESOLVENT_TAUS = (
    0.25,
    1.0,
    4.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
)
_GREEN_RESOLVENT_BAND_CACHE: dict[tuple[Any, ...], Tensor] = {}
_PARSEVAL_RESOLVENT_DIRECTIONAL_BAND_CACHE: dict[tuple[Any, ...], Tensor] = {}
_DCT_HYBRID_SHELL_CACHE: dict[tuple[Any, ...], Tensor] = {}
_DCT_HYBRID_LOCAL_FRAME_CACHE: dict[tuple[Any, ...], Tensor] = {}
_DCT_GLOBAL9_COORDINATE_MASK_CACHE: dict[tuple[Any, ...], Tensor] = {}
_Q9_FIXED_U9_CHART_CACHE: dict[tuple[Any, ...], Tensor] = {}
_DH_LMS4_ADDITIVE_BAND_CACHE: dict[tuple[Any, ...], Tensor] = {}
_DH_LMS4_ACT4_BAND_CACHE: dict[tuple[Any, ...], Tensor] = {}
_DCT_ORTHO_FACTOR_CACHE: dict[
    tuple[Any, ...], tuple[Tensor, Tensor, Tensor]
] = {}


def _q9_fixed_u9_chart(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return the fixed orthogonal 1-D DCT-II chart used by G-arm."""

    key = (device.type, device.index, dtype)
    cached = _Q9_FIXED_U9_CHART_CACHE.get(key)
    if cached is not None:
        return cached
    n = 9
    row = torch.arange(n, device=device, dtype=dtype)[:, None]
    column = torch.arange(n, device=device, dtype=dtype)[None, :]
    chart = torch.cos(
        torch.pi * (column + 0.5) * row / float(n)
    ) * (2.0 / float(n)) ** 0.5
    chart[0].fill_((1.0 / float(n)) ** 0.5)
    _Q9_FIXED_U9_CHART_CACHE[key] = chart
    return chart


def _q9_transform_gain_chart(
    depthwise: Tensor,
    *,
    gain_chart: str,
) -> Tensor:
    """Map raw rolled q9 coordinates to the physical complex gains."""

    if gain_chart == "direct":
        return depthwise
    if gain_chart != "fixed_u9":
        raise ValueError(f"unsupported q9 gain chart {gain_chart!r}")
    d = depthwise.shape[0]
    chart = _q9_fixed_u9_chart(device=depthwise.device, dtype=depthwise.dtype)
    rolled = torch.roll(
        depthwise[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    physical = torch.matmul(rolled, chart.transpose(0, 1))
    return torch.roll(
        physical.reshape(d, 3, 3), shifts=(1, 1), dims=(-2, -1)
    )[:, None]


def _dct_ortho_factors(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Cache the fixed phase and normalization of one orthonormal DCT axis."""

    key = (length, device.type, device.index, dtype)
    cached = _DCT_ORTHO_FACTOR_CACHE.get(key)
    if cached is not None:
        return cached
    frequency = torch.arange(length, device=device, dtype=dtype)
    angle = -torch.pi * frequency / float(2 * length)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    scale = torch.full_like(frequency, (2.0 / float(length)) ** 0.5)
    scale[0] = (1.0 / float(length)) ** 0.5
    result = (cosine, sine, scale)
    _DCT_ORTHO_FACTOR_CACHE[key] = result
    return result


def _dct_ii_ortho_1d(value: Tensor, dim: int) -> Tensor:
    """Apply an orthonormal DCT-II along ``dim`` using one real FFT."""

    moved = value.movedim(dim, -1)
    length = moved.shape[-1]
    if length <= 0:
        raise ValueError("DCT dimensions must be non-empty")
    extended = torch.cat((moved, moved.flip(-1)), dim=-1)
    spectrum = torch.fft.rfft(extended, n=2 * length, dim=-1)[..., :length]
    cosine, sine, scale = _dct_ortho_factors(
        length, device=value.device, dtype=value.dtype
    )
    cosine_sum = 0.5 * (
        spectrum.real * cosine - spectrum.imag * sine
    )
    return (cosine_sum * scale).movedim(-1, dim)


def _idct_iii_ortho_1d(value: Tensor, dim: int) -> Tensor:
    """Invert :func:`_dct_ii_ortho_1d` along ``dim``."""

    moved = value.movedim(dim, -1)
    length = moved.shape[-1]
    if length <= 0:
        raise ValueError("DCT dimensions must be non-empty")
    cosine, forward_sine, scale = _dct_ortho_factors(
        length, device=value.device, dtype=value.dtype
    )
    cosine_sum = moved / scale
    positive_spectrum = torch.complex(
        2.0 * cosine_sum * cosine,
        -2.0 * cosine_sum * forward_sine,
    )
    nyquist = torch.zeros_like(positive_spectrum[..., :1])
    spectrum = torch.cat((positive_spectrum, nyquist), dim=-1)
    extended = torch.fft.irfft(spectrum, n=2 * length, dim=-1)
    return extended[..., :length].movedim(-1, dim)


def _dct_ii_ortho_2d(value: Tensor) -> Tensor:
    """Apply a separable orthonormal DCT-II over the final two dimensions."""

    spectral_dtype = (
        torch.float32
        if value.dtype in {torch.float16, torch.bfloat16}
        else value.dtype
    )
    working = value.to(dtype=spectral_dtype)
    return _dct_ii_ortho_1d(_dct_ii_ortho_1d(working, -1), -2)


def _idct_iii_ortho_2d(value: Tensor, *, output_dtype: torch.dtype) -> Tensor:
    """Invert :func:`_dct_ii_ortho_2d` over the final two dimensions."""

    restored = _idct_iii_ortho_1d(
        _idct_iii_ortho_1d(value, -2),
        -1,
    )
    return restored.to(dtype=output_dtype)


def _green_resolvent_bands(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return nine positive Neumann-Laplacian resolvent bands.

    For ``A=-Delta_N`` and ``R_tau=(I+tau*A)^-1`` the bands are
    ``I-R_1``, seven adjacent resolvent differences, and ``R_8``.  They
    therefore form an exact non-negative partition of unity in the DCT-II
    spectrum.  The cache is runtime-only and creates no state-dict entry.
    """

    key = (height, width, device.type, device.index, dtype)
    cached = _GREEN_RESOLVENT_BAND_CACHE.get(key)
    if cached is not None:
        return cached
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    eigen_y = 4.0 * torch.sin(torch.pi * y / float(2 * height)).square()
    eigen_x = 4.0 * torch.sin(torch.pi * x / float(2 * width)).square()
    eigenvalue = eigen_y[:, None] + eigen_x[None, :]
    tau = torch.tensor(_GREEN_RESOLVENT_TAUS, device=device, dtype=dtype)
    resolvent = 1.0 / (1.0 + tau[:, None, None] * eigenvalue[None])
    bands = torch.cat(
        (
            1.0 - resolvent[:1],
            resolvent[:-1] - resolvent[1:],
            resolvent[-1:],
        ),
        dim=0,
    )
    _GREEN_RESOLVENT_BAND_CACHE[key] = bands
    return bands


def _parseval_resolvent_directional_bands(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return a nine-slot Parseval directional Neumann spectral frame.

    The existing nine positive resolvent bands are first grouped into one
    low-pass and four detail scales.  Pointwise square normalization makes the
    five radial windows a partition of squared energy.  Each detail window is
    then split by the two non-negative Neumann frequency-direction factors

        sqrt(lambda_x / (lambda_x + lambda_y)),
        sqrt(lambda_y / (lambda_x + lambda_y)).

    Hence the returned multipliers ``b_s`` satisfy ``sum_s b_s**2 == 1`` up
    to floating roundoff at every frequency.  They are even DCT multipliers,
    not odd Riesz transforms; this boundary-compatible construction keeps the
    existing one-DCT/one-IDCT fast path while exposing four scales in two
    spatial directions.  It is a fixed frame, not data whitening.
    """

    key = (height, width, device.type, device.index, dtype)
    cached = _PARSEVAL_RESOLVENT_DIRECTIONAL_BAND_CACHE.get(key)
    if cached is not None:
        return cached
    green = _green_resolvent_bands(
        height,
        width,
        device=device,
        dtype=dtype,
    )
    radial_raw = torch.stack(
        (
            green[8],
            torch.sqrt(green[6].square() + green[7].square()),
            torch.sqrt(green[4].square() + green[5].square()),
            torch.sqrt(green[2].square() + green[3].square()),
            torch.sqrt(green[0].square() + green[1].square()),
        ),
        dim=0,
    )
    radial_norm = torch.sqrt(
        radial_raw.square().sum(dim=0, keepdim=True)
    )
    radial = radial_raw / radial_norm.clamp_min(torch.finfo(dtype).tiny)

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    eigen_y = 4.0 * torch.sin(torch.pi * y / float(2 * height)).square()
    eigen_x = 4.0 * torch.sin(torch.pi * x / float(2 * width)).square()
    eigenvalue = eigen_y[:, None] + eigen_x[None, :]
    non_dc = eigenvalue > 0
    inverse = torch.where(
        non_dc,
        eigenvalue.reciprocal(),
        torch.zeros_like(eigenvalue),
    )
    direction_y = torch.sqrt(eigen_y[:, None] * inverse)
    direction_x = torch.sqrt(eigen_x[None, :] * inverse)
    directional = torch.stack(
        (
            radial[1:] * direction_x[None],
            radial[1:] * direction_y[None],
        ),
        dim=1,
    ).reshape(8, height, width)
    bands = torch.cat((radial[:1], directional), dim=0)
    _PARSEVAL_RESOLVENT_DIRECTIONAL_BAND_CACHE[key] = bands
    return bands


def _dct_hybrid_shells(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Five fixed low-frequency DCT shells used by the DH-9 compiler."""

    if height < 3 or width < 3:
        raise ValueError("dct_hybrid requires both spatial dimensions to be >= 3")
    key = (height, width, device.type, device.index, dtype)
    cached = _DCT_HYBRID_SHELL_CACHE.get(key)
    if cached is not None:
        return cached
    shells = torch.zeros(5, height, width, device=device, dtype=dtype)
    shells[0, 0, 0] = 1.0
    shells[1, 1, 0] = 1.0
    shells[1, 0, 1] = 1.0
    shells[2, 1, 1] = 1.0
    shells[3, 2, 0] = 1.0
    shells[3, 0, 2] = 1.0
    shells[4, 2, 1] = 1.0
    shells[4, 1, 2] = 1.0
    shells[4, 2, 2] = 1.0
    _DCT_HYBRID_SHELL_CACHE[key] = shells
    return shells


def _dct_global9_coordinate_masks(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Nine individual low-frequency DCT-II coordinate atoms.

    This is the exact global-only control dictionary for the canonical D arm.
    The nine coordinates use the row-major 3x3 chart, so the registered
    center-one q9 initialization stays on the DC coordinate while every other
    compiler action is exposed as one fixed global spectral atom.  There are
    no local spatial atoms or learned routing parameters in this dictionary.
    """

    if height < 3 or width < 3:
        raise ValueError(
            "global_dct9_only requires both spatial dimensions to be >= 3"
        )
    key = (height, width, device.type, device.index, dtype)
    cached = _DCT_GLOBAL9_COORDINATE_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    masks = torch.zeros(9, height, width, device=device, dtype=dtype)
    coordinates = (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    )
    for slot, (row, column) in enumerate(coordinates):
        masks[slot, row, column] = 1.0
    _DCT_GLOBAL9_COORDINATE_MASK_CACHE[key] = masks
    return masks


def _dct_global9_input_conditioning(
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    coordinate_masks: Tensor,
) -> Tensor:
    """Return bounded live-f gains for the nine global DCT coordinates."""

    if spectrum_real.shape != spectrum_imag.shape:
        raise ValueError("global DCT-9 real and imaginary spectra must match")
    if spectrum_real.ndim != 4:
        raise ValueError("global DCT-9 spectra expect [B,D,H,W]")
    if coordinate_masks.shape != (
        9,
        spectrum_real.shape[-2],
        spectrum_real.shape[-1],
    ):
        raise ValueError("global DCT-9 masks do not match the spectral grid")
    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    coordinate_mass = coordinate_masks.square().sum(dim=(-2, -1)).clamp_min(1.0)
    scores = (
        spectral_energy[:, :, None]
        * coordinate_masks.square()[None, None]
    ).sum(dim=(-2, -1)) / coordinate_mass[None, None]
    epsilon = torch.finfo(scores.dtype).eps
    probabilities = (scores + epsilon) / (
        scores + epsilon
    ).sum(dim=2, keepdim=True)
    return 0.5 + 4.5 * probabilities


def _dct_hybrid_local_frame(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Four fixed corner-supported even/odd tensor-product frame atoms.

    On an infinite or periodic grid their squared Fourier magnitudes sum to
    one.  Production convolution keeps the existing zero-padding boundary, so
    no Parseval claim is made at the image edge.
    """

    key = (device.type, device.index, dtype)
    cached = _DCT_HYBRID_LOCAL_FRAME_CACHE.get(key)
    if cached is not None:
        return cached
    plus = torch.tensor((1.0, 1.0), device=device, dtype=dtype)
    minus = torch.tensor((1.0, -1.0), device=device, dtype=dtype)
    signs = (plus, minus)
    frame = torch.zeros(4, 3, 3, device=device, dtype=dtype)
    slot = 0
    for vertical in signs:
        for horizontal in signs:
            frame[slot, 0, 0] = 0.25 * vertical[0] * horizontal[0]
            frame[slot, 0, 2] = 0.25 * vertical[0] * horizontal[1]
            frame[slot, 2, 0] = 0.25 * vertical[1] * horizontal[0]
            frame[slot, 2, 2] = 0.25 * vertical[1] * horizontal[1]
            slot += 1
    _DCT_HYBRID_LOCAL_FRAME_CACHE[key] = frame
    return frame


def _dct_hybrid_local_actions(paired_mixed: Tensor) -> Tensor:
    """Apply the four fixed corner atoms without materializing conv kernels."""

    if paired_mixed.ndim != 5 or paired_mixed.shape[2] != 2:
        raise ValueError("DH-9 local actions expect [B,D,2,H,W]")
    height, width = paired_mixed.shape[-2:]
    padded = F.pad(paired_mixed, (1, 1, 1, 1))
    top_left = padded[..., :height, :width]
    top_right = padded[..., :height, 2 : width + 2]
    bottom_left = padded[..., 2 : height + 2, :width]
    bottom_right = padded[..., 2 : height + 2, 2 : width + 2]
    return 0.25 * torch.stack(
        (
            top_left + top_right + bottom_left + bottom_right,
            top_left - top_right + bottom_left - bottom_right,
            top_left + top_right - bottom_left - bottom_right,
            top_left - top_right - bottom_left + bottom_right,
        ),
        dim=2,
    )


def _dct_hybrid_input_conditioning(
    paired_mixed: Tensor,
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    shells: Tensor,
    frame: Tensor,
    local_actions: Tensor | None = None,
) -> Tensor:
    """Return parameter-free ``[sample,state,slot]`` DH-9 routing gains.

    Each score is the RMS-normalized action energy of one fixed dictionary
    atom.  Adding dtype epsilon before normalization gives uniform routing for
    the zero input.  The affine map keeps every gain in ``[0.5, 5]`` and gives
    the nine gains exact mean one, matching the Green compiler's bounded
    forward-conditioning convention without whitening gradients or adding a
    trainable router.
    """

    if paired_mixed.ndim != 5 or paired_mixed.shape[2] != 2:
        raise ValueError("DH-9 conditioning expects [B,D,2,H,W]")
    if spectrum_real.shape != spectrum_imag.shape:
        raise ValueError("DH-9 real and imaginary spectra must match")
    if spectrum_real.shape != (
        paired_mixed.shape[0],
        paired_mixed.shape[1],
        paired_mixed.shape[3],
        paired_mixed.shape[4],
    ):
        raise ValueError("DH-9 spectra must match the paired latent grid")
    if shells.shape != (5, paired_mixed.shape[3], paired_mixed.shape[4]):
        raise ValueError("DH-9 conditioning requires five spectral shells")
    if frame.shape != (4, 3, 3):
        raise ValueError("DH-9 conditioning requires four local atoms")

    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    shell_mass = shells.square().sum(dim=(-2, -1)).clamp_min(1.0)
    shell_scores = (
        spectral_energy[:, :, None]
        * shells.square()[None, None]
    ).sum(dim=(-2, -1)) / shell_mass[None, None]

    # Compute the four fixed corner-atom responses directly.  This has the
    # exact zero-padding semantics of the production local convolution while
    # avoiding four additional convolution calls merely to obtain routing
    # statistics.
    if local_actions is None:
        local_actions = _dct_hybrid_local_actions(paired_mixed)
    if local_actions.shape != (
        paired_mixed.shape[0],
        paired_mixed.shape[1],
        4,
        2,
        paired_mixed.shape[3],
        paired_mixed.shape[4],
    ):
        raise ValueError("DH-9 local actions do not match the paired latent grid")
    frame_mass = frame.square().sum(dim=(-2, -1)).clamp_min(
        torch.finfo(paired_mixed.dtype).eps
    )
    local_scores = local_actions.square().sum(dim=3).mean(
        dim=(-2, -1)
    ) / frame_mass[None, None]

    scores = torch.cat((shell_scores, local_scores), dim=2)
    epsilon = torch.finfo(scores.dtype).eps
    probabilities = (scores + epsilon) / (
        scores + epsilon
    ).sum(dim=2, keepdim=True)
    return 0.5 + 4.5 * probabilities


def _dct_hybrid_input_conditioning_no_local(
    paired_mixed: Tensor,
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    shells: Tensor,
    frame: Tensor,
) -> Tensor:
    """GLRC9 routing scores for the static no-local endpoint.

    This routine deliberately does not call ``_dct_hybrid_local_actions`` and
    does not materialize a ``[B,D,4,2,H,W]`` response tensor.  It computes the
    four energy numerators directly from zero-padded corner combinations so
    the g=0 graph can be inspected as a genuinely no-local execution path.
    """

    if paired_mixed.ndim != 5 or paired_mixed.shape[2] != 2:
        raise ValueError("DH-9 no-local conditioning expects [B,D,2,H,W]")
    if spectrum_real.shape != spectrum_imag.shape:
        raise ValueError("DH-9 no-local spectra must match")
    height, width = paired_mixed.shape[-2:]
    if spectrum_real.shape != (paired_mixed.shape[0], paired_mixed.shape[1], height, width):
        raise ValueError("DH-9 no-local spectra must match the paired latent grid")
    if shells.shape != (5, height, width) or frame.shape != (4, 3, 3):
        raise ValueError("DH-9 no-local conditioning frame mismatch")

    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    shell_mass = shells.square().sum(dim=(-2, -1)).clamp_min(1.0)
    shell_scores = (
        spectral_energy[:, :, None] * shells.square()[None, None]
    ).sum(dim=(-2, -1)) / shell_mass[None, None]

    padded = F.pad(paired_mixed, (1, 1, 1, 1))
    top_left = padded[..., :height, :width]
    top_right = padded[..., :height, 2 : width + 2]
    bottom_left = padded[..., 2 : height + 2, :width]
    bottom_right = padded[..., 2 : height + 2, 2 : width + 2]
    # Keep each response's energy as a scalar route statistic; no response
    # tensor is retained or added to the latent.
    local_scores = torch.stack(
        (
            top_left + top_right + bottom_left + bottom_right,
            top_left - top_right + bottom_left - bottom_right,
            top_left + top_right - bottom_left - bottom_right,
            top_left - top_right - bottom_left + bottom_right,
        ),
        dim=2,
    )
    local_scores = (0.25 * local_scores).square().sum(dim=3).mean(
        dim=(-2, -1)
    )
    frame_mass = frame.square().sum(dim=(-2, -1)).clamp_min(
        torch.finfo(paired_mixed.dtype).eps
    )
    local_scores = local_scores / frame_mass[None, None]
    scores = torch.cat((shell_scores, local_scores), dim=2)
    epsilon = torch.finfo(scores.dtype).eps
    probabilities = (scores + epsilon) / (
        scores + epsilon
    ).sum(dim=2, keepdim=True)
    return 0.5 + 4.5 * probabilities


def _dct_hybrid_hierarchical_input_conditioning(
    paired_mixed: Tensor,
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    shells: Tensor,
    frame: Tensor,
    local_actions: Tensor | None = None,
) -> Tensor:
    """Exact-family factorization of GLRC9's flat action-energy routing.

    The global and local family probabilities are computed first, followed by
    the within-family action probabilities.  Their product is algebraically
    the original flat normalization.  Keeping this as a separate GCHAF graph
    lets the historical GLRC9 production function remain untouched while the
    parent section is verified numerically at the real FP32 code path.
    """

    if paired_mixed.ndim != 5 or paired_mixed.shape[2] != 2:
        raise ValueError("GCHAF routing expects [B,D,2,H,W]")
    if spectrum_real.shape != spectrum_imag.shape:
        raise ValueError("GCHAF real and imaginary spectra must match")
    if spectrum_real.shape != (
        paired_mixed.shape[0],
        paired_mixed.shape[1],
        paired_mixed.shape[3],
        paired_mixed.shape[4],
    ):
        raise ValueError("GCHAF spectra must match the paired latent grid")
    if shells.shape != (5, paired_mixed.shape[3], paired_mixed.shape[4]):
        raise ValueError("GCHAF routing requires five spectral shells")
    if frame.shape != (4, 3, 3):
        raise ValueError("GCHAF routing requires four local atoms")

    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    shell_mass = shells.square().sum(dim=(-2, -1)).clamp_min(1.0)
    shell_scores = (
        spectral_energy[:, :, None]
        * shells.square()[None, None]
    ).sum(dim=(-2, -1)) / shell_mass[None, None]
    if local_actions is None:
        local_actions = _dct_hybrid_local_actions(paired_mixed)
    if local_actions.shape != (
        paired_mixed.shape[0],
        paired_mixed.shape[1],
        4,
        2,
        paired_mixed.shape[3],
        paired_mixed.shape[4],
    ):
        raise ValueError("GCHAF local actions do not match the paired latent grid")
    frame_mass = frame.square().sum(dim=(-2, -1)).clamp_min(
        torch.finfo(paired_mixed.dtype).eps
    )
    local_scores = local_actions.square().sum(dim=3).mean(
        dim=(-2, -1)
    ) / frame_mass[None, None]

    epsilon = torch.finfo(shell_scores.dtype).eps
    shell_energy = shell_scores + epsilon
    local_energy = local_scores + epsilon
    shell_total = shell_energy.sum(dim=2, keepdim=True)
    local_total = local_energy.sum(dim=2, keepdim=True)
    family_total = shell_total + local_total
    global_family_probability = shell_total / family_total
    local_family_probability = local_total / family_total
    shell_probability = global_family_probability * (
        shell_energy / shell_total
    )
    local_probability = local_family_probability * (
        local_energy / local_total
    )
    probabilities = torch.cat((shell_probability, local_probability), dim=2)
    return 0.5 + 4.5 * probabilities


_GCHAF_CCAR9_HELMERT_CACHE: dict[
    tuple[str, int | None, torch.dtype], Tensor
] = {}


def gchaf_ccar9_helmert(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return the fixed orthonormal 8x9 Helmert contrast frame.

    Rows span the simplex tangent space: ``H @ 1 = 0`` and ``H @ H.T = I``.
    The frame is a fixed coordinate transform and is neither trainable nor
    stored in checkpoints.
    """

    if not dtype.is_floating_point:
        raise TypeError("CCAR9 Helmert frame requires a floating dtype")
    key = (device.type, device.index, dtype)
    cached = _GCHAF_CCAR9_HELMERT_CACHE.get(key)
    if cached is not None:
        return cached
    frame = torch.zeros(8, 9, device=device, dtype=dtype)
    for row in range(8):
        active = row + 1
        scale = 1.0 / math.sqrt(float(active * (active + 1)))
        frame[row, :active] = scale
        frame[row, active] = -float(active) * scale
    _GCHAF_CCAR9_HELMERT_CACHE[key] = frame
    return frame


def gchaf_ccar9_content_code(conditioning: Tensor) -> Tensor:
    """Decode GLRC9 routing into the bounded simplex-tangent content code.

    ``conditioning`` is the existing GLRC9 routing ``0.5 + 4.5 p``.  We decode
    ``p``, clamp and renormalize it for numerical safety, take centered log
    probabilities, then use the fixed Helmert frame:

    ``c = tanh(H8 @ (log(p) - mean(log(p)))) / sqrt(8)``.
    """

    if conditioning.ndim != 3 or conditioning.shape[2] != 9:
        raise ValueError("CCAR9 conditioning must have shape [B,D,9]")
    if not conditioning.is_floating_point():
        raise TypeError("CCAR9 conditioning must use a floating dtype")
    epsilon = torch.finfo(conditioning.dtype).eps
    probability = ((conditioning - 0.5) / 4.5).clamp(
        min=epsilon,
        max=1.0,
    )
    probability = probability / probability.sum(dim=2, keepdim=True).clamp_min(
        epsilon
    )
    log_probability = torch.log(probability)
    centered_log_probability = log_probability - log_probability.mean(
        dim=2, keepdim=True
    )
    helmert = gchaf_ccar9_helmert(
        device=conditioning.device,
        dtype=conditioning.dtype,
    )
    return torch.tanh(
        torch.einsum("rs,bds->bdr", helmert, centered_log_probability)
    ) / math.sqrt(8.0)


def _apply_ccar9_content_batch_permutation(content_code: Tensor) -> Tensor:
    permutation = _CCAR9_CONTENT_BATCH_PERMUTATION_CONTEXT
    if permutation is None:
        return content_code
    if content_code.ndim != 3 or content_code.shape[2] != 8:
        raise ValueError("CCAR9 content code must have shape [B,D,8]")
    if len(permutation) != content_code.shape[0]:
        raise ValueError(
            "CCAR9 diagnostic content permutation length does not match batch"
        )
    index = torch.tensor(
        permutation,
        device=content_code.device,
        dtype=torch.long,
    )
    return content_code.index_select(0, index)


def gchaf_ccar9_effective_action_weight(
    content_code: Tensor,
    content_action_weight: Tensor,
) -> Tensor:
    """Materialize diagnostic ``W_eff``; the production graph never calls it."""

    if content_code.ndim != 3 or content_code.shape[2] != 8:
        raise ValueError("CCAR9 content code must have shape [B,D,8]")
    if content_action_weight.ndim != 5:
        raise ValueError(
            "CCAR9 content-action weight must have shape [D,2,9,8,2]"
        )
    d = content_code.shape[1]
    if content_action_weight.shape != (d, 2, 9, 8, 2):
        raise ValueError(
            "CCAR9 content-action weight must have shape [D,2,9,8,2]"
        )
    return torch.einsum(
        "doari,bdr->bdoai",
        content_action_weight,
        content_code,
    )


def _dh_lms4_additive_bands(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Four disjoint broad bands added on top of the unchanged DH dictionary.

    The first band fills the ultra-low-frequency coordinates not already used
    by DH's five sparse shells.  The next two bands cover low-mid and mid
    frequencies; the final band is one combined high-frequency safety band.
    Together with the sparse-shell support they form an exact partition of the
    DCT-II grid.  The four original local shift actions remain untouched.
    """

    if height < 3 or width < 3:
        raise ValueError("DH-LMS4-A requires both spatial dimensions to be >= 3")
    key = (height, width, device.type, device.index, dtype)
    cached = _DH_LMS4_ADDITIVE_BAND_CACHE.get(key)
    if cached is not None:
        return cached
    y = torch.arange(height, device=device, dtype=dtype) / float(height)
    x = torch.arange(width, device=device, dtype=dtype) / float(width)
    radius = torch.sqrt(y[:, None].square() + x[None, :].square())
    sparse_support = _dct_hybrid_shells(
        height,
        width,
        device=device,
        dtype=dtype,
    ).sum(dim=0) > 0.0
    bands = torch.stack(
        (
            (radius <= (1.0 / 16.0)) & ~sparse_support,
            (radius > (1.0 / 16.0))
            & (radius <= (1.0 / 8.0))
            & ~sparse_support,
            (radius > (1.0 / 8.0))
            & (radius <= (1.0 / 4.0))
            & ~sparse_support,
            (radius > (1.0 / 4.0)) & ~sparse_support,
        ),
        dim=0,
    ).to(dtype=dtype)
    coverage = bands.sum(dim=0) + sparse_support.to(dtype=dtype)
    if not bool(torch.equal(coverage, torch.ones_like(coverage))):
        raise RuntimeError("DH-LMS4-A bands do not partition the DCT grid")
    _DH_LMS4_ADDITIVE_BAND_CACHE[key] = bands
    return bands


def _dh_lms4_additive_input_conditioning(
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    bands: Tensor,
) -> Tensor:
    """Return bounded sample-by-state gains for the four additive bands."""

    if spectrum_real.shape != spectrum_imag.shape or spectrum_real.ndim != 4:
        raise ValueError("DH-LMS4-A expects matching [B,D,H,W] spectra")
    if bands.shape != (4, spectrum_real.shape[-2], spectrum_real.shape[-1]):
        raise ValueError("DH-LMS4-A requires four broad spectral bands")
    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    mass = bands.sum(dim=(-2, -1)).clamp_min(1.0)
    scores = torch.einsum("bdhw,shw->bds", spectral_energy, bands) / mass[None, None]
    epsilon = torch.finfo(scores.dtype).eps
    probabilities = (scores + epsilon) / (
        scores + epsilon
    ).sum(dim=2, keepdim=True)
    # For four slots, 0.5 + 2 p has exact slot mean one and lies in [0.5, 2.5].
    return 0.5 + 2.0 * probabilities


def _dh_lms4_act4_bands(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Return norm-matched angular companions to the four LMS4 bands.

    The normalized vertical-minus-horizontal frequency coordinate is centered
    inside every parent band.  Hence each companion is orthogonal to its
    radial parent on rectangular as well as square grids.  Non-degenerate
    companions are rescaled to the parent Frobenius norm.
    """

    key = (height, width, device.type, device.index, dtype)
    cached = _DH_LMS4_ACT4_BAND_CACHE.get(key)
    if cached is not None:
        return cached
    bands = _dh_lms4_additive_bands(
        height,
        width,
        device=device,
        dtype=dtype,
    )
    y = torch.arange(height, device=device, dtype=dtype) / float(max(height - 1, 1))
    x = torch.arange(width, device=device, dtype=dtype) / float(max(width - 1, 1))
    angular_coordinate = y[:, None] - x[None, :]
    mass = bands.sum(dim=(-2, -1)).clamp_min(1.0)
    band_mean = torch.einsum("shw,hw->s", bands, angular_coordinate) / mass
    centered = bands * (angular_coordinate[None] - band_mean[:, None, None])
    parent_norm = bands.square().sum(dim=(-2, -1)).sqrt()
    centered_norm = centered.square().sum(dim=(-2, -1)).sqrt()
    epsilon = torch.finfo(dtype).eps
    scale = torch.where(
        centered_norm > epsilon,
        parent_norm / centered_norm.clamp_min(epsilon),
        torch.zeros_like(centered_norm),
    )
    companions = centered * scale[:, None, None]
    _DH_LMS4_ACT4_BAND_CACHE[key] = companions
    return companions


def _dh_lms4_act4_input_conditioning(
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    companions: Tensor,
) -> Tensor:
    """Return bounded scale-invariant angular-energy coordinates.

    The denominator uses the same energy weighted by the absolute companion,
    so the triangle inequality bounds every coordinate in [-1, 1].
    """

    if spectrum_real.shape != spectrum_imag.shape or spectrum_real.ndim != 4:
        raise ValueError("DH-LMS4-ACT4 expects matching [B,D,H,W] spectra")
    if companions.shape != (4, spectrum_real.shape[-2], spectrum_real.shape[-1]):
        raise ValueError("DH-LMS4-ACT4 requires four angular companions")
    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    numerator = torch.einsum("bdhw,shw->bds", spectral_energy, companions)
    denominator = torch.einsum(
        "bdhw,shw->bds", spectral_energy, companions.abs()
    )
    epsilon = torch.finfo(spectral_energy.dtype).eps
    return numerator / denominator.clamp_min(epsilon)


def _dh_lms4_dual_route_conditioning(
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    bands: Tensor,
    companions: Tensor,
) -> Tensor:
    """Return one bounded complex source code for every LMS4 band.

    The real coordinate is centered log band energy; the imaginary coordinate
    is within-band angular-energy asymmetry.  A radial ``tanh`` maps the pair
    into the open complex unit disk while retaining unit derivative at zero.
    Both source coordinates are invariant to positive global feature scaling.
    """

    radial = _dh_lms4_signed_input_conditioning(
        spectrum_real,
        spectrum_imag,
        bands,
    )
    angular = _dh_lms4_act4_input_conditioning(
        spectrum_real,
        spectrum_imag,
        companions,
    )
    magnitude = torch.sqrt(radial.square() + angular.square())
    epsilon = torch.finfo(magnitude.dtype).eps
    scale = torch.where(
        magnitude > epsilon,
        torch.tanh(magnitude) / magnitude.clamp_min(epsilon),
        torch.ones_like(magnitude),
    )
    return torch.stack((radial * scale, angular * scale), dim=-1)


def _dh_lms4_signed_input_conditioning(
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    bands: Tensor,
) -> Tensor:
    """Return centered-log-energy source conditioning for signed LMS4.

    The fixed-temperature ``tanh`` is applied after subtracting the mean log
    energy across the four bands. Consequently a positive rescaling of the
    source spectrum cancels before the nonlinearity up to floating-point
    roundoff, while the four routing coordinates retain both signs.
    """

    if spectrum_real.shape != spectrum_imag.shape or spectrum_real.ndim != 4:
        raise ValueError("signed DH-LMS4 expects matching [B,D,H,W] spectra")
    if bands.shape != (4, spectrum_real.shape[-2], spectrum_real.shape[-1]):
        raise ValueError("signed DH-LMS4 requires four broad spectral bands")
    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    mass = bands.sum(dim=(-2, -1)).clamp_min(1.0)
    energies = torch.einsum("bdhw,shw->bds", spectral_energy, bands) / mass[None, None]
    epsilon = torch.finfo(energies.dtype).eps
    log_energy = torch.log(energies + epsilon)
    centered_log_energy = log_energy - log_energy.mean(dim=2, keepdim=True)
    return torch.tanh(centered_log_energy)


DH_MWF9_ETA = 0.125
DH_MWF9_SUBBAND_PERMUTATION: tuple[int, ...] = tuple(range(9))

DH_SWF9_ETA = 0.125
DH_SWF9_SCALES: tuple[int, ...] = (1, 2, 4)
DH_SWF9_BANDS: tuple[str, ...] = ("LH", "HL", "HH")


def _dh_swf9_zero_shift(value: Tensor, *, dy: int = 0, dx: int = 0) -> Tensor:
    """Apply a finite zero-extension shift without periodic wrap-around."""

    if value.ndim < 2:
        raise ValueError("DH-SWF9 shifts require at least two spatial dimensions")
    height, width = value.shape[-2:]
    padded = F.pad(
        value,
        (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)),
    )
    return padded[
        ...,
        max(-dy, 0) : max(-dy, 0) + height,
        max(-dx, 0) : max(-dx, 0) + width,
    ]


def _dh_swf9_centered_pair(
    value: Tensor,
    distance: int,
    *,
    axis: str,
    transpose: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return the zero-boundary ``(L,H)`` pair or its discrete transpose."""

    if distance <= 0:
        raise ValueError("DH-SWF9 shift distance must be positive")
    if axis not in {"x", "y"}:
        raise ValueError("DH-SWF9 axis must be 'x' or 'y'")
    if axis == "x":
        positive = (0, distance)
        negative = (0, -distance)
    else:
        positive = (distance, 0)
        negative = (-distance, 0)
    if transpose:
        positive, negative = negative, positive
    positive_value = _dh_swf9_zero_shift(
        value, dy=positive[0], dx=positive[1]
    )
    negative_value = _dh_swf9_zero_shift(
        value, dy=negative[0], dx=negative[1]
    )
    return (positive_value + negative_value) * 0.5, (
        positive_value - negative_value
    ) * 0.5


def _dh_swf9_apply_band(
    value: Tensor,
    distance: int,
    band: str,
    *,
    transpose: bool = False,
) -> Tensor:
    """Apply one SWF9 band or its exact finite-grid discrete adjoint."""

    if band not in DH_SWF9_BANDS:
        raise ValueError(f"unknown DH-SWF9 band {band!r}")
    if not transpose:
        lx, hx = _dh_swf9_centered_pair(value, distance, axis="x")
        if band == "LH":
            return _dh_swf9_centered_pair(hx, distance, axis="y")[0]
        if band == "HL":
            return _dh_swf9_centered_pair(lx, distance, axis="y")[1]
        return _dh_swf9_centered_pair(hx, distance, axis="y")[1]

    ly_t, hy_t = _dh_swf9_centered_pair(
        value, distance, axis="y", transpose=True
    )
    if band == "LH":
        return _dh_swf9_centered_pair(ly_t, distance, axis="x", transpose=True)[1]
    if band == "HL":
        return _dh_swf9_centered_pair(hy_t, distance, axis="x", transpose=True)[0]
    return _dh_swf9_centered_pair(hy_t, distance, axis="x", transpose=True)[1]


def _dh_swf9_wavelet_bands(paired_u: Tensor) -> Tensor:
    """Return scale-major ``[B,D,9,2,H,W]`` zero-boundary SWF9 bands."""

    if paired_u.ndim != 5 or paired_u.shape[2] != 2:
        raise ValueError("DH-SWF9 expects paired latent shape [B,D,2,H,W]")
    return torch.stack(
        [
            _dh_swf9_apply_band(paired_u, distance, band)
            for distance in DH_SWF9_SCALES
            for band in DH_SWF9_BANDS
        ],
        dim=2,
    )


def _dh_swf9_complex_scale(value: Tensor, real: Tensor, imag: Tensor) -> Tensor:
    """Multiply paired complex channels by one state-wise complex scalar."""

    if value.ndim != 5 or value.shape[2] != 2:
        raise ValueError("DH-SWF9 complex scaling expects [B,D,2,H,W]")
    if real.shape != imag.shape or real.ndim != 1 or real.shape[0] != value.shape[1]:
        raise ValueError("DH-SWF9 complex scale must have shape [D]")
    value_real, value_imag = value.unbind(dim=2)
    real = real[None, :, None, None]
    imag = imag[None, :, None, None]
    return torch.stack(
        (
            real * value_real - imag * value_imag,
            imag * value_real + real * value_imag,
        ),
        dim=2,
    )


def _dh_swf9_wavelet_residual(
    paired_u: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """Return ``(eta/3) * sum K* [gamma K u]`` on a zero-boundary grid."""

    if paired_u.ndim != 5 or paired_u.shape[2] != 2:
        raise ValueError("DH-SWF9 expects paired latent shape [B,D,2,H,W]")
    expected = (paired_u.shape[1], 9)
    if gamma_real.shape != expected or gamma_imag.shape != expected:
        raise ValueError("DH-SWF9 gamma tensors must have shape [D,9]")
    bands = _dh_swf9_wavelet_bands(paired_u)
    residual_terms = []
    slot = 0
    for distance in DH_SWF9_SCALES:
        for band in DH_SWF9_BANDS:
            transformed = _dh_swf9_complex_scale(
                bands[:, :, slot], gamma_real[:, slot], gamma_imag[:, slot]
            )
            residual_terms.append(
                _dh_swf9_apply_band(transformed, distance, band, transpose=True)
            )
            slot += 1
    return (DH_SWF9_ETA / 3.0) * torch.stack(residual_terms, dim=0).sum(dim=0)


def _dh_mwf9_haar_analysis(value: Tensor) -> tuple[Tensor, Tensor]:
    """Apply one orthonormal 2-D Haar analysis level.

    The detail order is ``LH, HL, HH`` and follows the existing fixed Haar
    phase convention in ``operator_condition.py``.  No padding is performed:
    the three-level graft is admitted only on grids divisible by eight.
    """

    if value.ndim != 4 or value.shape[-2] % 2 or value.shape[-1] % 2:
        raise ValueError("DH-MWF9 Haar analysis requires an even [B,C,H,W]")
    batch, channels, height, width = value.shape
    phases = F.pixel_unshuffle(value, 2).view(
        batch, channels, 4, height // 2, width // 2
    )
    x00, x01, x10, x11 = phases.unbind(dim=2)
    low = (x00 + x01 + x10 + x11) * 0.5
    high = torch.cat(
        (
            (-x00 - x01 + x10 + x11) * 0.5,
            (-x00 + x01 - x10 + x11) * 0.5,
            (x00 - x01 - x10 + x11) * 0.5,
        ),
        dim=1,
    )
    return low, high


def _dh_mwf9_haar_synthesis(low: Tensor, high: Tensor) -> Tensor:
    """Invert one orthonormal 2-D Haar analysis level."""

    if low.ndim != 4 or high.ndim != 4:
        raise ValueError("DH-MWF9 Haar synthesis expects rank-4 tensors")
    batch, channels, height, width = low.shape
    if high.shape != (batch, 3 * channels, height, width):
        raise ValueError("DH-MWF9 Haar detail shape does not match its low band")
    h_lh, h_hl, h_hh = high.view(
        batch, 3, channels, height, width
    ).unbind(dim=1)
    x00 = (low - h_lh - h_hl + h_hh) * 0.5
    x01 = (low - h_lh + h_hl - h_hh) * 0.5
    x10 = (low + h_lh - h_hl - h_hh) * 0.5
    x11 = (low + h_lh + h_hl + h_hh) * 0.5
    phases = torch.stack((x00, x01, x10, x11), dim=2).reshape(
        batch, 4 * channels, height, width
    )
    return F.pixel_shuffle(phases, 2)


def _dh_mwf9_reorder_gains(
    gain: Tensor,
    permutation: tuple[int, ...] = DH_MWF9_SUBBAND_PERMUTATION,
) -> Tensor:
    """Apply a compile-time fixed detail-band permutation, if requested."""

    if gain.ndim != 2 or gain.shape[1] != 9:
        raise ValueError("DH-MWF9 gains must have shape [D,9]")
    if tuple(sorted(permutation)) != tuple(range(9)):
        raise ValueError("DH-MWF9 subband permutation must be a permutation of 0..8")
    if tuple(permutation) == tuple(range(9)):
        return gain
    index = torch.tensor(permutation, device=gain.device, dtype=torch.long)
    return gain.index_select(1, index)


def _dh_mwf9_apply_detail_gain(
    high: Tensor,
    gain_real: Tensor,
    gain_imag: Tensor,
    start: int,
) -> Tensor:
    """Apply one scale's three shared complex gains to ``LH,HL,HH``."""

    batch, channels, height, width = high.shape
    states = gain_real.shape[0]
    if channels != 6 * states or gain_imag.shape != gain_real.shape:
        raise ValueError("DH-MWF9 detail channels and complex gains do not match")
    detail = high.reshape(batch, 3, states, 2, height, width)
    detail_real = detail[:, :, :, 0]
    detail_imag = detail[:, :, :, 1]
    scale_real = gain_real[:, start : start + 3].transpose(0, 1)
    scale_imag = gain_imag[:, start : start + 3].transpose(0, 1)
    scale_real = scale_real[None, :, :, None, None]
    scale_imag = scale_imag[None, :, :, None, None]
    output_real = scale_real * detail_real - scale_imag * detail_imag
    output_imag = scale_imag * detail_real + scale_real * detail_imag
    return torch.stack((output_real, output_imag), dim=3).reshape(
        batch, 6 * states, height, width
    )


def _dh_mwf9_wavelet_residual(
    paired_u: Tensor,
    gain_real: Tensor,
    gain_imag: Tensor,
) -> Tensor:
    """Return ``W*diag(delta_theta, LL=0)*W u`` for the fixed Haar frame.

    ``paired_u`` is interleaved ``[B,D,2,H,W]``.  The nine residual gains are
    consumed in scale-major order ``(LH,HL,HH)`` for scales 1, 2, and 3;
    ``DH_MWF9_SUBBAND_PERMUTATION`` is a fixed, zero-parameter control hook.
    The same complex gain is applied to the real and imaginary coefficient
    pair, and the final level's LL coefficient is intentionally zeroed.
    """

    if paired_u.ndim != 5 or paired_u.shape[2] != 2:
        raise ValueError("DH-MWF9 expects paired latent shape [B,D,2,H,W]")
    batch, states, _, height, width = paired_u.shape
    if height % 8 or width % 8:
        raise ValueError("DH-MWF9 requires latent height and width divisible by 8")
    if gain_real.shape != (states, 9) or gain_imag.shape != (states, 9):
        raise ValueError("DH-MWF9 residual gains must have shape [D,9]")

    ordered_real = _dh_mwf9_reorder_gains(gain_real)
    ordered_imag = _dh_mwf9_reorder_gains(gain_imag)
    current = paired_u.reshape(batch, 2 * states, height, width)
    details: list[Tensor] = []
    for _level in range(3):
        current, high = _dh_mwf9_haar_analysis(current)
        details.append(high)

    reconstructed = torch.zeros_like(current)
    for level in (2, 1, 0):
        high = _dh_mwf9_apply_detail_gain(
            details[level], ordered_real, ordered_imag, 3 * level
        )
        reconstructed = _dh_mwf9_haar_synthesis(reconstructed, high)
    return reconstructed.reshape_as(paired_u)


def _dct_hybrid_learned_simplex_conditioning(
    base_conditioning: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
    *,
    parent_fraction: float = 0.5,
) -> Tensor:
    """Bounded learned transport of DH-9's action-energy simplex.

    ``base_conditioning`` is DH-9's ``0.5 + 4.5 p`` routing, where ``p`` is
    strictly positive and sums to one.  Each block/state channel learns a
    bounded log-simplex residual, then mixes it with the parent distribution:

    ``p_eff = parent_fraction * p + (1-parent_fraction) * softmax(log p + d)``.

    Consequently ``p_eff`` remains a probability vector and every parent
    route keeps at least ``parent_fraction`` of its original mass.  This is a
    forward compiler parameterization, not optimizer/gradient preconditioning.
    """

    if base_conditioning.ndim != 3 or base_conditioning.shape[-1] != 9:
        raise ValueError("learned DH routing expects [B,D,9]")
    states = base_conditioning.shape[1]
    if router_weight.shape != (states, 9, 9):
        raise ValueError("learned DH router weight must have shape [D,9,9]")
    if router_bias.shape != (states, 9):
        raise ValueError("learned DH router bias must have shape [D,9]")
    if not 0.0 <= parent_fraction <= 1.0:
        raise ValueError("parent_fraction must lie in [0,1]")

    # In trained DH checkpoints an action can be so weak that the FP32
    # encoding ``0.5 + 4.5 * p`` rounds back to exactly 0.5.  Taking log of
    # the naively decoded zero would poison the complete K12 task path.  The
    # router owns a log-simplex chart, so keep its chart strictly interior and
    # renormalize before applying the learned transport.
    probability = ((base_conditioning - 0.5) / 4.5).clamp_min(
        torch.finfo(base_conditioning.dtype).eps
    )
    probability = probability / probability.sum(dim=2, keepdim=True)
    log_probability = probability.log()
    feature = torch.tanh(
        log_probability - log_probability.mean(dim=2, keepdim=True)
    )
    residual_logit = (
        torch.einsum(
            "dst,bdt->bds",
            torch.tanh(router_weight),
            feature,
        )
        / 9.0
        + torch.tanh(router_bias)[None]
    )
    learned_probability = torch.softmax(
        log_probability + residual_logit,
        dim=2,
    )
    effective_probability = (
        parent_fraction * probability
        + (1.0 - parent_fraction) * learned_probability
    )
    return 0.5 + 4.5 * effective_probability


def _polyphase_local_cycle_actions(paired_mixed: Tensor) -> Tensor:
    """Return the four local directed 2x2 phase-cycle actions.

    The phase order is ``00, 01, 10, 11``.  Each action copies one source
    phase into one target phase at the same coarse cell:

    ``01 <- 00``, ``11 <- 01``, ``10 <- 11``, and ``00 <- 10``.

    The returned tensor stays on the coarse phase grid.  Odd spatial sizes
    are handled by zero padding on the bottom/right; the caller performs one
    aggregate PixelShuffle after the four weighted actions are summed.  This
    is a fixed coordinate transform; it introduces no parameter or persistent
    buffer.
    """

    if paired_mixed.ndim != 5 or paired_mixed.shape[2] != 2:
        raise ValueError("polyphase cycle expects [B,D,2,H,W]")
    batch, d, _, height, width = paired_mixed.shape
    pad_height = height % 2
    pad_width = width % 2
    padded = F.pad(
        paired_mixed.reshape(batch * d * 2, 1, height, width),
        (0, pad_width, 0, pad_height),
    )
    padded_height, padded_width = padded.shape[-2:]
    phase = F.pixel_unshuffle(padded, 2).reshape(
        batch,
        d,
        2,
        4,
        padded_height // 2,
        padded_width // 2,
    )
    zero = torch.zeros_like(phase[..., :1, :, :])
    phase_00 = phase[..., 0:1, :, :]
    phase_01 = phase[..., 1:2, :, :]
    phase_10 = phase[..., 2:3, :, :]
    phase_11 = phase[..., 3:4, :, :]
    return torch.stack(
        (
            torch.cat((zero, phase_00, zero, zero), dim=3),
            torch.cat((zero, zero, zero, phase_01), dim=3),
            torch.cat((zero, zero, phase_11, zero), dim=3),
            torch.cat((phase_10, zero, zero, zero), dim=3),
        ),
        dim=3,
    )


def _dct_phase_cycle_input_conditioning(
    paired_mixed: Tensor,
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    shells: Tensor,
    phase_actions: Tensor,
) -> Tensor:
    """Return bounded sample/state routing for DHP-9's 5+4 dictionary."""

    if paired_mixed.ndim != 5 or paired_mixed.shape[2] != 2:
        raise ValueError("DHP-9 conditioning expects [B,D,2,H,W]")
    if spectrum_real.shape != spectrum_imag.shape:
        raise ValueError("DHP-9 real and imaginary spectra must match")
    if spectrum_real.shape != (
        paired_mixed.shape[0],
        paired_mixed.shape[1],
        paired_mixed.shape[3],
        paired_mixed.shape[4],
    ):
        raise ValueError("DHP-9 spectra must match the paired latent grid")
    if shells.shape != (5, paired_mixed.shape[3], paired_mixed.shape[4]):
        raise ValueError("DHP-9 conditioning requires five spectral shells")
    padded_height = paired_mixed.shape[3] + paired_mixed.shape[3] % 2
    padded_width = paired_mixed.shape[4] + paired_mixed.shape[4] % 2
    if phase_actions.shape != (
        paired_mixed.shape[0],
        paired_mixed.shape[1],
        2,
        4,
        4,
        padded_height // 2,
        padded_width // 2,
    ):
        raise ValueError("DHP-9 conditioning requires four phase actions")

    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    shell_mass = shells.square().sum(dim=(-2, -1)).clamp_min(1.0)
    shell_scores = (
        spectral_energy[:, :, None] * shells.square()[None, None]
    ).sum(dim=(-2, -1)) / shell_mass[None, None]

    height, width = paired_mixed.shape[-2:]
    target_counts = torch.tensor(
        (
            ((height + 1) // 2) * (width // 2),
            (height // 2) * (width // 2),
            (height // 2) * ((width + 1) // 2),
            ((height + 1) // 2) * ((width + 1) // 2),
        ),
        device=paired_mixed.device,
        dtype=paired_mixed.dtype,
    ).clamp_min(1.0)
    phase_scores = phase_actions.square().sum(dim=2).sum(dim=(-3, -2, -1))
    phase_scores = phase_scores / target_counts[None, None]
    scores = torch.cat((shell_scores, phase_scores), dim=2)
    epsilon = torch.finfo(scores.dtype).eps
    probabilities = (scores + epsilon) / (
        scores + epsilon
    ).sum(dim=2, keepdim=True)
    return 0.5 + 4.5 * probabilities


def _green_resolvent_conditioning(
    spectrum_real: Tensor,
    spectrum_imag: Tensor,
    bands: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute parameter-free, per-input Green-band routing coefficients.

    ``q_bds=0.5+4.5*p_bds`` with ``p`` the normalized band-action energy for
    every sample and state channel.  This is part of the forward operator
    rather than gradient whitening or an optimizer preconditioner: it changes
    the input-conditioned Jacobian while leaving both the parameter count and
    the zero-residual initial function unchanged.  Each channel's nine
    coefficients have mean one and remain in [0.5, 5].
    """

    if spectrum_real.shape != spectrum_imag.shape:
        raise ValueError("real and imaginary Green spectra must have one shape")
    if spectrum_real.ndim != 4 or bands.ndim != 3 or bands.shape[0] != 9:
        raise ValueError("Green conditioning expects [B,D,H,W] and [9,H,W]")
    if tuple(spectrum_real.shape[-2:]) != tuple(bands.shape[-2:]):
        raise ValueError("Green spectra and bands must use one spatial grid")
    spectral_energy = spectrum_real.square() + spectrum_imag.square()
    band_energy = torch.einsum(
        "bdhw,shw->bds", spectral_energy, bands.square()
    ) / float(spectrum_real.shape[2] * spectrum_real.shape[3])
    epsilon = torch.finfo(band_energy.dtype).eps
    probability = (band_energy + epsilon) / (
        band_energy.sum(dim=2, keepdim=True) + 9.0 * epsilon
    )
    conditioning = 0.5 + 4.5 * probability
    return conditioning, band_energy


def _fused_q9_green_resolvent_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """Parameter-matched, input-conditioned nine-band Green q9 compiler.

    The registered complex ``D x 1 x 3 x 3`` tensor supplies nine per-state
    residual gains.  Fixed resolvent bands of the Neumann Laplacian supply the
    operator dictionary, while band energies of the current live ``f`` route
    those gains separately for every sample.  No trainable parameter, q9
    coordinate, legalizer, F9 transition, recurrence step, or outer-network
    tensor is added or changed.
    """

    d = mix_m_real.shape[0]
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag), dim=1
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real), dim=1
    )
    pointwise_weight = torch.stack((real_rows, imag_rows), dim=1).reshape(
        2 * d, 2 * d, 1, 1
    )
    mixed = F.conv2d(packed_f, pointwise_weight)
    batch, _, height, width = mixed.shape
    paired = mixed.view(batch, d, 2, height, width)
    mixed_real = paired[:, :, 0]
    mixed_imag = paired[:, :, 1]

    # The DCT is real-linear, so real and imaginary fields can share one
    # batched transform without changing the mathematical operator.
    spectrum = _dct_ii_ortho_2d(
        torch.stack((mixed_real, mixed_imag), dim=2)
    )
    spectrum_real = spectrum[:, :, 0]
    spectrum_imag = spectrum[:, :, 1]
    bands = _green_resolvent_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )
    conditioning, _band_energy = _green_resolvent_conditioning(
        spectrum_real, spectrum_imag, bands
    )

    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    anchor = F.pad(
        torch.ones_like(gain_real[:, :1]),
        (0, 8),
    )
    delta_real = gain_real - anchor
    delta_imag = gain_imag
    multiplier_real = torch.einsum(
        "bds,shw->bdhw", conditioning * delta_real[None], bands
    )
    multiplier_imag = torch.einsum(
        "bds,shw->bdhw", conditioning * delta_imag[None], bands
    )
    residual_spectrum_real = (
        multiplier_real * spectrum_real - multiplier_imag * spectrum_imag
    )
    residual_spectrum_imag = (
        multiplier_imag * spectrum_real + multiplier_real * spectrum_imag
    )
    residual = _idct_iii_ortho_2d(
        torch.stack(
            (residual_spectrum_real, residual_spectrum_imag), dim=2
        ),
        output_dtype=mixed_real.dtype,
    )
    residual_real = residual[:, :, 0]
    residual_imag = residual[:, :, 1]
    paired = torch.stack(
        (mixed_real + residual_real, mixed_imag + residual_imag), dim=2
    )
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    interleaved = (paired * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_parseval_resolvent_directional_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """Parameter-matched input-conditioned Parseval directional compiler.

    The nine existing complex depthwise scalars are interpreted as gains on
    one low-pass and four-scale-by-two-direction fixed DCT multipliers.  Their
    squared responses partition unity at every Neumann frequency.  Live-f
    band energies provide the same bounded sample-by-channel-by-nine routing
    used by the Green candidate.  The default residual gains are zero, so the
    registered initialization retains the exact latent identity path.
    """

    d = mix_m_real.shape[0]
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag), dim=1
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real), dim=1
    )
    pointwise_weight = torch.stack((real_rows, imag_rows), dim=1).reshape(
        2 * d, 2 * d, 1, 1
    )
    mixed = F.conv2d(packed_f, pointwise_weight)
    batch, _, height, width = mixed.shape
    paired = mixed.view(batch, d, 2, height, width)
    mixed_real = paired[:, :, 0]
    mixed_imag = paired[:, :, 1]

    spectrum = _dct_ii_ortho_2d(
        torch.stack((mixed_real, mixed_imag), dim=2)
    )
    spectrum_real = spectrum[:, :, 0]
    spectrum_imag = spectrum[:, :, 1]
    bands = _parseval_resolvent_directional_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )
    conditioning, _band_energy = _green_resolvent_conditioning(
        spectrum_real, spectrum_imag, bands
    )

    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    anchor = F.pad(torch.ones_like(gain_real[:, :1]), (0, 8))
    delta_real = gain_real - anchor
    delta_imag = gain_imag
    multiplier_real = torch.einsum(
        "bds,shw->bdhw", conditioning * delta_real[None], bands
    )
    multiplier_imag = torch.einsum(
        "bds,shw->bdhw", conditioning * delta_imag[None], bands
    )
    residual_spectrum_real = (
        multiplier_real * spectrum_real - multiplier_imag * spectrum_imag
    )
    residual_spectrum_imag = (
        multiplier_imag * spectrum_real + multiplier_real * spectrum_imag
    )
    residual = _idct_iii_ortho_2d(
        torch.stack(
            (residual_spectrum_real, residual_spectrum_imag), dim=2
        ),
        output_dtype=mixed_real.dtype,
    )
    paired = torch.stack(
        (mixed_real + residual[:, :, 0], mixed_imag + residual[:, :, 1]),
        dim=2,
    )
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    interleaved = (paired * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _q9_widely_linear_pointwise(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
) -> Tensor:
    """Return the interleaved complex output of the shared pointwise G map."""

    d = mix_m_real.shape[0]
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag), dim=1
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real), dim=1
    )
    pointwise_weight = torch.stack((real_rows, imag_rows), dim=1).reshape(
        2 * d, 2 * d, 1, 1
    )
    return F.conv2d(packed_f, pointwise_weight)


_POLYPHASE_LIFTING_RING = (
    (0, 1),  # 00 -> 01
    (1, 3),  # 01 -> 11
    (3, 2),  # 11 -> 10
    (2, 0),  # 10 -> 00
)


def _polyphase_lifting_phase_view(
    paired_mixed: Tensor,
) -> tuple[Tensor, Tensor]:
    """Split a complex field into the four PixelUnshuffle phases.

    Phase index order is exactly ``00, 01, 10, 11``.  Odd grids are padded
    only on the right and bottom.  The returned validity mask records the
    original samples so later phase means cannot see those padding zeros.
    """

    if paired_mixed.ndim != 5 or paired_mixed.shape[2] != 2:
        raise ValueError("PL-9 expects paired latent shape [B,D,2,H,W]")
    batch, states, _, height, width = paired_mixed.shape
    if height <= 0 or width <= 0:
        raise ValueError("PL-9 requires positive spatial dimensions")
    pad_bottom = height % 2
    pad_right = width % 2
    padded_height = height + pad_bottom
    padded_width = width + pad_right
    packed = paired_mixed.reshape(batch, 2 * states, height, width)
    packed = F.pad(packed, (0, pad_right, 0, pad_bottom))
    phases = F.pixel_unshuffle(packed, 2).view(
        batch,
        states,
        2,
        4,
        padded_height // 2,
        padded_width // 2,
    )

    valid = paired_mixed.new_ones((1, 1, height, width))
    valid = F.pad(valid, (0, pad_right, 0, pad_bottom))
    valid_phases = F.pixel_unshuffle(valid, 2).view(
        1,
        1,
        1,
        4,
        padded_height // 2,
        padded_width // 2,
    )
    return phases, valid_phases


def _polyphase_lifting_valid_phase_means(
    phases: Tensor,
    valid_phases: Tensor,
) -> Tensor:
    """Return complex phase means over original, non-padding samples only."""

    if phases.ndim != 6 or phases.shape[2:4] != (2, 4):
        raise ValueError("PL-9 phases must have shape [B,D,2,4,H2,W2]")
    if valid_phases.shape != (
        1,
        1,
        1,
        4,
        phases.shape[-2],
        phases.shape[-1],
    ):
        raise ValueError("PL-9 validity mask does not match phase grid")
    counts = valid_phases.sum(dim=(-2, -1)).clamp_min(1.0)
    return (phases * valid_phases).sum(dim=(-2, -1)) / counts


def _polyphase_lifting_conditioning_from_phases(
    phases: Tensor,
    valid_phases: Tensor,
) -> Tensor:
    """Return parameter-free ``[sample,state,slot]`` PL-9 routing gains.

    The nine scores are action energies for identity, four clockwise local
    2x2 phase transfers, and the corresponding four source-phase-global-mean
    transfers.  Normalization followed by ``0.5 + 4.5 p`` keeps every gain in
    ``[0.5, 5]`` with exact slot mean one.  This is forward conditioning, not
    gradient whitening or optimizer projection.
    """

    valid_count = valid_phases.sum().clamp_min(1.0)
    identity_score = (
        phases.square() * valid_phases
    ).sum(dim=(2, 3, 4, 5)) / valid_count

    local_scores = []
    for source, target in _POLYPHASE_LIFTING_RING:
        pair_valid = (
            valid_phases[:, :, :, source]
            * valid_phases[:, :, :, target]
        )
        pair_count = pair_valid.sum().clamp_min(1.0)
        local_scores.append(
            (
                phases[:, :, :, source].square() * pair_valid
            ).sum(dim=(2, 3, 4))
            / pair_count
        )

    phase_means = _polyphase_lifting_valid_phase_means(
        phases, valid_phases
    )
    global_scores = []
    for source, target in _POLYPHASE_LIFTING_RING:
        target_valid = valid_phases[:, :, :, target]
        target_count = target_valid.sum().clamp_min(1.0)
        global_action = (
            phase_means[:, :, :, source, None, None] * target_valid
        )
        global_scores.append(
            global_action.square().sum(dim=(2, 3, 4)) / target_count
        )
    scores = torch.stack(
        (identity_score, *local_scores, *global_scores), dim=2
    )
    epsilon = torch.finfo(scores.dtype).eps
    probabilities = (scores + epsilon) / (
        scores + epsilon
    ).sum(dim=2, keepdim=True)
    return 0.5 + 4.5 * probabilities


def _polyphase_lifting_input_conditioning(
    paired_mixed: Tensor,
) -> Tensor:
    """Compute PL-9 action-energy routing directly from a paired field."""

    phases, valid_phases = _polyphase_lifting_phase_view(paired_mixed)
    return _polyphase_lifting_conditioning_from_phases(
        phases, valid_phases
    )


def _polyphase_complex_scale(
    action: Tensor,
    scale_real: Tensor,
    scale_imag: Tensor,
) -> Tensor:
    """Multiply a paired-complex phase action by one complex slot scale."""

    scale_real = scale_real[:, :, None, None]
    scale_imag = scale_imag[:, :, None, None]
    return torch.stack(
        (
            scale_real * action[:, :, 0]
            - scale_imag * action[:, :, 1],
            scale_imag * action[:, :, 0]
            + scale_real * action[:, :, 1],
        ),
        dim=2,
    )


def _fused_q9_polyphase_lifting_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """PL-9: parameter-matched local/global polyphase lifting compiler.

    The existing nine complex depthwise entries become one identity residual
    slot, four clockwise within-cell phase transfers, and four matching
    source-phase-global-mean transfers.  Live action energies route them per
    sample and state channel.  Subtracting the registered center-one anchor
    makes every effective residual gain zero at initialization, preserving the
    tap-major DCT parent's raw compiler function without adding parameters.
    """

    states = mix_m_real.shape[0]
    mixed = _q9_widely_linear_pointwise(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
    )
    batch, _, height, width = mixed.shape
    paired_mixed = mixed.view(batch, states, 2, height, width)
    phases, valid_phases = _polyphase_lifting_phase_view(paired_mixed)
    phase_means = _polyphase_lifting_valid_phase_means(
        phases, valid_phases
    )
    conditioning = _polyphase_lifting_conditioning_from_phases(
        phases, valid_phases
    )

    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(states, 9)
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(states, 9)
    identity_anchor = F.pad(
        torch.ones_like(gain_real[:, :1]), (0, 8)
    )
    residual_scale_real = conditioning * (
        gain_real - identity_anchor
    )[None]
    residual_scale_imag = conditioning * gain_imag[None]

    residual_by_target = [
        _polyphase_complex_scale(
            phases[:, :, :, phase],
            residual_scale_real[:, :, 0],
            residual_scale_imag[:, :, 0],
        )
        for phase in range(4)
    ]
    for ring_slot, (source, target) in enumerate(
        _POLYPHASE_LIFTING_RING
    ):
        source_valid = valid_phases[:, :, :, source]
        target_valid = valid_phases[:, :, :, target]
        local_action = (
            phases[:, :, :, source] * source_valid * target_valid
        )
        global_action = (
            phase_means[:, :, :, source, None, None] * target_valid
        )
        local_slot = 1 + ring_slot
        global_slot = 5 + ring_slot
        residual_by_target[target] = (
            residual_by_target[target]
            + _polyphase_complex_scale(
                local_action,
                residual_scale_real[:, :, local_slot],
                residual_scale_imag[:, :, local_slot],
            )
            + _polyphase_complex_scale(
                global_action,
                residual_scale_real[:, :, global_slot],
                residual_scale_imag[:, :, global_slot],
            )
        )

    residual_phases = torch.stack(residual_by_target, dim=3)
    phase_height, phase_width = residual_phases.shape[-2:]
    residual_padded = F.pixel_shuffle(
        residual_phases.reshape(
            batch, 8 * states, phase_height, phase_width
        ),
        2,
    ).view(batch, states, 2, 2 * phase_height, 2 * phase_width)
    paired = paired_mixed + residual_padded[..., :height, :width]
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    interleaved = (paired * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * states, height, width
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=states,
    )


def _fused_q9_glrc9_latent_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    *,
    include_mwf9: bool = False,
    mwf9_gain_real: Tensor | None = None,
    mwf9_gain_imag: Tensor | None = None,
    include_swf9: bool = False,
    swf9_gain_real: Tensor | None = None,
    swf9_gain_imag: Tensor | None = None,
    include_lms4_additive: bool = False,
    lms4_gain_real: Tensor | None = None,
    lms4_gain_imag: Tensor | None = None,
    include_lms4_signed: bool = False,
    lms4_signed_gain_real: Tensor | None = None,
    lms4_signed_gain_imag: Tensor | None = None,
    include_lms4_act4: bool = False,
    lms4_act4_gain_real: Tensor | None = None,
    lms4_act4_gain_imag: Tensor | None = None,
    include_lms4_dual_route: bool = False,
    lms4_dual_gain_real: Tensor | None = None,
    lms4_dual_gain_imag: Tensor | None = None,
    hierarchical_routing: bool = False,
    role_gain_real: Tensor | None = None,
    role_gain_imag: Tensor | None = None,
    block_role_code: Tensor | None = None,
    full_action_weight: Tensor | None = None,
    content_action_weight: Tensor | None = None,
    routing_continuation: Tensor | None = None,
    local_family_continuation: Tensor | None = None,
    mn_offdiag_continuation: Tensor | None = None,
    mn_offdiag_diagonal_endpoint: bool = False,
    mn_offdiag_materialized_reference: bool = False,
    omit_local_family: bool = False,
    direct_low3_spectrum: bool = False,
) -> Tensor:
    """Return the DH-9 latent immediately before the block-local H head.

    The existing nine complex depthwise scalars are reinterpreted without
    adding parameters.  Rolled DCT-coordinate slots ``00,10,11,20,21``
    control fixed low-frequency shells; ``01,02,12,22`` control a fixed
    compact even/odd tensor-product frame.  Parameter-free atom-action energy
    produces one bounded ``[sample,state,slot]`` routing tensor.  Routing
    multiplies residual coordinates, so the registered center-one
    initialization remains the exact identity latent map before radial SiLU.
    """

    if sum(
        (
            include_lms4_additive,
            include_lms4_signed,
            include_lms4_act4,
            include_lms4_dual_route,
        )
    ) > 1:
        raise ValueError(
            "DH-LMS4 additive, signed, ACT4, and dual-route modes are mutually exclusive"
        )
    d = mix_m_real.shape[0]
    if mn_offdiag_materialized_reference:
        if mn_offdiag_continuation is not None or mn_offdiag_diagonal_endpoint:
            raise ValueError(
                "materialized M/N reference graph must not receive endpoint flags"
            )
        mixed = mn_offdiag_materialized_reference_pointwise(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
        )
    elif mn_offdiag_diagonal_endpoint:
        if mn_offdiag_continuation is not None:
            raise ValueError(
                "diagonal M/N endpoint graph must not receive a scalar g"
            )
        mixed = mn_offdiag_diagonal_pointwise(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
        )
    elif mn_offdiag_continuation is None:
        mixed = _q9_widely_linear_pointwise(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
        )
    else:
        if mn_offdiag_continuation.numel() != 1:
            raise ValueError("GLRC9 M/N continuation g must be scalar")
        mixed = mn_offdiag_continuation_pointwise(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            mn_offdiag_continuation,
        )
    batch, _, height, width = mixed.shape
    paired_mixed = mixed.view(batch, d, 2, height, width)
    spectrum = (
        direct_low3_dct(paired_mixed)
        if direct_low3_spectrum
        else _dct_ii_ortho_2d(paired_mixed)
    )
    spectrum_real = spectrum[:, :, 0]
    spectrum_imag = spectrum[:, :, 1]
    shells = _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )
    frame = _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )
    # The RC2 g=0 endpoint has a dedicated no-local graph.  It computes only
    # scalar routing scores (without constructing the four response tensor)
    # and never injects a local action into the latent.
    local_actions = (
        None
        if omit_local_family
        else _dct_hybrid_local_actions(paired_mixed)
    )

    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    role_values = (role_gain_real, role_gain_imag, block_role_code)
    if any(value is not None for value in role_values):
        if not all(value is not None for value in role_values):
            raise ValueError("GCHAF-R1 requires both role gains and one role code")
        if not hierarchical_routing:
            raise ValueError("GCHAF-R1 role gains require hierarchical routing")
        assert role_gain_real is not None
        assert role_gain_imag is not None
        assert block_role_code is not None
        if role_gain_real.shape != (d, 9) or role_gain_imag.shape != (d, 9):
            raise ValueError("GCHAF-R1 role gains must have shape [D,9]")
        if block_role_code.numel() != 1:
            raise ValueError("GCHAF-R1 block role code must be scalar")
        role = block_role_code.reshape(()).to(
            device=gain_real.device, dtype=gain_real.dtype
        )
        gain_real = gain_real + role * role_gain_real
        gain_imag = gain_imag + role * role_gain_imag
    if omit_local_family:
        conditioning = _dct_hybrid_input_conditioning_no_local(
            paired_mixed,
            spectrum_real,
            spectrum_imag,
            shells,
            frame,
        )
    else:
        conditioning_function = (
            _dct_hybrid_hierarchical_input_conditioning
            if hierarchical_routing
            else _dct_hybrid_input_conditioning
        )
        conditioning = conditioning_function(
            paired_mixed,
            spectrum_real,
            spectrum_imag,
            shells,
            frame,
            local_actions,
        )
    if routing_continuation is not None:
        # Parent GLRC9 uses ``conditioning = 0.5 + 4.5*p``.  Interpolating the
        # underlying probability with uniform p=1/9 is exactly the affine
        # conditioning map below.  The affine path is exact at g=1 and the
        # explicit g=0 branch removes endpoint roundoff while remaining
        # torch.compile-safe.
        conditioning = glrc9_routing_conditioning(
            conditioning,
            routing_continuation,
        )
    identity_slot = F.pad(
        torch.ones_like(gain_real[:, :1]), (0, 8)
    )
    residual_gain_real = (gain_real - identity_slot)[None]
    residual_gain_imag = gain_imag[None]
    shell_slots = (0, 3, 4, 6, 7)
    local_slots = (1, 2, 5, 8)
    shell_conditioning = conditioning[:, :, :5]
    local_conditioning = conditioning[:, :, 5:]
    response_scale = _GLRC9_RESPONSE_SCALE_CONTEXT
    if response_scale is not None:
        global_scale, local_scale = response_scale
        # Slot 0 is the identity/DC anchor.  Only the remaining four shell
        # coordinates and all four local coordinates are interpolated.  The
        # result still passes through the unchanged grouped H readout and
        # ``Pi_q9`` legalizer, so negative or >1 scales cannot bypass the
        # physical action contract.
        shell_conditioning = torch.cat(
            (
                shell_conditioning[:, :, :1],
                shell_conditioning[:, :, 1:] * global_scale,
            ),
            dim=2,
        )
        local_conditioning = local_conditioning * local_scale
    shell_real = (
        shell_conditioning * residual_gain_real[:, :, shell_slots]
    )
    shell_imag = (
        shell_conditioning * residual_gain_imag[:, :, shell_slots]
    )
    multiplier_real = torch.einsum("bds,shw->bdhw", shell_real, shells)
    multiplier_imag = torch.einsum("bds,shw->bdhw", shell_imag, shells)
    if include_lms4_additive or include_lms4_act4 or include_lms4_dual_route:
        if lms4_gain_real is None or lms4_gain_imag is None:
            raise ValueError("DH-LMS4-A requires both complex additive gains")
        additive_bands = _dh_lms4_additive_bands(
            height,
            width,
            device=packed_f.device,
            dtype=spectrum_real.dtype,
        )
        additive_conditioning = _dh_lms4_additive_input_conditioning(
            spectrum_real,
            spectrum_imag,
            additive_bands,
        )
        # The dual-route candidate keeps its unconditional positive-energy
        # branch on the three low/mid bands only.  Its fourth (fused high and
        # ultra-high) band survives solely in the zero-initialized complex
        # residual below, where live-f conditioning can close, reverse, or
        # phase-rotate it instead of forcing a positive high-frequency gain.
        positive_band_count = 3 if include_lms4_dual_route else 4
        positive_bands = additive_bands[:positive_band_count]
        positive_conditioning = additive_conditioning[:, :, :positive_band_count]
        additive_real = positive_conditioning * lms4_gain_real[None]
        additive_imag = positive_conditioning * lms4_gain_imag[None]
        multiplier_real = multiplier_real + torch.einsum(
            "bds,shw->bdhw", additive_real, positive_bands
        )
        multiplier_imag = multiplier_imag + torch.einsum(
            "bds,shw->bdhw", additive_imag, positive_bands
        )
        if include_lms4_act4:
            if lms4_act4_gain_real is None or lms4_act4_gain_imag is None:
                raise ValueError(
                    "DH-LMS4-ACT4 requires both complex angular gains"
                )
            angular_bands = _dh_lms4_act4_bands(
                height,
                width,
                device=packed_f.device,
                dtype=spectrum_real.dtype,
            )
            angular_conditioning = _dh_lms4_act4_input_conditioning(
                spectrum_real,
                spectrum_imag,
                angular_bands,
            )
            angular_real = angular_conditioning * lms4_act4_gain_real[None]
            angular_imag = angular_conditioning * lms4_act4_gain_imag[None]
            multiplier_real = multiplier_real + torch.einsum(
                "bds,shw->bdhw", angular_real, angular_bands
            )
            multiplier_imag = multiplier_imag + torch.einsum(
                "bds,shw->bdhw", angular_imag, angular_bands
            )
        if include_lms4_dual_route:
            if lms4_dual_gain_real is None or lms4_dual_gain_imag is None:
                raise ValueError(
                    "DH-LMS4 dual route requires both complex residual gains"
                )
            angular_bands = _dh_lms4_act4_bands(
                height,
                width,
                device=packed_f.device,
                dtype=spectrum_real.dtype,
            )
            source_code = _dh_lms4_dual_route_conditioning(
                spectrum_real,
                spectrum_imag,
                additive_bands,
                angular_bands,
            )
            route_real = source_code[..., 0]
            route_imag = source_code[..., 1]
            residual_real = (
                route_real * lms4_dual_gain_real[None]
                - route_imag * lms4_dual_gain_imag[None]
            )
            residual_imag = (
                route_real * lms4_dual_gain_imag[None]
                + route_imag * lms4_dual_gain_real[None]
            )
            multiplier_real = multiplier_real + torch.einsum(
                "bds,shw->bdhw", residual_real, additive_bands
            )
            multiplier_imag = multiplier_imag + torch.einsum(
                "bds,shw->bdhw", residual_imag, additive_bands
            )
    if include_lms4_signed:
        if lms4_signed_gain_real is None or lms4_signed_gain_imag is None:
            raise ValueError("signed DH-LMS4 requires both complex gamma tensors")
        signed_bands = _dh_lms4_additive_bands(
            height,
            width,
            device=packed_f.device,
            dtype=spectrum_real.dtype,
        )
        signed_conditioning = _dh_lms4_signed_input_conditioning(
            spectrum_real,
            spectrum_imag,
            signed_bands,
        )
        signed_real = signed_conditioning * lms4_signed_gain_real[None]
        signed_imag = signed_conditioning * lms4_signed_gain_imag[None]
        multiplier_real = multiplier_real + torch.einsum(
            "bds,shw->bdhw", signed_real, signed_bands
        )
        multiplier_imag = multiplier_imag + torch.einsum(
            "bds,shw->bdhw", signed_imag, signed_bands
        )
    residual_spectrum_real = (
        multiplier_real * spectrum_real
        - multiplier_imag * spectrum_imag
    )
    residual_spectrum_imag = (
        multiplier_imag * spectrum_real
        + multiplier_real * spectrum_imag
    )
    spectral_value = torch.stack(
        (residual_spectrum_real, residual_spectrum_imag), dim=2
    )
    spectral_residual = (
        direct_low3_idct(spectral_value, output_dtype=paired_mixed.dtype)
        if direct_low3_spectrum
        else _idct_iii_ortho_2d(
            spectral_value,
            output_dtype=paired_mixed.dtype,
        )
    )

    if omit_local_family:
        local_residual = torch.zeros_like(paired_mixed)
    else:
        assert local_actions is not None
        local_gain_real = (
            local_conditioning * residual_gain_real[:, :, local_slots]
        )
        local_gain_imag = (
            local_conditioning * residual_gain_imag[:, :, local_slots]
        )
        local_action_real = local_actions[:, :, :, 0]
        local_action_imag = local_actions[:, :, :, 1]
        local_residual = torch.stack(
            (
                (
                    local_gain_real[..., None, None] * local_action_real
                    - local_gain_imag[..., None, None] * local_action_imag
                ).sum(dim=2),
                (
                    local_gain_imag[..., None, None] * local_action_real
                    + local_gain_real[..., None, None] * local_action_imag
                ).sum(dim=2),
            ),
            dim=2,
        )
    if local_family_continuation is not None:
        if local_family_continuation.numel() != 1:
            raise ValueError("GLRC9 local-family continuation g must be scalar")
        local_residual = scale_local_family_residual(
            local_residual, local_family_continuation
        )

    paired = paired_mixed + spectral_residual + local_residual
    if full_action_weight is not None and content_action_weight is not None:
        raise ValueError("AR9F and CCAR9 action residuals are separate arms")
    if full_action_weight is not None:
        if full_action_weight.shape != (d, 2, 9, 2):
            raise ValueError(
                "GCHAF-AR9F full action weight must have shape [D,2,9,2]"
            )

        # Preserve all nine pre-sum action identities without materializing a
        # nine-way full-resolution activation.  Action zero is the identity
        # carrier plus its DC-shell residual; actions 1--4 are the remaining
        # global shells and actions 5--8 are the four local frame responses.
        # A general real 2x2 map is retained per action, so this includes both
        # complex-linear and conjugate response components.
        base_action_residual = torch.einsum(
            "doi,bdihw->bdohw",
            full_action_weight[:, :, 0],
            paired_mixed,
        )

        shell_weight_real = full_action_weight[:, :, :5, 0]
        shell_weight_imag = full_action_weight[:, :, :5, 1]
        mapped_spectrum_from_real = (
            torch.einsum(
                "dos,bds,shw->bdohw",
                shell_weight_real,
                shell_real,
                shells,
            )
            + torch.einsum(
                "dos,bds,shw->bdohw",
                shell_weight_imag,
                shell_imag,
                shells,
            )
        )
        mapped_spectrum_from_imag = (
            -torch.einsum(
                "dos,bds,shw->bdohw",
                shell_weight_real,
                shell_imag,
                shells,
            )
            + torch.einsum(
                "dos,bds,shw->bdohw",
                shell_weight_imag,
                shell_real,
                shells,
            )
        )
        mapped_spectrum = (
            mapped_spectrum_from_real * spectrum_real.unsqueeze(2)
            + mapped_spectrum_from_imag * spectrum_imag.unsqueeze(2)
        )
        mapped_spectral_residual = _idct_iii_ortho_2d(
            mapped_spectrum,
            output_dtype=paired_mixed.dtype,
        )

        local_action_contributions = torch.stack(
            (
                local_gain_real[..., None, None] * local_action_real
                - local_gain_imag[..., None, None] * local_action_imag,
                local_gain_imag[..., None, None] * local_action_real
                + local_gain_real[..., None, None] * local_action_imag,
            ),
            dim=3,
        )
        mapped_local_residual = torch.einsum(
            "dosi,bdsihw->bdohw",
            full_action_weight[:, :, 5:],
            local_action_contributions,
        )
        paired = paired + (
            base_action_residual
            + mapped_spectral_residual
            + mapped_local_residual
        )
    if content_action_weight is not None:
        if content_action_weight.shape != (d, 2, 9, 8, 2):
            raise ValueError(
                "GCHAF-CCAR9 content-action weight must have shape "
                "[D,2,9,8,2]"
            )

        # Decode content from the unchanged GLRC9 router, then contract the
        # eight-dimensional code directly into each action response.  This
        # retains the full general-real action map while avoiding any
        # [action,content,height,width] outer-product activation.
        content_code = _apply_ccar9_content_batch_permutation(
            gchaf_ccar9_content_code(conditioning)
        )
        base_action_residual = torch.einsum(
            "dori,bdr,bdihw->bdohw",
            content_action_weight[:, :, 0],
            content_code,
            paired_mixed,
        )

        shell_weight_real = content_action_weight[:, :, :5, :, 0]
        shell_weight_imag = content_action_weight[:, :, :5, :, 1]
        mapped_spectrum_from_real = (
            torch.einsum(
                "dosr,bdr,bds,shw->bdohw",
                shell_weight_real,
                content_code,
                shell_real,
                shells,
            )
            + torch.einsum(
                "dosr,bdr,bds,shw->bdohw",
                shell_weight_imag,
                content_code,
                shell_imag,
                shells,
            )
        )
        mapped_spectrum_from_imag = (
            -torch.einsum(
                "dosr,bdr,bds,shw->bdohw",
                shell_weight_real,
                content_code,
                shell_imag,
                shells,
            )
            + torch.einsum(
                "dosr,bdr,bds,shw->bdohw",
                shell_weight_imag,
                content_code,
                shell_real,
                shells,
            )
        )
        mapped_spectrum = (
            mapped_spectrum_from_real * spectrum_real.unsqueeze(2)
            + mapped_spectrum_from_imag * spectrum_imag.unsqueeze(2)
        )
        mapped_spectral_residual = _idct_iii_ortho_2d(
            mapped_spectrum,
            output_dtype=paired_mixed.dtype,
        )

        local_action_contributions = torch.stack(
            (
                local_gain_real[..., None, None] * local_action_real
                - local_gain_imag[..., None, None] * local_action_imag,
                local_gain_imag[..., None, None] * local_action_real
                + local_gain_real[..., None, None] * local_action_imag,
            ),
            dim=3,
        )
        mapped_local_residual = torch.einsum(
            "dosri,bdr,bdsihw->bdohw",
            content_action_weight[:, :, 5:],
            content_code,
            local_action_contributions,
        )
        paired = paired + (
            base_action_residual
            + mapped_spectral_residual
            + mapped_local_residual
        )
    if include_mwf9:
        graft_gain_real = (
            residual_gain_real[0]
            if mwf9_gain_real is None
            else mwf9_gain_real
        )
        graft_gain_imag = (
            residual_gain_imag[0]
            if mwf9_gain_imag is None
            else mwf9_gain_imag
        )
        paired = paired + DH_MWF9_ETA * _dh_mwf9_wavelet_residual(
            paired_mixed,
            graft_gain_real,
            graft_gain_imag,
        )
    if include_swf9:
        if swf9_gain_real is None or swf9_gain_imag is None:
            raise ValueError("DH-SWF9-I requires both complex gamma tensors")
        paired = paired + _dh_swf9_wavelet_residual(
            paired_mixed,
            swf9_gain_real,
            swf9_gain_imag,
        )
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    interleaved = (paired * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    return interleaved


def _fused_q9_dct_hybrid_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    routing_continuation: Tensor | None = None,
) -> Tensor:
    """DH-9: five global DCT shells plus four local directional atoms."""

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            routing_continuation=routing_continuation,
            direct_low3_spectrum=True,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_dct_hybrid_local_family_continuation_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    local_family_continuation: Tensor,
) -> Tensor:
    """RC2 interior graph: only the separated four-local residual is scaled."""

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            local_family_continuation=local_family_continuation,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_dct_hybrid_no_local_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """RC2 g=0 endpoint; no local action response is constructed/injected."""

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            omit_local_family=True,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_dct_hybrid_mn_offdiag_continuation_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    mn_offdiag_continuation: Tensor,
) -> Tensor:
    """RC3 interior M/N continuation graph (strictly ``0 < g < 1``).

    Endpoint routing is performed in Python by :class:`ComplexF9Q9Compiler`;
    this graph never inspects the value of ``g``.
    """

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            mn_offdiag_continuation=mn_offdiag_continuation,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_dct_hybrid_mn_offdiag_diagonal_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """RC3 g=0 diagonal-only graph.

    This graph has no scalar input and calls the diagonal pointwise map
    directly, so the dense off-diagonal entries are not materialized or
    consumed at the endpoint.
    """

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            mn_offdiag_diagonal_endpoint=True,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_dct_hybrid_mn_offdiag_materialized_reference_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """Independent masked materialized reference graph for Canary only."""

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            mn_offdiag_materialized_reference=True,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_gchaf_r1_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    role_gain_real: Tensor,
    role_gain_imag: Tensor,
    block_role_code: Tensor,
) -> Tensor:
    """GCHAF-R1: exact hierarchical GLRC9 plus one fixed block-role tangent."""

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            hierarchical_routing=True,
            role_gain_real=role_gain_real,
            role_gain_imag=role_gain_imag,
            block_role_code=block_role_code,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_gchaf_ar9f_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    full_action_weight: Tensor,
) -> Tensor:
    """GCHAF-AR9F: exact GLRC9 plus one full pre-sum action residual."""

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            full_action_weight=full_action_weight,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_gchaf_ccar9_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    content_action_weight: Tensor,
) -> Tensor:
    """GCHAF-CCAR9: exact GLRC9 plus a content-conditioned action residual."""

    d = mix_m_real.shape[0]
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            content_action_weight=content_action_weight,
        ),
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_dct_global9_conditioned_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """D-arm control: nine live-conditioned global DCT-II coordinates only."""

    d = mix_m_real.shape[0]
    mixed = _q9_widely_linear_pointwise(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
    )
    batch, _, height, width = mixed.shape
    paired_mixed = mixed.view(batch, d, 2, height, width)
    spectrum = _dct_ii_ortho_2d(paired_mixed)
    spectrum_real = spectrum[:, :, 0]
    spectrum_imag = spectrum[:, :, 1]
    coordinate_masks = _dct_global9_coordinate_masks(
        height,
        width,
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )

    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    conditioning = _dct_global9_input_conditioning(
        spectrum_real,
        spectrum_imag,
        coordinate_masks,
    )
    identity_slot = F.pad(
        torch.ones_like(gain_real[:, :1]), (0, 8)
    )
    residual_gain_real = (gain_real - identity_slot)[None]
    residual_gain_imag = gain_imag[None]
    multiplier_real = torch.einsum(
        "bds,shw->bdhw",
        conditioning * residual_gain_real,
        coordinate_masks,
    )
    multiplier_imag = torch.einsum(
        "bds,shw->bdhw",
        conditioning * residual_gain_imag,
        coordinate_masks,
    )
    residual_spectrum_real = (
        multiplier_real * spectrum_real
        - multiplier_imag * spectrum_imag
    )
    residual_spectrum_imag = (
        multiplier_imag * spectrum_real
        + multiplier_real * spectrum_imag
    )
    spectral_residual = _idct_iii_ortho_2d(
        torch.stack(
            (residual_spectrum_real, residual_spectrum_imag),
            dim=2,
        ),
        output_dtype=paired_mixed.dtype,
    )
    paired = paired_mixed + spectral_residual
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    interleaved = (paired * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    head_padding = (head_weight.shape[-1] - 1) // 2
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=head_padding,
        groups=d,
    )


def _fused_q9_dct_hybrid_cseb_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """DH-9 readout with centered state-energy bias modulation.

    The DH-9 latent is ``[B, 2D, H, W]`` with state-major interleaved
    ``[re_0, im_0, re_1, im_1, ...]`` channels.  The grouped H rows are
    likewise state-major ``[state, tap]``.  CSEB keeps the bias-free grouped
    convolution unchanged and only reinterprets the existing bias as a
    bounded, spatially centered live-latent modulation.
    """

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    batch, channels, height, width = latent.shape
    if channels != 2 * d:
        raise ValueError("CSEB latent channels must equal 2 * state_channels")

    state_complex = latent.reshape(batch, d, 2, height, width)
    eps = torch.finfo(latent.dtype).eps
    energy = torch.sqrt(state_complex.square().sum(dim=2) + eps)
    energy_mean = energy.mean(dim=(-2, -1), keepdim=True)
    amplitude = torch.tanh(energy / (energy_mean + eps) - 1.0)
    psi = amplitude - amplitude.mean(dim=(-2, -1), keepdim=True)

    conv_without_bias = F.conv2d(
        latent,
        head_weight,
        None,
        padding=1,
        groups=d,
    ).reshape(batch, d, 9, height, width)
    modulated_bias = head_bias.reshape(1, d, 9, 1, 1) * (
        1.0 + 0.25 * psi.unsqueeze(2)
    )
    return (conv_without_bias + modulated_bias).reshape(
        batch, 9 * d, height, width
    )


def _fused_q9_dct_hybrid_pdr_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    pdr_v_weight: Tensor,
    pdr_u_weight: Tensor,
) -> Tensor:
    """DH-9 plus a block-private post-latent low-rank q9 tangent.

    The latent is shared by the parent ``H`` readout and the PDR branch.  Both
    branches therefore consume the same ``z_b`` immediately before ``H_b``;
    the residual is added in the existing grouped-H raw channel layout and is
    passed to the unchanged q9 legalizer by the caller.
    """

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    parent_raw = F.conv2d(
        latent,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )
    pdr_raw = F.conv2d(
        F.conv2d(latent, pdr_v_weight),
        pdr_u_weight,
    )
    return parent_raw + pdr_raw


def _fused_q9_dct_hybrid_hlr_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    hlr_down_weight: Tensor,
    hlr_up_weight: Tensor,
    hlr_gate_logit: Tensor,
) -> Tensor:
    """DH-9 with a bounded all-block low-rank remap before H."""

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    low_rank = F.conv2d(
        F.conv2d(latent, hlr_down_weight),
        hlr_up_weight,
    )
    denominator = 1.0 + (
        torch.linalg.vector_norm(hlr_down_weight)
        * torch.linalg.vector_norm(hlr_up_weight)
    )
    envelope = 0.5 * torch.sigmoid(hlr_gate_logit)
    remapped = latent + (envelope / denominator) * low_rank
    return F.conv2d(
        remapped,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dct_hybrid_h2_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    h2_weight: Tensor,
    h2_gain: Tensor,
) -> Tensor:
    """DH-9 with a bounded second latent decoding depth before H."""

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    refined = F.conv2d(latent, h2_weight, padding=1, groups=d)
    batch, _, height, width = refined.shape
    paired = refined.reshape(batch, d, 2, height, width)
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(refined.dtype).eps
    )
    refined = paired * torch.sigmoid(magnitude).unsqueeze(2)
    # Young's inequality and Cauchy--Schwarz give
    # ||conv_W||_2 <= sum_delta ||W_delta||_2 <= 3 ||W||_F
    # for a 3x3 grouped convolution.  The radial SiLU used above is
    # norm-shrinking, and the per-state tanh envelope remains below 1/2.
    # Normalize every 2x2 state block independently.  A global Frobenius
    # norm would scale as sqrt(D) at the Dirac initialization and would make
    # the first gain tangent artificially weaker in wider CLPO blocks.
    state_weight_norm = torch.linalg.vector_norm(
        h2_weight.reshape(d, -1),
        dim=1,
    )
    denominator = 1.0 + 3.0 * state_weight_norm
    envelope = 0.5 * torch.tanh(h2_gain)
    envelope = (envelope / denominator).view(1, d, 1, 1, 1)
    remapped = latent + (envelope * refined).reshape_as(latent)
    return F.conv2d(
        remapped,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dct_hybrid_h2_givens_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    amplitude_gain: Tensor,
    horizontal_gain: Tensor,
    vertical_gain: Tensor,
) -> Tensor:
    """DH-9 with credit-decoupled bounded Givens refinement before H."""

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    batch, _, height, width = latent.shape
    paired = latent.reshape(batch, d, 2, height, width)
    transported = givens_strang_transport(
        paired,
        paired,
        horizontal_gain,
        vertical_gain,
        eta=0.5,
    )
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(latent.dtype).eps
    )
    radial = paired * torch.sigmoid(magnitude).unsqueeze(2)
    amplitude = (0.1 * torch.tanh(amplitude_gain)).view(1, d, 1, 1, 1)
    remapped = (transported + amplitude * radial).reshape_as(latent)
    return F.conv2d(
        remapped,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_lms4_additive_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """DH-LMS4-A: unchanged DH plus four additive broad DCT bands."""

    d = mix_m_real.shape[0]
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            include_lms4_additive=True,
            lms4_gain_real=gamma_real,
            lms4_gain_imag=gamma_imag,
        ),
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_lms4_signed_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """Signed source-conditioned LMS4: DH plus centered-log-energy bands."""

    d = mix_m_real.shape[0]
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            include_lms4_signed=True,
            lms4_signed_gain_real=gamma_real,
            lms4_signed_gain_imag=gamma_imag,
        ),
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_lms4_act4_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    eta_real: Tensor,
    eta_imag: Tensor,
) -> Tensor:
    """DH-LMS4-ACT4: radial parent plus four angular companions."""

    d = mix_m_real.shape[0]
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            include_lms4_act4=True,
            lms4_gain_real=gamma_real,
            lms4_gain_imag=gamma_imag,
            lms4_act4_gain_real=eta_real,
            lms4_act4_gain_imag=eta_imag,
        ),
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_lms4_dual_route_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
    eta_real: Tensor,
    eta_imag: Tensor,
) -> Tensor:
    """DH-LMS4 with a bounded complex source-conditioned residual route."""

    d = mix_m_real.shape[0]
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            include_lms4_dual_route=True,
            lms4_gain_real=gamma_real,
            lms4_gain_imag=gamma_imag,
            lms4_dual_gain_real=eta_real,
            lms4_dual_gain_imag=eta_imag,
        ),
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_mwf9_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """DH-MWF9: DH-9 plus a fixed three-level Haar detail graft."""

    d = mix_m_real.shape[0]
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            include_mwf9=True,
        ),
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_mwf9_independent_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """DH-MWF9-I: DH-9 plus an independent fixed Haar tangent."""

    d = mix_m_real.shape[0]
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            include_mwf9=True,
            mwf9_gain_real=gamma_real,
            mwf9_gain_imag=gamma_imag,
        ),
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_swf9_independent_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """DH-SWF9-I: DH latent plus an independent zero-boundary SWF9 tangent."""

    d = mix_m_real.shape[0]
    return F.conv2d(
        _fused_q9_glrc9_latent_graph(
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            include_swf9=True,
            swf9_gain_real=gamma_real,
            swf9_gain_imag=gamma_imag,
        ),
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dh_swf9_scale_tied_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """DH-SWF9-ST: share each directional gain across all three scales."""

    d = mix_m_real.shape[0]
    expected = (d, len(DH_SWF9_BANDS))
    if gamma_real.shape != expected or gamma_imag.shape != expected:
        raise ValueError("DH-SWF9-ST gamma tensors must have shape [D,3]")
    repeated_real = gamma_real.repeat(1, len(DH_SWF9_SCALES))
    repeated_imag = gamma_imag.repeat(1, len(DH_SWF9_SCALES))
    return _fused_q9_dh_swf9_independent_operator_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
        head_weight,
        head_bias,
        repeated_real,
        repeated_imag,
    )


def _fused_q9_dh_swf9_scale_rank2_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    gamma_real: Tensor,
    gamma_imag: Tensor,
) -> Tensor:
    """DH-SWF9-SR2: fixed orthonormal rank-2 chart over three scales.

    Each direction owns a complex scale mean and coarse/fine contrast.  The
    omitted ``(1,-2,1)/sqrt(6)`` coordinate is the scale-curvature mode.  This
    preserves a task-active local/global distinction without introducing an
    input router or a learned scale basis.
    """

    d = mix_m_real.shape[0]
    expected = (d, 2 * len(DH_SWF9_BANDS))
    if gamma_real.shape != expected or gamma_imag.shape != expected:
        raise ValueError("DH-SWF9-SR2 gamma tensors must have shape [D,6]")

    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def expand_scale_chart(gamma: Tensor) -> Tensor:
        coefficient = gamma.view(d, 2, len(DH_SWF9_BANDS))
        mean = coefficient[:, 0] * inv_sqrt3
        contrast = coefficient[:, 1] * inv_sqrt2
        return torch.stack(
            (mean + contrast, mean, mean - contrast), dim=1
        ).reshape(d, 9)

    return _fused_q9_dh_swf9_independent_operator_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
        head_weight,
        head_bias,
        expand_scale_chart(gamma_real),
        expand_scale_chart(gamma_imag),
    )


def _q9_split_half_to_state_interleaved(packed_f: Tensor) -> Tensor:
    """Convert canonical ``[Re_0..Re_D, Im_0..Im_D]`` packing to pairs.

    The public q9/compiler boundary remains split-half packed.  A grouped
    convolution with ``groups=D`` instead requires each state's real and
    imaginary coordinates to be adjacent: ``[Re_0, Im_0, ..., Re_D, Im_D]``.
    This helper is deliberately used only by the CCP tangent and its gate so
    that every pre-existing compiler mode keeps its frozen coordinate order.
    """

    if packed_f.ndim != 4 or packed_f.shape[1] <= 0 or packed_f.shape[1] % 2:
        raise ValueError("packed_f must have shape [B,2D,H,W] with D > 0")
    real, imag = packed_f.chunk(2, dim=1)
    return torch.stack((real, imag), dim=2).flatten(1, 2)


def _glrc9_ccp_channel_gate(packed_f: Tensor) -> Tensor:
    """Return a parameter-free positive bounded per-state channel gate.

    With ``e_d = mean_{real,imag,h,w}(f_d^2) >= 0``, use
    ``phi_d = 1/4 + 1/2 sigmoid(e_d)``.  Mathematically, finite real inputs
    give ``1/2 <= phi_d < 3/4``.  Floating-point sigmoid can round to exactly
    ``3/4``, so the executable contract accepts the closed interval
    ``[1/2, 3/4]`` without adding a router or optimizer transform.
    """

    if packed_f.ndim != 4 or packed_f.shape[1] % 2:
        raise ValueError("packed_f must have shape [B,2D,H,W]")
    batch, packed_channels, height, width = packed_f.shape
    d = packed_channels // 2
    paired = _q9_split_half_to_state_interleaved(packed_f).view(
        batch, d, 2, height, width
    )
    energy_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    energy = paired.to(dtype=energy_dtype).square().mean(dim=(2, 3, 4))
    phi = 0.25 + 0.5 * torch.sigmoid(energy)
    return phi.to(dtype=packed_f.dtype).unsqueeze(-1).unsqueeze(-1)


def glrc9_ccp_heads_enabled(state_channels: int) -> bool:
    """Keep private live-f tangents where DH showed non-negative geometry.

    The D16 blocks retain the exact DH-9 compiler.  D32/D64 receive the
    block-private raw-q9 tangent.  This is a structural width mask, not a
    gradient rescaling or optimizer intervention.
    """

    return state_channels >= 32


def _fused_q9_glrc9_ccp_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    private_head_weight: Tensor,
    private_head_bias: Tensor,
) -> Tensor:
    """GLRC9-CCP raw field ``DH + phi_b(f) B_b(f)``."""

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    raw_dh = F.conv2d(
        latent, head_weight, head_bias, padding=1, groups=d
    )
    private_input = _q9_split_half_to_state_interleaved(packed_f)
    private = F.conv2d(
        private_input,
        private_head_weight,
        private_head_bias,
        padding=1,
        groups=d,
    )
    output_gate = _glrc9_ccp_channel_gate(packed_f).repeat_interleave(9, dim=1)
    return raw_dh + output_gate * private


def _fused_q9_glrc9_clustered_h_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    clustered_h_weight: Tensor,
) -> Tensor:
    """GLRC9 raw field with a shared, content-gated cluster H residual.

    The ordinary block-private GLRC9 head remains the parent path.  One zero-
    initialized grouped 3x3 readout is shared by the two blocks in a frozen
    width/role cluster and is modulated only by a parameter-free live-f gate.
    """

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    raw_dh = F.conv2d(
        latent, head_weight, head_bias, padding=1, groups=d
    )
    clustered = F.conv2d(
        latent,
        clustered_h_weight,
        None,
        padding=1,
        groups=d,
    )
    output_gate = _glrc9_ccp_channel_gate(packed_f).repeat_interleave(9, dim=1)
    return raw_dh + output_gate * clustered


def _fused_q9_glrc9_role_film_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    role_film_a: Tensor,
    role_film_b: Tensor,
) -> Tensor:
    """GLRC9 latent with a bounded, content-gated role-FiLM before private H.

    The live-f gate is parameter-free and computed once from the same packed
    drive used by GLRC9.  The two zero-initialized vectors are bounded with tanh;
    the shift is scaled by the per-pixel latent channel RMS.  The unchanged
    grouped H, q9 legalizer, F9 boundary, and K12 carrier remain downstream.
    """

    d = mix_m_real.shape[0]
    latent = _fused_q9_glrc9_latent_graph(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
        depthwise_real,
        depthwise_imag,
    )
    gate = _glrc9_ccp_channel_gate(packed_f).repeat_interleave(2, dim=1)
    scale = (0.25 * torch.tanh(role_film_a)).view(1, 2 * d, 1, 1)
    shift = (0.25 * torch.tanh(role_film_b)).view(1, 2 * d, 1, 1)
    latent_rms = torch.sqrt(
        latent.square().mean(dim=1, keepdim=True)
        + torch.finfo(latent.dtype).eps
    )
    modulated = (1.0 + gate * scale) * latent + gate * shift * latent_rms
    return F.conv2d(
        modulated,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dct_hybrid_learned_router_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    router_weight: Tensor,
    router_bias: Tensor,
) -> Tensor:
    """DH-MSR-9: DH-9 with a bounded block-local simplex router.

    The five shell and four local actions are identical to DH-9.  Only their
    parameter-free action-energy distribution is transported by the learned
    mass-preserving simplex chart.  The fixed parent mixture protects the
    empirically valuable DH routes while exposing independent routing tangent
    directions to ordinary end-to-end optimization.
    """

    d = mix_m_real.shape[0]
    mixed = _q9_widely_linear_pointwise(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
    )
    batch, _, height, width = mixed.shape
    paired_mixed = mixed.view(batch, d, 2, height, width)
    spectrum = _dct_ii_ortho_2d(paired_mixed)
    spectrum_real = spectrum[:, :, 0]
    spectrum_imag = spectrum[:, :, 1]
    shells = _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )
    frame = _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )
    local_actions = _dct_hybrid_local_actions(paired_mixed)

    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    base_conditioning = _dct_hybrid_input_conditioning(
        paired_mixed,
        spectrum_real,
        spectrum_imag,
        shells,
        frame,
        local_actions,
    )
    conditioning = _dct_hybrid_learned_simplex_conditioning(
        base_conditioning,
        router_weight,
        router_bias,
        parent_fraction=0.5,
    )
    identity_slot = F.pad(
        torch.ones_like(gain_real[:, :1]), (0, 8)
    )
    residual_gain_real = (gain_real - identity_slot)[None]
    residual_gain_imag = gain_imag[None]
    shell_slots = (0, 3, 4, 6, 7)
    local_slots = (1, 2, 5, 8)
    shell_conditioning = conditioning[:, :, :5]
    local_conditioning = conditioning[:, :, 5:]
    shell_real = (
        shell_conditioning * residual_gain_real[:, :, shell_slots]
    )
    shell_imag = (
        shell_conditioning * residual_gain_imag[:, :, shell_slots]
    )
    multiplier_real = torch.einsum("bds,shw->bdhw", shell_real, shells)
    multiplier_imag = torch.einsum("bds,shw->bdhw", shell_imag, shells)
    residual_spectrum_real = (
        multiplier_real * spectrum_real
        - multiplier_imag * spectrum_imag
    )
    residual_spectrum_imag = (
        multiplier_imag * spectrum_real
        + multiplier_real * spectrum_imag
    )
    spectral_residual = _idct_iii_ortho_2d(
        torch.stack(
            (residual_spectrum_real, residual_spectrum_imag), dim=2
        ),
        output_dtype=paired_mixed.dtype,
    )

    local_gain_real = (
        local_conditioning * residual_gain_real[:, :, local_slots]
    )
    local_gain_imag = (
        local_conditioning * residual_gain_imag[:, :, local_slots]
    )
    local_action_real = local_actions[:, :, :, 0]
    local_action_imag = local_actions[:, :, :, 1]
    local_residual = torch.stack(
        (
            (
                local_gain_real[..., None, None] * local_action_real
                - local_gain_imag[..., None, None] * local_action_imag
            ).sum(dim=2),
            (
                local_gain_imag[..., None, None] * local_action_real
                + local_gain_real[..., None, None] * local_action_imag
            ).sum(dim=2),
        ),
        dim=2,
    )

    paired = paired_mixed + spectral_residual + local_residual
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    interleaved = (paired * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dct_phase_cycle_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """DHP-9: five DCT shells plus four directed polyphase local edges.

    The nine registered complex slots are used once as a single dictionary:
    ``00,10,11,20,21`` control the five DCT shells and ``01,02,12,22``
    control the four phase-cycle edges ``01<-00, 11<-01, 10<-11, 00<-10``.
    The phase branch has a fixed one-half scale; it is not a second q9
    operator and adds no parameter, buffer, head, or routing coordinate.
    """

    d = mix_m_real.shape[0]
    mixed = _q9_widely_linear_pointwise(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
    )
    batch, _, height, width = mixed.shape
    if height < 3 or width < 3:
        raise ValueError("dct_phase_cycle requires both spatial dimensions >= 3")
    paired_mixed = mixed.view(batch, d, 2, height, width)
    spectrum = _dct_ii_ortho_2d(paired_mixed)
    spectrum_real = spectrum[:, :, 0]
    spectrum_imag = spectrum[:, :, 1]
    shells = _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectrum_real.dtype,
    )
    phase_actions = _polyphase_local_cycle_actions(paired_mixed)
    conditioning = _dct_phase_cycle_input_conditioning(
        paired_mixed,
        spectrum_real,
        spectrum_imag,
        shells,
        phase_actions,
    )

    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    ).reshape(d, 9)
    identity_slot = F.pad(
        torch.ones_like(gain_real[:, :1]), (0, 8)
    )
    residual_gain_real = (gain_real - identity_slot)[None]
    residual_gain_imag = gain_imag[None]

    shell_slots = (0, 3, 4, 6, 7)
    phase_slots = (1, 2, 5, 8)
    shell_conditioning = conditioning[:, :, :5]
    phase_conditioning = conditioning[:, :, 5:]

    shell_real = shell_conditioning * residual_gain_real[:, :, shell_slots]
    shell_imag = shell_conditioning * residual_gain_imag[:, :, shell_slots]
    multiplier_real = torch.einsum("bds,shw->bdhw", shell_real, shells)
    multiplier_imag = torch.einsum("bds,shw->bdhw", shell_imag, shells)
    residual_spectrum_real = (
        multiplier_real * spectrum_real - multiplier_imag * spectrum_imag
    )
    residual_spectrum_imag = (
        multiplier_imag * spectrum_real + multiplier_real * spectrum_imag
    )
    spectral_residual = _idct_iii_ortho_2d(
        torch.stack((residual_spectrum_real, residual_spectrum_imag), dim=2),
        output_dtype=paired_mixed.dtype,
    )

    phase_gain_real = (
        0.5 * phase_conditioning * residual_gain_real[:, :, phase_slots]
    )
    phase_gain_imag = (
        0.5 * phase_conditioning * residual_gain_imag[:, :, phase_slots]
    )
    phase_real = phase_actions[:, :, 0]
    phase_imag = phase_actions[:, :, 1]
    phase_residual_real = (
        phase_gain_real[..., None, None, None] * phase_real
        - phase_gain_imag[..., None, None, None] * phase_imag
    ).sum(dim=2)
    phase_residual_imag = (
        phase_gain_imag[..., None, None, None] * phase_real
        + phase_gain_real[..., None, None, None] * phase_imag
    ).sum(dim=2)
    phase_residual = torch.stack(
        (phase_residual_real, phase_residual_imag), dim=2
    )
    padded_height = height + height % 2
    padded_width = width + width % 2
    phase_residual = F.pixel_shuffle(
        phase_residual.reshape(
            batch * d * 2,
            4,
            padded_height // 2,
            padded_width // 2,
        ),
        2,
    ).reshape(batch, d, 2, padded_height, padded_width)[
        ..., :height, :width
    ]

    paired = paired_mixed + spectral_residual + phase_residual
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    interleaved = (paired * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    return F.conv2d(
        interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )


def _fused_q9_dct_dsl8_operator_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
) -> Tensor:
    """DSL-8 in grouped-H coordinates, before the tap-major reorder.

    For every state, grouped row zero (the physical center coordinate) reads
    the complete DCT-plus-radial latent.  Grouped rows one through eight take
    the shorter pointwise path.  ``ComplexF9Q9Compiler`` then applies the same
    explicit ``[state,tap] -> [tap,state]`` conversion as every new q9 mode.
    """

    d = mix_m_real.shape[0]
    mixed = _q9_widely_linear_pointwise(
        packed_f,
        mix_m_real,
        mix_m_imag,
        mix_n_real,
        mix_n_imag,
    )
    batch, _, height, width = mixed.shape
    if height < 3 or width < 3:
        raise ValueError("dct_dsl8 requires both spatial dimensions to be >= 3")
    paired_mixed = mixed.view(batch, d, 2, height, width)
    mixed_real = paired_mixed[:, :, 0]
    mixed_imag = paired_mixed[:, :, 1]

    mode = torch.arange(3, device=packed_f.device, dtype=packed_f.dtype)
    y = torch.arange(height, device=packed_f.device, dtype=packed_f.dtype)
    x = torch.arange(width, device=packed_f.device, dtype=packed_f.dtype)
    scale = torch.sqrt(
        torch.where(mode == 0, torch.ones_like(mode), 2.0 * torch.ones_like(mode))
    )
    basis_y = (
        scale[:, None]
        * torch.cos(
            torch.pi * mode[:, None] * (y[None, :] + 0.5) / float(height)
        )
        / float(height) ** 0.5
    )
    basis_x = (
        scale[:, None]
        * torch.cos(
            torch.pi * mode[:, None] * (x[None, :] + 0.5) / float(width)
        )
        / float(width) ** 0.5
    )
    gain_real = torch.roll(
        depthwise_real[:, 0], shifts=(-1, -1), dims=(-2, -1)
    )
    gain_imag = torch.roll(
        depthwise_imag[:, 0], shifts=(-1, -1), dims=(-2, -1)
    )
    gain_real = gain_real - F.pad(
        torch.ones_like(gain_real[:, :1, :1]), (0, 2, 0, 2)
    )
    coefficient_real = torch.einsum(
        "bdhw,mh,nw->bdmn", mixed_real, basis_y, basis_x
    )
    coefficient_imag = torch.einsum(
        "bdhw,mh,nw->bdmn", mixed_imag, basis_y, basis_x
    )
    filtered_real = (
        gain_real[None] * coefficient_real
        - gain_imag[None] * coefficient_imag
    )
    filtered_imag = (
        gain_imag[None] * coefficient_real
        + gain_real[None] * coefficient_imag
    )
    integral_real = torch.einsum(
        "bdmn,mh,nw->bdhw", filtered_real, basis_y, basis_x
    )
    integral_imag = torch.einsum(
        "bdmn,mh,nw->bdhw", filtered_imag, basis_y, basis_x
    )
    full = torch.stack(
        (mixed_real + integral_real, mixed_imag + integral_imag), dim=2
    )
    magnitude = torch.sqrt(
        full.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    radial_gain = F.silu(magnitude) / magnitude
    full_interleaved = (full * radial_gain.unsqueeze(2)).reshape(
        batch, 2 * d, height, width
    )
    raw_full = F.conv2d(
        full_interleaved,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )
    raw_short = F.conv2d(
        mixed,
        head_weight,
        head_bias,
        padding=1,
        groups=d,
    )
    full_state_major = raw_full.view(batch, d, 9, height, width)
    short_state_major = raw_short.view(batch, d, 9, height, width)
    return torch.cat(
        (full_state_major[:, :, :1], short_state_major[:, :, 1:]),
        dim=2,
    ).reshape(batch, 9 * d, height, width)


_COMPILED_FUSED_Q9_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_PHASE_GAUGE_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_COSINE_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_GREEN_RESOLVENT_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_PARSEVAL_RESOLVENT_DIRECTIONAL_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_GCHAF_R1_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_GCHAF_AR9F_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_GCHAF_CCAR9_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_GLOBAL9_CONDITIONED_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_CSEB_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_PDR_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_LMS4_ADDITIVE_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_LMS4_SIGNED_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_LMS4_ACT4_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_LMS4_DUAL_ROUTE_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_MWF9_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_MWF9_INDEPENDENT_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_SWF9_INDEPENDENT_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_SWF9_SCALE_TIED_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DH_SWF9_SCALE_RANK2_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_LEARNED_ROUTER_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_GLRC9_CCP_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_GLRC9_CLUSTERED_H_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_GLRC9_ROLE_FILM_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_ROUTING_CONTINUATION_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_LOCAL_FAMILY_CONTINUATION_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_MN_OFFDIAG_CONTINUATION_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_MN_OFFDIAG_DIAGONAL_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_MN_OFFDIAG_REFERENCE_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_NO_LOCAL_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_HYBRID_HLR_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}
_COMPILED_FUSED_Q9_DCT_HYBRID_H2_OPERATOR_CACHE: dict[
    tuple[object, ...], Callable[..., Tensor]
] = {}
_COMPILED_FUSED_Q9_DCT_HYBRID_H2_GIVENS_OPERATOR_CACHE: dict[
    tuple[object, ...], Callable[..., Tensor]
] = {}

_COMPILED_FUSED_Q9_DCT_PHASE_CYCLE_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_DCT_DSL8_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}

_COMPILED_FUSED_Q9_POLYPHASE_LIFTING_OPERATOR_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., Tensor], Callable[..., Tensor]],
] = {}


def _fused_q9_compiler_vjp_graph(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    depthwise_real: Tensor,
    depthwise_imag: Tensor,
    head_weight: Tensor,
    head_bias: Tensor,
    grad_output: Tensor,
) -> tuple[Tensor, ...]:
    """Analytic VJP for the fused q9 compiler graph.

    The forward compiler is a pointwise widely-linear map, a complex
    depthwise convolution, radial SiLU, and a grouped real readout.  The
    default autograd path retains every construction op used to assemble the
    real convolution weights.  This graph recomputes the two latent tensors
    once and uses convolution VJPs directly, so the q9 compiler contributes a
    compact, compilable reverse graph instead of a long saved autograd chain.
    """

    d = mix_m_real.shape[0]
    real_rows = torch.cat(
        (mix_m_real + mix_n_real, -mix_m_imag + mix_n_imag),
        dim=1,
    )
    imag_rows = torch.cat(
        (mix_m_imag + mix_n_imag, mix_m_real - mix_n_real),
        dim=1,
    )
    pointwise_weight = torch.stack(
        (real_rows, imag_rows),
        dim=1,
    ).reshape(2 * d, 2 * d, 1, 1)
    pointwise_output = F.conv2d(packed_f, pointwise_weight)

    depthwise_wr = depthwise_real[:, 0]
    depthwise_wi = depthwise_imag[:, 0]
    depthwise_real_rows = torch.stack(
        (depthwise_wr, -depthwise_wi),
        dim=1,
    )
    depthwise_imag_rows = torch.stack(
        (depthwise_wi, depthwise_wr),
        dim=1,
    )
    depthwise_weight = torch.stack(
        (depthwise_real_rows, depthwise_imag_rows),
        dim=1,
    ).reshape(2 * d, 2, 3, 3)
    depthwise_output = F.conv2d(
        pointwise_output,
        depthwise_weight,
        padding=1,
        groups=d,
    )

    batch, _, height, width = depthwise_output.shape
    paired = depthwise_output.view(batch, d, 2, height, width)
    magnitude = torch.sqrt(
        paired.square().sum(dim=2) + torch.finfo(packed_f.dtype).eps
    )
    silu = F.silu(magnitude)
    gain = silu / magnitude
    latent = (paired * gain.unsqueeze(2)).reshape(
        batch,
        2 * d,
        height,
        width,
    )

    head_padding = (head_weight.shape[-1] - 1) // 2
    grad_head_weight = torch.nn.grad.conv2d_weight(
        latent,
        head_weight.shape,
        grad_output,
        padding=head_padding,
        groups=d,
    )
    grad_head_bias = grad_output.sum(dim=(0, 2, 3))
    grad_latent = F.conv_transpose2d(
        grad_output,
        head_weight,
        padding=head_padding,
        groups=d,
    )
    grad_latent = grad_latent.view(batch, d, 2, height, width)

    # v = u * silu(||u||) / ||u||, with the same epsilon as the forward.
    # The following is the exact radial Jacobian-vector product in R^2.
    sigmoid = torch.sigmoid(magnitude)
    silu_prime = sigmoid + magnitude * sigmoid * (1.0 - sigmoid)
    gain_prime = (silu_prime * magnitude - silu) / magnitude.square()
    radial_dot = (grad_latent * paired).sum(dim=2, keepdim=True)
    grad_paired = (
        grad_latent * gain.unsqueeze(2)
        + paired * (radial_dot * (gain_prime / magnitude).unsqueeze(2))
    )
    grad_depthwise_output = grad_paired.reshape(
        batch,
        2 * d,
        height,
        width,
    )

    grad_depthwise_weight = torch.nn.grad.conv2d_weight(
        pointwise_output,
        depthwise_weight.shape,
        grad_depthwise_output,
        padding=1,
        groups=d,
    )
    grad_pointwise_output = F.conv_transpose2d(
        grad_depthwise_output,
        depthwise_weight,
        padding=1,
        groups=d,
    )
    grad_pointwise_weight = torch.nn.grad.conv2d_weight(
        packed_f,
        pointwise_weight.shape,
        grad_pointwise_output,
    )
    grad_packed_f = F.conv_transpose2d(
        grad_pointwise_output,
        pointwise_weight,
    )

    pointwise_weight_grad_rows = grad_pointwise_weight[:, :, 0, 0].view(
        d,
        2,
        2 * d,
    )
    grad_real_rows = pointwise_weight_grad_rows[:, 0]
    grad_imag_rows = pointwise_weight_grad_rows[:, 1]
    grad_m_real = grad_real_rows[:, :d] + grad_imag_rows[:, d:]
    grad_m_imag = -grad_real_rows[:, d:] + grad_imag_rows[:, :d]
    grad_n_real = grad_real_rows[:, :d] - grad_imag_rows[:, d:]
    grad_n_imag = grad_real_rows[:, d:] + grad_imag_rows[:, :d]

    grad_depthwise_real = (
        grad_depthwise_weight[0::2, 0]
        + grad_depthwise_weight[1::2, 1]
    ).unsqueeze(1)
    grad_depthwise_imag = (
        -grad_depthwise_weight[0::2, 1]
        + grad_depthwise_weight[1::2, 0]
    ).unsqueeze(1)
    return (
        grad_packed_f,
        grad_m_real,
        grad_m_imag,
        grad_n_real,
        grad_n_imag,
        grad_depthwise_real,
        grad_depthwise_imag,
        grad_head_weight,
        grad_head_bias,
    )


_COMPILED_FUSED_Q9_VJP_CACHE: dict[
    tuple[Any, ...],
    tuple[Callable[..., tuple[Tensor, ...]], Callable[..., tuple[Tensor, ...]]],
] = {}


_COMPILED_Q9_MOMENT_EXPONENTIAL_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Tensor], Callable[..., Tensor]]
] = {}
_COMPILED_Q9_LOGPOLAR_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Tensor], Callable[..., Tensor]]
] = {}
_COMPILED_Q9_MASS_SIMPLEX_PHASE_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Tensor], Callable[..., Tensor]]
] = {}
_COMPILED_Q9_CENTER_MASS_INVERSE_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Tensor], Callable[..., Tensor]]
] = {}


def _compiled_fused_q9_compiler(tensors: tuple[Tensor, ...]) -> Tensor:
    """Cache a static-signature compiled q9 compiler graph."""

    key = _tensor_static_signatures(tensors)
    entry = _COMPILED_FUSED_Q9_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_FUSED_Q9_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_fused_q9_compiler_graph__signature_{ordinal}",
        }
        if hasattr(_fused_q9_compiler_graph.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _fused_q9_compiler_graph.__code__.replace(**code_fields),
            _fused_q9_compiler_graph.__globals__,
            code_fields["co_name"],
            argdefs=_fused_q9_compiler_graph.__defaults__,
            closure=_fused_q9_compiler_graph.__closure__,
        )
        target.__kwdefaults__ = _fused_q9_compiler_graph.__kwdefaults__
        target.__annotations__ = _fused_q9_compiler_graph.__annotations__
        target.__doc__ = _fused_q9_compiler_graph.__doc__
        target.__module__ = _fused_q9_compiler_graph.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_FUSED_Q9_CACHE[key] = entry
    return entry[0](*tensors)


def _compiled_q9_moment_exponential_raw(
    coordinate_major: Tensor,
    rho_max: float,
) -> Tensor:
    """Cache the candidate chart graph by static tensor signature and rho."""

    key = _tensor_static_signatures((coordinate_major,)) + (float(rho_max),)
    entry = _COMPILED_Q9_MOMENT_EXPONENTIAL_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_Q9_MOMENT_EXPONENTIAL_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_q9_moment_exponential_raw__signature_{ordinal}",
        }
        if hasattr(_q9_moment_exponential_raw.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _q9_moment_exponential_raw.__code__.replace(**code_fields),
            _q9_moment_exponential_raw.__globals__,
            code_fields["co_name"],
            argdefs=_q9_moment_exponential_raw.__defaults__,
            closure=_q9_moment_exponential_raw.__closure__,
        )
        target.__kwdefaults__ = _q9_moment_exponential_raw.__kwdefaults__
        target.__annotations__ = _q9_moment_exponential_raw.__annotations__
        target.__doc__ = _q9_moment_exponential_raw.__doc__
        target.__module__ = _q9_moment_exponential_raw.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_Q9_MOMENT_EXPONENTIAL_CACHE[key] = entry
    return entry[0](coordinate_major, float(rho_max))


def _compiled_q9_logpolar_raw(coordinate_major: Tensor) -> Tensor:
    """Cache the LP-9 raw chart by its static tensor signature."""

    key = _tensor_static_signatures((coordinate_major,))
    entry = _COMPILED_Q9_LOGPOLAR_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_Q9_LOGPOLAR_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_q9_logpolar_raw__signature_{ordinal}",
        }
        if hasattr(_q9_logpolar_raw.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _q9_logpolar_raw.__code__.replace(**code_fields),
            _q9_logpolar_raw.__globals__,
            code_fields["co_name"],
            argdefs=_q9_logpolar_raw.__defaults__,
            closure=_q9_logpolar_raw.__closure__,
        )
        target.__kwdefaults__ = _q9_logpolar_raw.__kwdefaults__
        target.__annotations__ = _q9_logpolar_raw.__annotations__
        target.__doc__ = _q9_logpolar_raw.__doc__
        target.__module__ = _q9_logpolar_raw.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_Q9_LOGPOLAR_CACHE[key] = entry
    return entry[0](coordinate_major)


def _compiled_q9_mass_simplex_phase_raw(coordinate_major: Tensor) -> Tensor:
    """Cache the AP-S9 raw chart by its static tensor signature."""

    key = _tensor_static_signatures((coordinate_major,))
    entry = _COMPILED_Q9_MASS_SIMPLEX_PHASE_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_Q9_MASS_SIMPLEX_PHASE_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_q9_mass_simplex_phase_raw__signature_{ordinal}",
        }
        if hasattr(_q9_mass_simplex_phase_raw.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _q9_mass_simplex_phase_raw.__code__.replace(**code_fields),
            _q9_mass_simplex_phase_raw.__globals__,
            code_fields["co_name"],
            argdefs=_q9_mass_simplex_phase_raw.__defaults__,
            closure=_q9_mass_simplex_phase_raw.__closure__,
        )
        target.__kwdefaults__ = _q9_mass_simplex_phase_raw.__kwdefaults__
        target.__annotations__ = _q9_mass_simplex_phase_raw.__annotations__
        target.__doc__ = _q9_mass_simplex_phase_raw.__doc__
        target.__module__ = _q9_mass_simplex_phase_raw.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_Q9_MASS_SIMPLEX_PHASE_CACHE[key] = entry
    return entry[0](coordinate_major)


def _compiled_q9_center_mass_inverse_raw(
    coordinate_major: Tensor,
    rho_max: float,
) -> Tensor:
    """Cache the CM-I9 chart graph by static tensor signature and rho."""

    key = _tensor_static_signatures((coordinate_major,)) + (float(rho_max),)
    entry = _COMPILED_Q9_CENTER_MASS_INVERSE_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_Q9_CENTER_MASS_INVERSE_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_q9_center_mass_inverse_raw__signature_{ordinal}",
        }
        if hasattr(_q9_center_mass_inverse_raw.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _q9_center_mass_inverse_raw.__code__.replace(**code_fields),
            _q9_center_mass_inverse_raw.__globals__,
            code_fields["co_name"],
            argdefs=_q9_center_mass_inverse_raw.__defaults__,
            closure=_q9_center_mass_inverse_raw.__closure__,
        )
        target.__kwdefaults__ = _q9_center_mass_inverse_raw.__kwdefaults__
        target.__annotations__ = _q9_center_mass_inverse_raw.__annotations__
        target.__doc__ = _q9_center_mass_inverse_raw.__doc__
        target.__module__ = _q9_center_mass_inverse_raw.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_Q9_CENTER_MASS_INVERSE_CACHE[key] = entry
    return entry[0](coordinate_major, float(rho_max))


def _compiled_fused_q9_phase_gauge_compiler(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled phase-gauge q9 graph."""

    key = _tensor_static_signatures(tensors)
    entry = _COMPILED_FUSED_Q9_PHASE_GAUGE_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_FUSED_Q9_PHASE_GAUGE_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_fused_q9_phase_gauge_compiler_graph__signature_{ordinal}",
        }
        if hasattr(_fused_q9_phase_gauge_compiler_graph.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _fused_q9_phase_gauge_compiler_graph.__code__.replace(**code_fields),
            _fused_q9_phase_gauge_compiler_graph.__globals__,
            code_fields["co_name"],
            argdefs=_fused_q9_phase_gauge_compiler_graph.__defaults__,
            closure=_fused_q9_phase_gauge_compiler_graph.__closure__,
        )
        target.__kwdefaults__ = _fused_q9_phase_gauge_compiler_graph.__kwdefaults__
        target.__annotations__ = _fused_q9_phase_gauge_compiler_graph.__annotations__
        target.__doc__ = _fused_q9_phase_gauge_compiler_graph.__doc__
        target.__module__ = _fused_q9_phase_gauge_compiler_graph.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_FUSED_Q9_PHASE_GAUGE_CACHE[key] = entry
    return entry[0](*tensors)


def _compiled_fused_q9_cosine_operator(tensors: tuple[Tensor, ...]) -> Tensor:
    """Cache a static-signature compiled cosine-integral q9 graph."""

    key = _tensor_static_signatures(tensors)
    entry = _COMPILED_FUSED_Q9_COSINE_OPERATOR_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_FUSED_Q9_COSINE_OPERATOR_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_fused_q9_cosine_operator_graph__signature_{ordinal}",
        }
        if hasattr(_fused_q9_cosine_operator_graph.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _fused_q9_cosine_operator_graph.__code__.replace(**code_fields),
            _fused_q9_cosine_operator_graph.__globals__,
            code_fields["co_name"],
            argdefs=_fused_q9_cosine_operator_graph.__defaults__,
            closure=_fused_q9_cosine_operator_graph.__closure__,
        )
        target.__kwdefaults__ = _fused_q9_cosine_operator_graph.__kwdefaults__
        target.__annotations__ = _fused_q9_cosine_operator_graph.__annotations__
        target.__doc__ = _fused_q9_cosine_operator_graph.__doc__
        target.__module__ = _fused_q9_cosine_operator_graph.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_FUSED_Q9_COSINE_OPERATOR_CACHE[key] = entry
    return entry[0](*tensors)


def _compiled_fused_q9_green_resolvent_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled exact-autograd Green q9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    # Populate runtime-only fixed tensors before Dynamo traces the pure graph.
    # The traced lookups therefore resolve to constants without adding model
    # buffers, parameters, or checkpoint keys.
    _green_resolvent_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )

    key = _tensor_static_signatures(tensors)
    entry = _COMPILED_FUSED_Q9_GREEN_RESOLVENT_OPERATOR_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_FUSED_Q9_GREEN_RESOLVENT_OPERATOR_CACHE)
        code_fields: dict[str, str] = {
            "co_name": (
                "_fused_q9_green_resolvent_operator_graph"
                f"__signature_{ordinal}"
            ),
        }
        if hasattr(
            _fused_q9_green_resolvent_operator_graph.__code__, "co_qualname"
        ):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _fused_q9_green_resolvent_operator_graph.__code__.replace(
                **code_fields
            ),
            _fused_q9_green_resolvent_operator_graph.__globals__,
            code_fields["co_name"],
            argdefs=_fused_q9_green_resolvent_operator_graph.__defaults__,
            closure=_fused_q9_green_resolvent_operator_graph.__closure__,
        )
        target.__kwdefaults__ = (
            _fused_q9_green_resolvent_operator_graph.__kwdefaults__
        )
        target.__annotations__ = (
            _fused_q9_green_resolvent_operator_graph.__annotations__
        )
        target.__doc__ = _fused_q9_green_resolvent_operator_graph.__doc__
        target.__module__ = _fused_q9_green_resolvent_operator_graph.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_FUSED_Q9_GREEN_RESOLVENT_OPERATOR_CACHE[key] = entry
    return entry[0](*tensors)


def _compiled_fused_q9_parseval_resolvent_directional_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled Parseval directional q9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _parseval_resolvent_directional_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )

    key = _tensor_static_signatures(tensors)
    entry = _COMPILED_FUSED_Q9_PARSEVAL_RESOLVENT_DIRECTIONAL_OPERATOR_CACHE.get(key)
    if entry is None:
        ordinal = len(
            _COMPILED_FUSED_Q9_PARSEVAL_RESOLVENT_DIRECTIONAL_OPERATOR_CACHE
        )
        code_fields: dict[str, str] = {
            "co_name": (
                "_fused_q9_parseval_resolvent_directional_operator_graph"
                f"__signature_{ordinal}"
            ),
        }
        if hasattr(
            _fused_q9_parseval_resolvent_directional_operator_graph.__code__,
            "co_qualname",
        ):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _fused_q9_parseval_resolvent_directional_operator_graph.__code__.replace(
                **code_fields
            ),
            _fused_q9_parseval_resolvent_directional_operator_graph.__globals__,
            code_fields["co_name"],
            argdefs=(
                _fused_q9_parseval_resolvent_directional_operator_graph.__defaults__
            ),
            closure=(
                _fused_q9_parseval_resolvent_directional_operator_graph.__closure__
            ),
        )
        target.__kwdefaults__ = (
            _fused_q9_parseval_resolvent_directional_operator_graph.__kwdefaults__
        )
        target.__annotations__ = (
            _fused_q9_parseval_resolvent_directional_operator_graph.__annotations__
        )
        target.__doc__ = (
            _fused_q9_parseval_resolvent_directional_operator_graph.__doc__
        )
        target.__module__ = (
            _fused_q9_parseval_resolvent_directional_operator_graph.__module__
        )
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_FUSED_Q9_PARSEVAL_RESOLVENT_DIRECTIONAL_OPERATOR_CACHE[key] = (
            entry
        )
    return entry[0](*tensors)


def _compile_static_q9_operator_graph(
    tensors: tuple[Tensor, ...],
    *,
    graph: Callable[..., Tensor],
    cache: dict[
        tuple[Any, ...],
        tuple[Callable[..., Tensor], Callable[..., Tensor]],
    ],
    name: str,
) -> Tensor:
    """Compile one exact-autograd q9 operator graph per static signature."""

    key = _tensor_static_signatures(tensors)
    entry = cache.get(key)
    if entry is None:
        ordinal = len(cache)
        code_name = f"{name}__signature_{ordinal}"
        code_fields: dict[str, str] = {"co_name": code_name}
        if hasattr(graph.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_name
        target = types.FunctionType(
            graph.__code__.replace(**code_fields),
            graph.__globals__,
            code_name,
            argdefs=graph.__defaults__,
            closure=graph.__closure__,
        )
        target.__kwdefaults__ = graph.__kwdefaults__
        target.__annotations__ = graph.__annotations__
        target.__doc__ = graph.__doc__
        target.__module__ = graph.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        cache[key] = entry
    return entry[0](*tensors)


def _compiled_fused_q9_dct_hybrid_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_routing_continuation_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache the exact GLRC9 routing-continuation graph per tensor signature."""

    if len(tensors) != 10:
        raise ValueError(
            "GLRC9 routing continuation expects nine compiler tensors and scalar g"
        )
    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_ROUTING_CONTINUATION_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_routing_continuation_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_local_family_continuation_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache the RC2 interior graph per static tensor signature."""

    if len(tensors) != 10:
        raise ValueError(
            "GLRC9 local-family continuation expects nine compiler tensors and scalar g"
        )
    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(height, width, device=packed_f.device, dtype=spectral_dtype)
    _dct_hybrid_local_frame(device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_local_family_continuation_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_LOCAL_FAMILY_CONTINUATION_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_local_family_continuation_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_mn_offdiag_continuation_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache the RC3 M/N continuation graph per static tensor signature."""

    if len(tensors) != 10:
        raise ValueError(
            "GLRC9 M/N continuation expects nine compiler tensors and scalar g"
        )
    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(height, width, device=packed_f.device, dtype=spectral_dtype)
    _dct_hybrid_local_frame(device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_mn_offdiag_continuation_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_MN_OFFDIAG_CONTINUATION_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_mn_offdiag_continuation_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_mn_offdiag_diagonal_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache the RC3 g=0 diagonal-only graph per static signature."""

    if len(tensors) != 9:
        raise ValueError(
            "GLRC9 M/N diagonal endpoint expects nine compiler tensors"
        )
    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(height, width, device=packed_f.device, dtype=spectral_dtype)
    _dct_hybrid_local_frame(device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_mn_offdiag_diagonal_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_MN_OFFDIAG_DIAGONAL_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_mn_offdiag_diagonal_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_mn_offdiag_materialized_reference_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Compile the independent masked reference graph per static signature."""

    if len(tensors) != 9:
        raise ValueError(
            "GLRC9 M/N materialized reference expects nine compiler tensors"
        )
    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(height, width, device=packed_f.device, dtype=spectral_dtype)
    _dct_hybrid_local_frame(device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_mn_offdiag_materialized_reference_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_MN_OFFDIAG_REFERENCE_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_mn_offdiag_materialized_reference_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_no_local_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache the independent RC2 g=0 no-local graph."""

    if len(tensors) != 9:
        raise ValueError("GLRC9 no-local endpoint expects nine compiler tensors")
    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(height, width, device=packed_f.device, dtype=spectral_dtype)
    _dct_hybrid_local_frame(device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_no_local_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_NO_LOCAL_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_no_local_operator_graph",
    )


def _compiled_fused_q9_gchaf_r1_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled GCHAF-R1 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_gchaf_r1_operator_graph,
        cache=_COMPILED_FUSED_Q9_GCHAF_R1_OPERATOR_CACHE,
        name="_fused_q9_gchaf_r1_operator_graph",
    )


def _compiled_fused_q9_gchaf_ar9f_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled GCHAF-AR9F graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_gchaf_ar9f_operator_graph,
        cache=_COMPILED_FUSED_Q9_GCHAF_AR9F_OPERATOR_CACHE,
        name="_fused_q9_gchaf_ar9f_operator_graph",
    )


def _compiled_fused_q9_gchaf_ccar9_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled GCHAF-CCAR9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    gchaf_ccar9_helmert(device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(height, device=packed_f.device, dtype=spectral_dtype)
    _dct_ortho_factors(width, device=packed_f.device, dtype=spectral_dtype)
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_gchaf_ccar9_operator_graph,
        cache=_COMPILED_FUSED_Q9_GCHAF_CCAR9_OPERATOR_CACHE,
        name="_fused_q9_gchaf_ccar9_operator_graph",
    )


def _compiled_fused_q9_dct_global9_conditioned_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache the exact compiled D-arm global-only graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_global9_coordinate_masks(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_global9_conditioned_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_GLOBAL9_CONDITIONED_OPERATOR_CACHE,
        name="_fused_q9_dct_global9_conditioned_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_cseb_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-9 CSEB graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_cseb_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_CSEB_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_cseb_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_pdr_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-PDR graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height, width, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_hybrid_local_frame(
        device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_pdr_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_PDR_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_pdr_operator_graph",
    )


def _compiled_fused_q9_dh_lms4_additive_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-LMS4-A graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dh_lms4_additive_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_lms4_additive_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_LMS4_ADDITIVE_OPERATOR_CACHE,
        name="_fused_q9_dh_lms4_additive_operator_graph",
    )


def _compiled_fused_q9_dh_lms4_signed_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled signed DH-LMS4 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dh_lms4_additive_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_lms4_signed_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_LMS4_SIGNED_OPERATOR_CACHE,
        name="_fused_q9_dh_lms4_signed_operator_graph",
    )


def _compiled_fused_q9_dh_lms4_act4_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-LMS4-ACT4 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dh_lms4_additive_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dh_lms4_act4_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_lms4_act4_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_LMS4_ACT4_OPERATOR_CACHE,
        name="_fused_q9_dh_lms4_act4_operator_graph",
    )


def _compiled_fused_q9_dh_lms4_dual_route_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-LMS4 dual-route graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dh_lms4_additive_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dh_lms4_act4_bands(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_lms4_dual_route_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_LMS4_DUAL_ROUTE_OPERATOR_CACHE,
        name="_fused_q9_dh_lms4_dual_route_operator_graph",
    )


def _compiled_fused_q9_dh_mwf9_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-MWF9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_mwf9_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_MWF9_OPERATOR_CACHE,
        name="_fused_q9_dh_mwf9_operator_graph",
    )


def _compiled_fused_q9_dh_mwf9_independent_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-MWF9-I graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_mwf9_independent_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_MWF9_INDEPENDENT_OPERATOR_CACHE,
        name="_fused_q9_dh_mwf9_independent_operator_graph",
    )


def _compiled_fused_q9_dh_swf9_independent_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-SWF9-I graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_swf9_independent_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_SWF9_INDEPENDENT_OPERATOR_CACHE,
        name="_fused_q9_dh_swf9_independent_operator_graph",
    )


def _compiled_fused_q9_dh_swf9_scale_tied_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled scale-tied DH-SWF9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_swf9_scale_tied_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_SWF9_SCALE_TIED_OPERATOR_CACHE,
        name="_fused_q9_dh_swf9_scale_tied_operator_graph",
    )


def _compiled_fused_q9_dh_swf9_scale_rank2_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled scale-rank2 DH-SWF9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dh_swf9_scale_rank2_operator_graph,
        cache=_COMPILED_FUSED_Q9_DH_SWF9_SCALE_RANK2_OPERATOR_CACHE,
        name="_fused_q9_dh_swf9_scale_rank2_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_learned_router_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-MSR-9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_learned_router_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_LEARNED_ROUTER_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_learned_router_operator_graph",
    )


def _compiled_fused_q9_glrc9_ccp_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled GLRC9-CCP graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_glrc9_ccp_operator_graph,
        cache=_COMPILED_FUSED_Q9_GLRC9_CCP_OPERATOR_CACHE,
        name="_fused_q9_glrc9_ccp_operator_graph",
    )


def _compiled_fused_q9_glrc9_clustered_h_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled clustered-H GLRC9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_glrc9_clustered_h_operator_graph,
        cache=_COMPILED_FUSED_Q9_GLRC9_CLUSTERED_H_OPERATOR_CACHE,
        name="_fused_q9_glrc9_clustered_h_operator_graph",
    )


def _compiled_fused_q9_glrc9_role_film_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature GLRC9 role-FiLM graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_glrc9_role_film_operator_graph,
        cache=_COMPILED_FUSED_Q9_GLRC9_ROLE_FILM_OPERATOR_CACHE,
        name="_fused_q9_glrc9_role_film_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_hlr_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled all-block DH-HLR graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_hybrid_local_frame(
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_hlr_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_HLR_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_hlr_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_h2_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-H2 latent decoder graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height, width, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_hybrid_local_frame(
        device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_h2_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_H2_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_h2_operator_graph",
    )


def _compiled_fused_q9_dct_hybrid_h2_givens_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DH-H2-Givens decoder graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height, width, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_hybrid_local_frame(
        device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_hybrid_h2_givens_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_HYBRID_H2_GIVENS_OPERATOR_CACHE,
        name="_fused_q9_dct_hybrid_h2_givens_operator_graph",
    )


def _compiled_fused_q9_dct_phase_cycle_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DHP-9 graph."""

    packed_f = tensors[0]
    height, width = packed_f.shape[-2:]
    spectral_dtype = (
        torch.float32
        if packed_f.dtype in {torch.float16, torch.bfloat16}
        else packed_f.dtype
    )
    _dct_hybrid_shells(
        height,
        width,
        device=packed_f.device,
        dtype=spectral_dtype,
    )
    _dct_ortho_factors(
        height, device=packed_f.device, dtype=spectral_dtype
    )
    _dct_ortho_factors(
        width, device=packed_f.device, dtype=spectral_dtype
    )
    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_phase_cycle_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_PHASE_CYCLE_OPERATOR_CACHE,
        name="_fused_q9_dct_phase_cycle_operator_graph",
    )


def _compiled_fused_q9_dct_dsl8_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled DSL-8 graph."""

    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_dct_dsl8_operator_graph,
        cache=_COMPILED_FUSED_Q9_DCT_DSL8_OPERATOR_CACHE,
        name="_fused_q9_dct_dsl8_operator_graph",
    )


def _compiled_fused_q9_polyphase_lifting_operator(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    """Cache a static-signature compiled PL-9 graph."""

    return _compile_static_q9_operator_graph(
        tensors,
        graph=_fused_q9_polyphase_lifting_operator_graph,
        cache=_COMPILED_FUSED_Q9_POLYPHASE_LIFTING_OPERATOR_CACHE,
        name="_fused_q9_polyphase_lifting_operator_graph",
    )


def _tensor_static_signatures(
    tensors: tuple[Tensor, ...],
) -> tuple[Any, ...]:
    storage_groups: dict[tuple[str, int | None, int], int] = {}
    signatures = []
    for value in tensors:
        storage = value.untyped_storage()
        storage_token = (
            value.device.type,
            value.device.index,
            int(storage.data_ptr()),
        )
        alias_group = storage_groups.setdefault(
            storage_token,
            len(storage_groups),
        )
        signatures.append((
            value.device.type,
            value.device.index,
            value.dtype,
            value.layout,
            tuple(value.shape),
            tuple(value.stride()),
            int(value.storage_offset()),
            int(storage.nbytes()),
            bool(value.requires_grad),
            bool(value.is_conj()),
            bool(value.is_neg()),
            alias_group,
        ))
    return tuple(signatures)


def _compiled_fused_q9_vjp(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    """Cache a static-signature compiled analytic q9 reverse graph."""

    key = _tensor_static_signatures(tensors)
    entry = _COMPILED_FUSED_Q9_VJP_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_FUSED_Q9_VJP_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_fused_q9_compiler_vjp_graph__signature_{ordinal}",
        }
        if hasattr(_fused_q9_compiler_vjp_graph.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _fused_q9_compiler_vjp_graph.__code__.replace(**code_fields),
            _fused_q9_compiler_vjp_graph.__globals__,
            code_fields["co_name"],
            argdefs=_fused_q9_compiler_vjp_graph.__defaults__,
            closure=_fused_q9_compiler_vjp_graph.__closure__,
        )
        target.__kwdefaults__ = _fused_q9_compiler_vjp_graph.__kwdefaults__
        target.__annotations__ = _fused_q9_compiler_vjp_graph.__annotations__
        target.__doc__ = _fused_q9_compiler_vjp_graph.__doc__
        target.__module__ = _fused_q9_compiler_vjp_graph.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_FUSED_Q9_VJP_CACHE[key] = entry
    return entry[0](*tensors)


class _FusedQ9CompilerFunction(torch.autograd.Function):
    """Custom first-order VJP for the q9 compiler's fused forward graph."""

    @staticmethod
    def forward(ctx: object, *args: object) -> Tensor:
        tensors = args[:-1]
        use_compiled = bool(args[-1])
        if not all(isinstance(value, Tensor) for value in tensors):
            raise TypeError("q9 custom VJP expects tensor inputs")
        ctx.save_for_backward(*tensors)
        ctx.use_compiled = use_compiled
        packed_f = tensors[0]
        if use_compiled and packed_f.is_cuda and packed_f.dtype == torch.float32:
            return _compiled_fused_q9_compiler(tensors)
        return _fused_q9_compiler_graph(*tensors)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, ...]:
        tensors = ctx.saved_tensors
        vjp_tensors = (*tensors, grad_output)
        if ctx.use_compiled and tensors[0].is_cuda and tensors[0].dtype == torch.float32:
            gradients = _compiled_fused_q9_vjp(vjp_tensors)
        else:
            gradients = _fused_q9_compiler_vjp_graph(*vjp_tensors)
        return (*gradients, None)


def _compiled_fused_complex_jet(
    tensors: tuple[Tensor, ...],
) -> Tensor:
    key = _tensor_static_signatures(tensors)
    entry = _COMPILED_FUSED_COMPLEX_JET_CACHE.get(key)
    if entry is None:
        ordinal = len(_COMPILED_FUSED_COMPLEX_JET_CACHE)
        code_fields: dict[str, str] = {
            "co_name": f"_fused_complex_jet_graph__signature_{ordinal}",
        }
        if hasattr(_fused_complex_jet_graph.__code__, "co_qualname"):
            code_fields["co_qualname"] = code_fields["co_name"]
        target = types.FunctionType(
            _fused_complex_jet_graph.__code__.replace(**code_fields),
            _fused_complex_jet_graph.__globals__,
            code_fields["co_name"],
            argdefs=_fused_complex_jet_graph.__defaults__,
            closure=_fused_complex_jet_graph.__closure__,
        )
        target.__kwdefaults__ = _fused_complex_jet_graph.__kwdefaults__
        target.__annotations__ = _fused_complex_jet_graph.__annotations__
        target.__doc__ = _fused_complex_jet_graph.__doc__
        target.__module__ = _fused_complex_jet_graph.__module__
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_FUSED_COMPLEX_JET_CACHE[key] = entry
    return entry[0](*tensors)


class ComplexF9JetCompiler(nn.Module):
    """Compile packed complex states into raw complex F9 coefficient rows.

    The input has shape ``[B, 2D, H, W]`` with all real channels followed by
    all imaginary channels.  The output has shape ``[B, 18D, H, W]`` and is
    laid out so that

    ``raw.view(B, 2, 9, D, H, W)``

    indexes complex part, F9 tap, and state in that order.

    ``widely_linear=True`` uses the general real-linear complex pointwise map
    ``M f + N conjugate(f)``.  Strict mode fixes ``N=0`` and is retained only
    as a mathematical control.
    """

    tap_count = 9

    def __init__(
        self,
        state_channels: int,
        *,
        widely_linear: bool = True,
    ) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        if type(widely_linear) is not bool:
            raise TypeError("widely_linear must be bool")

        self.state_channels = state_channels
        self.widely_linear = widely_linear
        d = state_channels

        # Dense complex cross-state map M.  The optional N parameters complete
        # the unique Wirtinger form of an arbitrary real-linear map on C^D.
        self.mix_m_real = nn.Parameter(torch.empty(d, d))
        self.mix_m_imag = nn.Parameter(torch.empty(d, d))
        if widely_linear:
            self.mix_n_real = nn.Parameter(torch.empty(d, d))
            self.mix_n_imag = nn.Parameter(torch.empty(d, d))
        else:
            self.register_parameter("mix_n_real", None)
            self.register_parameter("mix_n_imag", None)

        # One complex 3x3 spatial jet per state, with no hidden bias.
        self.depthwise_real = nn.Parameter(torch.empty(d, 1, 3, 3))
        self.depthwise_imag = nn.Parameter(torch.empty(d, 1, 3, 3))

        # Each state maps its 3x3 complex jet to nine complex F9 outputs.
        # Grouped-convolution output is state-major and is permuted to the
        # tap-major layout required by the canonical raw projection.
        self.expand_real = nn.Parameter(
            torch.empty(self.tap_count * d, 1, 3, 3)
        )
        self.expand_imag = nn.Parameter(
            torch.empty(self.tap_count * d, 1, 3, 3)
        )

        # This is the only bias in the compiler.
        self.output_bias = nn.Parameter(
            torch.empty(2, self.tap_count, d)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset to a near-canonical F9 field with an identity latent path."""

        d = self.state_channels
        with torch.no_grad():
            self.mix_m_real.copy_(
                torch.eye(
                    d,
                    device=self.mix_m_real.device,
                    dtype=self.mix_m_real.dtype,
                )
            )
            self.mix_m_imag.zero_()
            if self.mix_n_real is not None:
                self.mix_n_real.zero_()
                self.mix_n_imag.zero_()

            self.depthwise_real.zero_()
            self.depthwise_real[:, 0, 1, 1].fill_(1.0)
            self.depthwise_imag.zero_()
            nn.init.normal_(self.expand_real, mean=0.0, std=1.0e-3)
            nn.init.normal_(self.expand_imag, mean=0.0, std=1.0e-3)

            self.output_bias.zero_()
            self.output_bias[0, 0].fill_(1.0)
            self.output_bias[0, 1:].fill_(0.5)

    def _pointwise_mix(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply ``M f + N conjugate(f)`` using one paired-real convolution."""

        mr = self.mix_m_real
        mi = self.mix_m_imag
        if self.mix_n_real is None:
            nr = torch.zeros_like(mr)
            ni = torch.zeros_like(mi)
        else:
            nr = self.mix_n_real
            ni = self.mix_n_imag

        # For f=x+iy:
        # Re(Mf+Nf*) = (Mr+Nr)x + (-Mi+Ni)y
        # Im(Mf+Nf*) = (Mi+Ni)x + (Mr-Nr)y.
        top = torch.cat((mr + nr, -mi + ni), dim=1)
        bottom = torch.cat((mi + ni, mr - nr), dim=1)
        paired_weight = torch.cat((top, bottom), dim=0)[..., None, None]
        mixed = F.conv2d(torch.cat((real, imag), dim=1), paired_weight)
        return mixed.chunk(2, dim=1)

    def _complex_depthwise(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        real_part = (
            F.conv2d(
                real,
                self.depthwise_real,
                padding=1,
                groups=self.state_channels,
            )
            - F.conv2d(
                imag,
                self.depthwise_imag,
                padding=1,
                groups=self.state_channels,
            )
        )
        imag_part = (
            F.conv2d(
                real,
                self.depthwise_imag,
                padding=1,
                groups=self.state_channels,
            )
            + F.conv2d(
                imag,
                self.depthwise_real,
                padding=1,
                groups=self.state_channels,
            )
        )
        return real_part, imag_part

    @staticmethod
    def _radial_silu(
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply a smooth, phase-preserving SiLU to a paired complex tensor."""

        # The dtype epsilon makes the derivative finite at exactly zero while
        # converging to |z| at ordinary magnitudes.
        magnitude = torch.sqrt(
            real.square() + imag.square() + torch.finfo(real.dtype).eps
        )
        gain = F.silu(magnitude) / magnitude
        return real * gain, imag * gain

    def _complex_expand(
        self,
        real: Tensor,
        imag: Tensor,
    ) -> tuple[Tensor, Tensor]:
        real_part = (
            F.conv2d(
                real,
                self.expand_real,
                padding=1,
                groups=self.state_channels,
            )
            - F.conv2d(
                imag,
                self.expand_imag,
                padding=1,
                groups=self.state_channels,
            )
        )
        imag_part = (
            F.conv2d(
                real,
                self.expand_imag,
                padding=1,
                groups=self.state_channels,
            )
            + F.conv2d(
                imag,
                self.expand_real,
                padding=1,
                groups=self.state_channels,
            )
        )
        return real_part, imag_part

    def forward_fused_complex_convolutions(
        self,
        packed_f: Tensor,
        *,
        use_compiled: bool = True,
    ) -> Tensor:
        """Exact Jet algebra with one real kernel per complex grouped convolution.

        This is used only for the consequence recompile. The first/direct Jet
        keeps ``forward`` so a zero consequence blend remains bitwise identical
        to the registered Direct Jet initialization and shared gradients.
        """

        if packed_f.ndim != 4:
            raise ValueError("packed_f must have shape [B,2D,H,W]")
        if packed_f.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "packed_f channel dimension must equal 2 * state_channels"
            )
        if not packed_f.is_floating_point():
            raise TypeError("packed_f must use a real floating dtype")
        if type(use_compiled) is not bool:
            raise TypeError("use_compiled must be bool")

        if self.mix_n_real is None:
            mix_n_real = torch.zeros_like(self.mix_m_real)
            mix_n_imag = torch.zeros_like(self.mix_m_imag)
        else:
            mix_n_real = self.mix_n_real
            mix_n_imag = self.mix_n_imag
        tensors = (
            packed_f,
            self.mix_m_real,
            self.mix_m_imag,
            mix_n_real,
            mix_n_imag,
            self.depthwise_real,
            self.depthwise_imag,
            self.expand_real,
            self.expand_imag,
            self.output_bias,
        )
        if (
            use_compiled
            and packed_f.is_cuda
            and packed_f.dtype == torch.float32
        ):
            return _compiled_fused_complex_jet(tensors)
        return _fused_complex_jet_graph(*tensors)

    def forward(self, packed_f: Tensor) -> Tensor:
        if packed_f.ndim != 4:
            raise ValueError("packed_f must have shape [B,2D,H,W]")
        if packed_f.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "packed_f channel dimension must equal 2 * state_channels"
            )
        if not packed_f.is_floating_point():
            raise TypeError("packed_f must use a real floating dtype")

        real, imag = packed_f.chunk(2, dim=1)
        real, imag = self._pointwise_mix(real, imag)
        real, imag = self._complex_depthwise(real, imag)
        real, imag = self._radial_silu(real, imag)
        real, imag = self._complex_expand(real, imag)

        batch, _, height, width = real.shape
        d = self.state_channels
        real = real.view(batch, d, self.tap_count, height, width).permute(
            0, 2, 1, 3, 4
        )
        imag = imag.view(batch, d, self.tap_count, height, width).permute(
            0, 2, 1, 3, 4
        )
        # Preserve the active autocast dtype.  Adding an FP32 Parameter
        # directly to BF16/FP16 convolution output would silently promote the
        # compiled raw field back to FP32.
        output_bias = self.output_bias.to(dtype=real.dtype)
        real = real + output_bias[0][None, :, :, None, None]
        imag = imag + output_bias[1][None, :, :, None, None]
        return torch.cat(
            (
                real.reshape(
                    batch, self.tap_count * d, height, width
                ),
                imag.reshape(
                    batch, self.tap_count * d, height, width
                ),
            ),
            dim=1,
        )


class ComplexF9Q9Generator(nn.Module):
    """Shared ``G`` trunk for the reciprocal nine-real-coordinate compiler.

    The trunk maps ``[B,2D,H,W]`` packed complex features to another packed
    complex latent with the same state width.  It deliberately keeps the
    dense widely-linear state mixing and the local complex 3x3 jet from the
    direct Jet, so q9 changes the output chart and sharing policy rather than
    silently removing cross-state or local spatial capacity.
    """

    def __init__(
        self,
        state_channels: int,
        *,
        widely_linear: bool = True,
        spatial_operator: str = "local_depthwise",
        gain_chart: str = "direct",
    ) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        if type(widely_linear) is not bool:
            raise TypeError("widely_linear must be bool")
        if gain_chart not in {"direct", "fixed_u9"}:
            raise ValueError("gain_chart must be 'direct' or 'fixed_u9'")
        if spatial_operator not in {
            "local_depthwise",
            "nk9",
            "phase_gauge",
            "cosine_integral",
            "green_resolvent",
            "parseval_resolvent_directional",
            "dct_hybrid",
            "dct_hybrid_routing_continuation",
            "dct_hybrid_local_family_continuation",
            GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR,
            "gchaf_r1",
            "gchaf_ar9f",
            "gchaf_ccar9",
            "dct_global9_conditioned",
            "dh_lms4_additive",
            "dh_lms4_signed",
            "dh_lms4_act4",
            "dh_lms4_dual_route",
            "dh_mwf9",
            "dh_mwf9_independent",
            "dh_swf9_independent",
            "dh_swf9_scale_tied",
            "dh_swf9_scale_rank2",
            "dct_hybrid_learned_router",
            "glrc9_ccp",
            "dct_phase_cycle",
            "dct_dsl8",
            "polyphase_lifting",
        }:
            raise ValueError(
                "spatial_operator must be 'local_depthwise', 'nk9', 'phase_gauge', "
                "'cosine_integral', 'green_resolvent', or "
                "'parseval_resolvent_directional', 'dct_hybrid', "
                "'gchaf_r1', 'gchaf_ar9f', 'gchaf_ccar9', "
                "'dct_global9_conditioned', "
                "'dh_lms4_additive', 'dh_lms4_signed', 'dh_lms4_act4', "
                "'dh_lms4_dual_route', "
                "'dh_mwf9', "
                "'dh_mwf9_independent', 'dh_swf9_independent', "
                "'dh_swf9_scale_tied', 'dh_swf9_scale_rank2', "
                "'dct_hybrid_learned_router', 'glrc9_ccp', 'dct_phase_cycle', "
                "'dct_dsl8', or 'polyphase_lifting'"
            )
        self.state_channels = state_channels
        self.widely_linear = widely_linear
        self.spatial_operator = spatial_operator
        self.gain_chart = gain_chart
        d = state_channels
        self.mix_m_real = nn.Parameter(torch.empty(d, d))
        self.mix_m_imag = nn.Parameter(torch.empty(d, d))
        if widely_linear:
            self.mix_n_real = nn.Parameter(torch.empty(d, d))
            self.mix_n_imag = nn.Parameter(torch.empty(d, d))
        else:
            self.register_parameter("mix_n_real", None)
            self.register_parameter("mix_n_imag", None)
        self.depthwise_real = nn.Parameter(torch.empty(d, 1, 3, 3))
        self.depthwise_imag = nn.Parameter(torch.empty(d, 1, 3, 3))
        if spatial_operator in {"dh_mwf9_independent", "dh_swf9_independent"}:
            self.gamma_real = nn.Parameter(torch.empty(d, 9))
            self.gamma_imag = nn.Parameter(torch.empty(d, 9))
        elif spatial_operator == "dh_lms4_dual_route":
            self.gamma_real = nn.Parameter(torch.empty(d, 3))
            self.gamma_imag = nn.Parameter(torch.empty(d, 3))
        elif spatial_operator in {
            "dh_lms4_additive",
            "dh_lms4_signed",
            "dh_lms4_act4",
        }:
            self.gamma_real = nn.Parameter(torch.empty(d, 4))
            self.gamma_imag = nn.Parameter(torch.empty(d, 4))
        elif spatial_operator == "dh_swf9_scale_tied":
            self.gamma_real = nn.Parameter(torch.empty(d, 3))
            self.gamma_imag = nn.Parameter(torch.empty(d, 3))
        elif spatial_operator == "dh_swf9_scale_rank2":
            self.gamma_real = nn.Parameter(torch.empty(d, 6))
            self.gamma_imag = nn.Parameter(torch.empty(d, 6))
        else:
            self.register_parameter("gamma_real", None)
            self.register_parameter("gamma_imag", None)
        if spatial_operator == "gchaf_r1":
            self.role_gain_real = nn.Parameter(torch.empty(d, 9))
            self.role_gain_imag = nn.Parameter(torch.empty(d, 9))
        else:
            self.register_parameter("role_gain_real", None)
            self.register_parameter("role_gain_imag", None)
        if spatial_operator in {"dh_lms4_act4", "dh_lms4_dual_route"}:
            self.eta_real = nn.Parameter(torch.empty(d, 4))
            self.eta_imag = nn.Parameter(torch.empty(d, 4))
        else:
            self.register_parameter("eta_real", None)
            self.register_parameter("eta_imag", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        d = self.state_channels
        with torch.no_grad():
            self.mix_m_real.copy_(
                torch.eye(
                    d,
                    device=self.mix_m_real.device,
                    dtype=self.mix_m_real.dtype,
                )
            )
            self.mix_m_imag.zero_()
            if self.mix_n_real is not None:
                self.mix_n_real.zero_()
                self.mix_n_imag.zero_()
            self.depthwise_real.zero_()
            if self.gain_chart == "direct":
                self.depthwise_real[:, 0, 1, 1].fill_(1.0)
            else:
                chart = _q9_fixed_u9_chart(
                    device=self.depthwise_real.device,
                    dtype=self.depthwise_real.dtype,
                )
                identity = torch.zeros(
                    d,
                    9,
                    device=self.depthwise_real.device,
                    dtype=self.depthwise_real.dtype,
                )
                identity[:, 0] = 1.0
                raw_identity = torch.matmul(identity, chart)
                self.depthwise_real[:, 0].copy_(
                    torch.roll(
                        raw_identity.reshape(d, 3, 3),
                        shifts=(1, 1),
                        dims=(-2, -1),
                    )
                )
            self.depthwise_imag.zero_()
            if self.gamma_real is not None:
                self.gamma_real.zero_()
                self.gamma_imag.zero_()
            if self.eta_real is not None:
                self.eta_real.zero_()
                self.eta_imag.zero_()
            if self.role_gain_real is not None:
                self.role_gain_real.zero_()
                self.role_gain_imag.zero_()
            if self.spatial_operator == "dct_hybrid_learned_router":
                # A zero residual makes every routing tangent exactly dead.
                # Seed all nine actions with a small balanced complex phase
                # ring.  The roots sum to zero, so this is a near-identity,
                # non-degenerate compiler initialization rather than common
                # positive gain inflation.  Final F9 stability still comes
                # exclusively from the unchanged q9 legalizer.
                slot = torch.arange(
                    9,
                    device=self.depthwise_real.device,
                    dtype=self.depthwise_real.dtype,
                )
                angle = (2.0 * torch.pi / 9.0) * slot
                residual_real = 0.02 * torch.cos(angle)
                residual_imag = 0.02 * torch.sin(angle)
                rolled_real = torch.roll(
                    self.depthwise_real[:, 0],
                    shifts=(-1, -1),
                    dims=(-2, -1),
                ).reshape(d, 9)
                rolled_imag = torch.roll(
                    self.depthwise_imag[:, 0],
                    shifts=(-1, -1),
                    dims=(-2, -1),
                ).reshape(d, 9)
                rolled_real.add_(residual_real[None])
                rolled_imag.add_(residual_imag[None])
                self.depthwise_real[:, 0].copy_(
                    torch.roll(
                        rolled_real.reshape(d, 3, 3),
                        shifts=(1, 1),
                        dims=(-2, -1),
                    )
                )
                self.depthwise_imag[:, 0].copy_(
                    torch.roll(
                        rolled_imag.reshape(d, 3, 3),
                        shifts=(1, 1),
                        dims=(-2, -1),
                    )
                )


def initialize_variance_matched_random_compiler_q9_(
    q9_generator: ComplexF9Q9Generator,
    *,
    seed: int,
    calibration_size: int = 32,
) -> float:
    """Seed random q9 gains with a registered latent-action RMS match.

    The control is the unchanged GLRC9 identity compiler.  The treatment draws
    independent unit-variance real and imaginary 3x3 gains from a fixed CPU
    generator, then applies one scalar chosen by a deterministic bisection so
    that the pre-head GLRC9 latent RMS on a fixed Gaussian calibration feature
    matches the identity control.  No outer weight, head, parameter, or
    checkpoint key is changed.  The returned scalar is an audit value only.
    """

    if q9_generator.spatial_operator != "dct_hybrid":
        raise ValueError("variance-matched random initialization requires GLRC9")
    if q9_generator.gain_chart != "direct":
        raise ValueError("variance-matched random initialization uses direct gamma")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if type(calibration_size) is not int or calibration_size < 3:
        raise ValueError("calibration_size must be an integer >= 3")

    device = q9_generator.depthwise_real.device
    dtype = q9_generator.depthwise_real.dtype
    d = q9_generator.state_channels
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(seed)
    random_real = torch.randn(
        (d, 1, 3, 3), generator=cpu_rng, dtype=dtype, device="cpu"
    ).to(device=device)
    random_imag = torch.randn(
        (d, 1, 3, 3), generator=cpu_rng, dtype=dtype, device="cpu"
    ).to(device=device)
    calibration_rng = torch.Generator(device="cpu")
    calibration_rng.manual_seed(seed ^ 0x5A17_9E21)
    calibration = torch.randn(
        (1, 2 * d, calibration_size, calibration_size),
        generator=calibration_rng,
        dtype=dtype,
        device="cpu",
    ).to(device=device)
    if q9_generator.mix_n_real is None:
        mix_n_real = torch.zeros_like(q9_generator.mix_m_real)
        mix_n_imag = torch.zeros_like(q9_generator.mix_m_imag)
    else:
        mix_n_real = q9_generator.mix_n_real
        mix_n_imag = q9_generator.mix_n_imag

    identity_real = torch.zeros_like(q9_generator.depthwise_real)
    identity_real[:, 0, 1, 1] = 1.0
    identity_imag = torch.zeros_like(q9_generator.depthwise_imag)

    def latent_rms(gain_real: Tensor, gain_imag: Tensor) -> Tensor:
        latent = _fused_q9_glrc9_latent_graph(
            calibration,
            q9_generator.mix_m_real,
            q9_generator.mix_m_imag,
            mix_n_real,
            mix_n_imag,
            gain_real,
            gain_imag,
        )
        return torch.sqrt(latent.square().mean().clamp_min(torch.finfo(dtype).eps))

    with torch.no_grad():
        target = latent_rms(identity_real, identity_imag)
        lo = 0.0
        hi = 1.0
        hi_rms = latent_rms(hi * random_real, hi * random_imag)
        for _ in range(32):
            if bool(hi_rms >= target):
                break
            hi *= 2.0
            hi_rms = latent_rms(hi * random_real, hi * random_imag)
        if not bool(torch.isfinite(hi_rms)) or not bool(hi_rms >= target):
            raise RuntimeError(
                "variance-matched q9 random initialization could not bracket "
                "the GLRC9 identity latent RMS"
            )
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            mid_rms = latent_rms(mid * random_real, mid * random_imag)
            if bool(mid_rms < target):
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
        q9_generator.depthwise_real.copy_(scale * random_real)
        q9_generator.depthwise_imag.copy_(scale * random_imag)
    return float(scale)


class Q9BlockGAdapter(nn.Module):
    """Low-rank block-local real-linear adapter for the q9 ``G`` map.

    The shared q9 generator remains the main compiler trunk.  This module adds
    a small block-local update to its widely-linear pointwise matrices before
    the unchanged complex depthwise jet and q9 head:

    ``M_b = M_shared + U_M V`` and ``N_b = N_shared + U_N V``.

    ``U`` is zero-initialized, so the candidate has the same forward action at
    initialization while contributing new first-order compiler directions.
    ``V`` is a fixed orthonormal row basis and is deliberately not a
    parameter.  The q9 chart, reciprocal projection, F9 stencil, and K12
    recurrence are untouched.
    """

    def __init__(
        self,
        state_channels: int,
        *,
        rank: int = 4,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        if type(rank) is not int or not 1 <= rank <= state_channels:
            raise ValueError("adapter rank must lie in [1,state_channels]")
        if type(seed) is not int:
            raise TypeError("adapter seed must be an integer")
        self.state_channels = state_channels
        self.rank = rank
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            basis = torch.randn(rank, state_channels)
            basis = torch.linalg.qr(basis.transpose(0, 1), mode="reduced").Q
            basis = basis.transpose(0, 1).contiguous()
        self.register_buffer("basis", basis)
        d = state_channels
        self.delta_m_real = nn.Parameter(torch.zeros(d, rank))
        self.delta_m_imag = nn.Parameter(torch.zeros(d, rank))
        self.delta_n_real = nn.Parameter(torch.zeros(d, rank))
        self.delta_n_imag = nn.Parameter(torch.zeros(d, rank))

    def effective_matrices(
        self,
        generator: ComplexF9Q9Generator,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if generator.state_channels != self.state_channels:
            raise ValueError("q9 adapter/generator state widths do not match")
        basis = self.basis.to(dtype=generator.mix_m_real.dtype)
        return (
            generator.mix_m_real + self.delta_m_real @ basis,
            generator.mix_m_imag + self.delta_m_imag @ basis,
            generator.mix_n_real + self.delta_n_real @ basis,
            generator.mix_n_imag + self.delta_n_imag @ basis,
        )


class Q9BlockSimplexRouter(nn.Module):
    """Block-local learned tangent on DH-9's nine-action simplex.

    A full 9-by-9 map is kept per complex state channel, together with one
    slot bias.  Both tensors start at zero, so the effective routing begins on
    the parent DH energy rule while their ordinary task gradients are active.
    """

    slot_count = 9

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.weight = nn.Parameter(
            torch.zeros(state_channels, self.slot_count, self.slot_count)
        )
        self.bias = nn.Parameter(torch.zeros(state_channels, self.slot_count))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class Q9FullActionResidual(nn.Module):
    """Zero-parent block-private map over all nine pre-sum GLRC9 actions.

    Every state owns a general real ``2 x (9 x 2)`` map.  This retains both
    complex-linear and conjugate directions for every action while forbidding
    cross-state mixing.  The bias-free zero initialization is the exact GLRC9
    parent section.
    """

    action_count = 9
    complex_coordinates = 2

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.weight = nn.Parameter(
            torch.zeros(
                self.complex_coordinates * state_channels,
                self.complex_coordinates * self.action_count,
                1,
                1,
            )
        )

    @property
    def action_weight(self) -> Tensor:
        return self.weight.reshape(
            self.state_channels,
            self.complex_coordinates,
            self.action_count,
            self.complex_coordinates,
        )

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)


class Q9ContentConditionedActionResidual(nn.Module):
    """Zero-parent block-private CCAR9 content/action coefficient tensor.

    Every state owns ``T[D,2,9,8,2]``: output real coordinate, GLRC9 action,
    fixed content contrast, and input real coordinate.  It is stored under the
    checkpoint-friendly conv-shaped key ``[2D,144,1,1]`` with no bias.
    """

    action_count = 9
    content_count = 8
    complex_coordinates = 2

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.weight = nn.Parameter(
            torch.zeros(
                self.complex_coordinates * state_channels,
                self.action_count
                * self.content_count
                * self.complex_coordinates,
                1,
                1,
            )
        )

    @property
    def content_action_weight(self) -> Tensor:
        return self.weight.reshape(
            self.state_channels,
            self.complex_coordinates,
            self.action_count,
            self.content_count,
            self.complex_coordinates,
        )

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)


class Q9DirectActionResidual(nn.Module):
    """Zero-initialized, block-local direct residual into the q9 raw field.

    The existing shared-G/H compiler remains the main path. This grouped
    3x3 path reads the original packed complex drive directly and emits the
    same nine-real-coordinate field as H. Its zero initialization preserves
    the initial forward action while exposing an independent first-order
    compiler direction once the task path is live:

    ``raw = raw_shared_G_H + R_b(packed_f)``.

    The q9 chart, reciprocal projection, F9 stencil, and K12 recurrence are
    unchanged. The module is owned by each CLPO block.
    """

    coordinate_count = 9

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.head = nn.Conv2d(
            2 * state_channels,
            9 * state_channels,
            kernel_size=3,
            padding=1,
            groups=state_channels,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, packed_f: Tensor) -> Tensor:
        if packed_f.ndim != 4:
            raise ValueError("packed_f must have shape [B,2D,H,W]")
        if packed_f.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "packed_f channel dimension must equal 2 * state_channels"
            )
        return self.head(packed_f)


class Q9PostHLowRankActionResidual(nn.Module):
    """Zero-initialized block-private low-rank residual in raw q9 space.

    The DH parent remains ``H_b(G_D(f))``.  This branch reads the same live
    packed drive and adds ``U_b V_b f`` only after the parent readout, so its
    first active parameter tangent reaches the raw q9 coordinates without a
    second H re-encoding or a scalar credit gate.  Every block uses the same
    relative-rank law ``ceil(D / 16)``.  ``V_b`` starts orthogonal and ``U_b``
    starts at zero, preserving the exact DH initialization while exposing a
    nonzero first-step gradient for ``U_b``.
    """

    coordinate_count = 9

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.rank = max(1, (state_channels + 15) // 16)
        self.down = nn.Conv2d(
            2 * state_channels,
            self.rank,
            kernel_size=1,
            bias=False,
        )
        self.up = nn.Conv2d(
            self.rank,
            self.coordinate_count * state_channels,
            kernel_size=1,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight.view(self.rank, -1))
        nn.init.zeros_(self.up.weight)

    def forward(self, packed_f: Tensor) -> Tensor:
        if packed_f.ndim != 4:
            raise ValueError("packed_f must have shape [B,2D,H,W]")
        if packed_f.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "packed_f channel dimension must equal 2 * state_channels"
            )
        return self.up(self.down(packed_f))


class Q9BlockPDRResidual(nn.Module):
    """Block-private post-latent low-rank q9 residual for PDR.

    For the DH-9 latent ``z_b`` immediately before ``H_b``, this module
    computes ``U_b(V_b(z_b))`` with ``rank=ceil(D/rank_divisor)``.  The
    registered divisors are 8 and 16, with 8 as the default.  ``U_b`` is zero at
    initialization and ``V_b`` is row-orthogonal with gain ``1/sqrt(3)`` under
    the module's RNG seed, so the complete compiler initially equals DH-9 while
    the first active tangent is carried by ``U_b``.  Both projections are
    bias-free 1x1 convolutions.
    """

    coordinate_count = 9

    def __init__(self, state_channels: int, rank_divisor: int = 8) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        if type(rank_divisor) is not int or rank_divisor not in {8, 16}:
            raise ValueError("PDR rank_divisor must be one of {8, 16}")
        self.state_channels = state_channels
        self.rank_divisor = rank_divisor
        self.rank = max(1, (state_channels + rank_divisor - 1) // rank_divisor)
        self.v = nn.Conv2d(
            2 * state_channels,
            self.rank,
            kernel_size=1,
            bias=False,
        )
        self.u = nn.Conv2d(
            self.rank,
            self.coordinate_count * state_channels,
            kernel_size=1,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(
            self.v.weight.reshape(self.rank, -1),
            gain=1.0 / math.sqrt(3.0),
        )
        nn.init.zeros_(self.u.weight)

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4:
            raise ValueError("latent must have shape [B,2D,H,W]")
        if latent.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "latent channel dimension must equal 2 * state_channels"
            )
        return self.u(self.v(latent))


class Q9BlockLowRankHeadResidual(nn.Module):
    """Bounded all-block low-rank cross-state remap before the DH q9 head.

    Every CLPO block receives the same relative-rank law
    ``rank = ceil(state_channels / 16)``.  The down projection is orthogonal at
    initialization and remains learnable; the up projection starts at zero.
    Consequently the initial compiler function is exactly the DH parent while
    the new block-local H-input tangent is active on the first optimizer step.
    The learned scalar envelope stays below one half and the Frobenius-product
    denominator bounds the remap operator norm by that envelope.
    """

    coordinate_count = 9

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.rank = max(1, (state_channels + 15) // 16)
        self.down = nn.Conv2d(
            2 * state_channels,
            self.rank,
            kernel_size=1,
            bias=False,
        )
        self.up = nn.Conv2d(
            self.rank,
            2 * state_channels,
            kernel_size=1,
            bias=False,
        )
        self.gate_logit = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight.view(self.rank, -1))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.gate_logit)

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4:
            raise ValueError("latent must have shape [B,2D,H,W]")
        if latent.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "latent channel dimension must equal 2 * state_channels"
            )
        residual = self.up(self.down(latent))
        denominator = 1.0 + (
            torch.linalg.vector_norm(self.down.weight)
            * torch.linalg.vector_norm(self.up.weight)
        )
        envelope = 0.5 * torch.sigmoid(self.gate_logit)
        return latent + (envelope / denominator) * residual


class Q9ClusteredHResidual(nn.Module):
    """Zero-initialized grouped H residual shared by one frozen block cluster."""

    coordinate_count = 9

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels not in {32, 64}:
            raise ValueError("clustered H residual state width must be D32 or D64")
        self.state_channels = state_channels
        self.weight = nn.Parameter(
            torch.zeros(
                self.coordinate_count * state_channels,
                2,
                3,
                3,
            )
        )

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4 or latent.shape[1] != 2 * self.state_channels:
            raise ValueError("latent must have shape [B,2D,H,W]")
        return F.conv2d(
            latent,
            self.weight,
            None,
            padding=1,
            groups=self.state_channels,
        )


class Q9BlockDepth2HeadResidual(nn.Module):
    """Bounded block-private second latent depth before the DH q9 head.

    This is the q9 analogue of the historical depth-2 coefficient hidden
    residual: a grouped 3x3 refinement is added at the latent input of H, so
    every new contribution still passes through the unchanged parent H.  The
    per-state gain starts at zero; therefore initialization is exactly DH while
    the gain receives a first-step task gradient.  The refinement weights
    unlock after the gain leaves zero.
    """

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.refine = nn.Conv2d(
            2 * state_channels,
            2 * state_channels,
            kernel_size=3,
            padding=1,
            groups=state_channels,
            bias=False,
        )
        self.gain = nn.Parameter(torch.zeros(state_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.dirac_(self.refine.weight, groups=self.state_channels)
        nn.init.zeros_(self.gain)

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4:
            raise ValueError("latent must have shape [B,2D,H,W]")
        if latent.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "latent channel dimension must equal 2 * state_channels"
            )
        refined = self.refine(latent)
        batch, _, height, width = refined.shape
        paired = refined.reshape(
            batch, self.state_channels, 2, height, width
        )
        magnitude = torch.sqrt(
            paired.square().sum(dim=2) + torch.finfo(refined.dtype).eps
        )
        refined = (
            paired * torch.sigmoid(magnitude).unsqueeze(2)
        ).reshape_as(refined)
        state_weight_norm = torch.linalg.vector_norm(
            self.refine.weight.reshape(self.state_channels, -1),
            dim=1,
        )
        denominator = 1.0 + 3.0 * state_weight_norm
        envelope = 0.5 * torch.tanh(self.gain) / denominator
        envelope = envelope.repeat_interleave(2).view(
            1, 2 * self.state_channels, 1, 1
        )
        return latent + envelope * refined


class Q9BlockGivensHeadResidual(nn.Module):
    """Block-private, credit-decoupled bounded transport before the DH head.

    The three zero-initialized per-state gains independently control radial
    amplitude, horizontal transport, and vertical transport.  Hence step zero
    is exactly the DH parent while all three parameter groups have a direct
    first-order task tangent.  The same law is used at every CLPO block.
    """

    amplitude_scale = 0.1
    transport_blend = 0.5

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        self.state_channels = state_channels
        self.amplitude_gain = nn.Parameter(torch.zeros(state_channels))
        self.horizontal_gain = nn.Parameter(torch.zeros(state_channels))
        self.vertical_gain = nn.Parameter(torch.zeros(state_channels))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.amplitude_gain)
        nn.init.zeros_(self.horizontal_gain)
        nn.init.zeros_(self.vertical_gain)

    def forward(self, latent: Tensor) -> Tensor:
        if latent.ndim != 4:
            raise ValueError("latent must have shape [B,2D,H,W]")
        if latent.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "latent channel dimension must equal 2 * state_channels"
            )
        batch, _, height, width = latent.shape
        paired = latent.reshape(
            batch, self.state_channels, 2, height, width
        )
        transported = givens_strang_transport(
            paired,
            paired,
            self.horizontal_gain,
            self.vertical_gain,
            eta=self.transport_blend,
        )
        magnitude = torch.sqrt(
            paired.square().sum(dim=2) + torch.finfo(latent.dtype).eps
        )
        radial = paired * torch.sigmoid(magnitude).unsqueeze(2)
        amplitude = (
            self.amplitude_scale * torch.tanh(self.amplitude_gain)
        ).view(1, self.state_channels, 1, 1, 1)
        return (transported + amplitude * radial).reshape_as(latent)


class Q9RealReadoutHead(nn.Module):
    """Grouped real ``H`` head producing nine raw q9 coordinates per state.

    ``kernel_size=3`` is the historical GLRC9 readout.  A 1x1 readout is an
    explicit candidate mode, not a silent change to the parent, so the
    spatial support is carried by the module shape and the compiler uses the
    corresponding exact convolution padding in both directions.
    """

    coordinate_count = 9

    def __init__(self, state_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels <= 0:
            raise ValueError("state_channels must be a positive integer")
        if type(kernel_size) is not int or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.state_channels = state_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(9 * state_channels, 2, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(9 * state_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.bias)


# GLRC9-AW1 deliberately leaves the canonical GLRC9 generator/head weight draw
# untouched and changes only the raw q9 head bias.  The four complex direction
# slots form a small east/north/west/south ring in the fixed Cartesian chart.
GLRC9_AW1_DIRECTION_AMPLITUDE = 0.10
GLRC9_AW1_RAW_BIAS = (
    0.0,
    GLRC9_AW1_DIRECTION_AMPLITUDE,
    0.0,
    0.0,
    GLRC9_AW1_DIRECTION_AMPLITUDE,
    -GLRC9_AW1_DIRECTION_AMPLITUDE,
    0.0,
    0.0,
    -GLRC9_AW1_DIRECTION_AMPLITUDE,
)


def glrc9_aw1_row_radius(
    *,
    rho_max: float = 0.95,
    dtype: torch.dtype = torch.float32,
) -> float:
    """Return the FP32 q9 row radius induced by the GLRC9-AW1 bias anchor.

    The production q9 legalizer uses ``raw_pair_mass=2*sum(|pair|)`` and
    ``scale=0.5*(safe_rho-center_floor)/(1+raw_pair_mass)``.  At center logit
    zero, sigmoid contributes one half, so this closed value is a useful
    contract check without running a network.
    """

    if not 0.0 < float(rho_max) < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    epsilon = float(torch.finfo(dtype).eps)
    safe_rho = float(rho_max) - 32.0 * epsilon
    center_floor = 0.10
    pair_mass = 2.0 * 4.0 * GLRC9_AW1_DIRECTION_AMPLITUDE
    scale = 0.5 * (safe_rho - center_floor) / (1.0 + pair_mass)
    off_mass = scale * pair_mass
    center = center_floor + off_mass + (
        safe_rho - center_floor - 2.0 * off_mass
    ) * 0.5
    return center + off_mass


def initialize_glrc9_aw1_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
) -> None:
    """Initialize the GLRC9-AW1 raw q9 bias while preserving GLRC9 weights.

    Coordinate order is ``(c, ax.re, ax.im, ay.re, ay.im, d1.re, d1.im,
    d2.re, d2.im)``.  The center logit is exactly zero (maximum sigmoid
    derivative); the four direction slots are a small orthogonal/anti-
    symmetric ring.  The legalizer and F9/K12 recurrence remain unchanged.
    """

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("GLRC9-AW1 initializer requires a Q9RealReadoutHead")
    if not 0.0 < float(rho_max) < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    theta = head.bias.new_tensor(GLRC9_AW1_RAW_BIAS)
    with torch.no_grad():
        head.bias.copy_(theta.repeat(head.state_channels))
        torch.testing.assert_close(
            head.bias.view(head.state_channels, head.coordinate_count),
            theta.expand(head.state_channels, -1),
            rtol=0.0,
            atol=0.0,
        )


def initialize_moment_exponential_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize the M2-9 natural chart at its specified zero-latent anchor."""

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("M2-9 initializer requires a Q9RealReadoutHead")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 1)
    epsilon = torch.finfo(head.bias.dtype).eps
    safe_rho = float(rho_max) - 32.0 * epsilon
    center_floor = 0.10
    contraction_budget = 0.5 * (safe_rho - center_floor)
    rho_star = math.exp(-1.0 / 11.0)
    tau_star = 1.0 / 11.0
    a_star = rho_star * tau_star / 8.0
    mass_star = 4.0 * a_star
    eta = a_star / 8.0
    mass_probability = 2.0 * mass_star / contraction_budget
    if not 0.0 < mass_probability < 1.0:
        raise ValueError("M2-9 anchor mass is outside the sigmoid domain")
    theta = head.bias.new_tensor(
        (
            math.log(mass_probability / (1.0 - mass_probability)),
            eta,
            -eta / 2.0,
            eta / 3.0,
            0.0,
            0.0,
            0.0,
            0.0,
            math.log((1.0 - epsilon) / epsilon),
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))


def initialize_moment_exponential_aw1_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize M2-AW1 at its AdamW-aware nonsaturated natural anchor."""

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("M2-AW1 initializer requires a Q9RealReadoutHead")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 1)
    theta = head.bias.new_tensor(
        (
            -1.4158274722,
            0.014,
            -0.010,
            0.008,
            0.30,
            -0.20,
            0.15,
            -0.10,
            0.9444616088,
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))
        torch.testing.assert_close(
            head.bias.view(head.state_channels, head.coordinate_count),
            theta.expand(head.state_channels, -1),
            rtol=0.0,
            atol=0.0,
        )


def initialize_logpolar_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize LP-9 at its closed raw log-radius/phase anchor."""

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("LP-9 initializer requires a Q9RealReadoutHead")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 2)
    theta = head.bias.new_tensor(
        (
            math.log(0.35 / 0.65),
            math.log(0.60),
            math.log(0.40),
            math.log(0.25),
            math.log(0.15),
            0.20,
            -0.40,
            0.70,
            -0.90,
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))


def initialize_logpolar_aw1_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize LP-AW1 with the raw-first AdamW-aware LP-9 anchor.

    LP-AW1 deliberately shares LP-9's weight draw, raw chart, and downstream
    legalizer.  Only the initialization bias is different, so the checkpoint
    identity remains explicit without adding parameters or changing the
    operator equation.
    """

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("LP-AW1 initializer requires a Q9RealReadoutHead")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 2)
    theta = head.bias.new_tensor(
        (
            0.10,
            -0.7765287895,
            -1.1086626245,
            -1.3093333200,
            -1.6607312068,
            0.35,
            -0.75,
            1.10,
            -2.20,
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))
        torch.testing.assert_close(
            head.bias.view(head.state_channels, head.coordinate_count),
            theta.expand(head.state_channels, -1),
            rtol=0.0,
            atol=0.0,
        )


def initialize_mass_simplex_phase_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize AP-S9 at its specified FP32 production anchor."""

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError(
            "mass-simplex-phase initializer requires a Q9RealReadoutHead"
        )
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 3)
    theta = head.bias.new_tensor(
        (
            0.8472978472709656,
            1.8718022108078003,
            1.0560526847839355,
            0.7537717819213867,
            0.6283185482025146,
            -1.2566370964050293,
            1.346396803855896,
            -2.6179938316345215,
            -0.6190392374992371,
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))


def initialize_mass_simplex_phase_aw1_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize AP-AW1 with a less-starved AdamW-aware simplex anchor."""

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("AP-AW1 initializer requires a Q9RealReadoutHead")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 3)
    theta = head.bias.new_tensor(
        (
            0.3227733923,
            1.1239300967,
            0.7308875085,
            0.4307829161,
            0.45,
            -0.80,
            1.35,
            -2.35,
            0.10,
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))
        torch.testing.assert_close(
            head.bias.view(head.state_channels, head.coordinate_count),
            theta.expand(head.state_channels, -1),
            rtol=0.0,
            atol=0.0,
        )


def initialize_center_mass_inverse_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize CM-I9 at the fixed physical off-mass/chart anchor."""

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("CM-I9 initializer requires a Q9RealReadoutHead")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 4)
    theta = head.bias.new_tensor(
        (
            -0.4895482361316681,
            1.7917594909667969,
            1.216395378112793,
            0.7537717819213867,
            0.0,
            1.0471975805062866,
            -1.5707963705062866,
            2.6179938316345215,
            -0.9946225881576538,
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))


def initialize_center_mass_inverse_aw1_q9_(
    head: Q9RealReadoutHead,
    *,
    rho_max: float = 0.95,
    generator: torch.Generator | None = None,
) -> None:
    """Initialize CMI-AW1 at its balanced physical-mass inverse anchor."""

    if not isinstance(head, Q9RealReadoutHead):
        raise TypeError("CMI-AW1 initializer requires a Q9RealReadoutHead")
    if not 0.0 < rho_max < 1.0:
        raise ValueError("rho_max must lie strictly between zero and one")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0xE9027000 + 4)
    theta = head.bias.new_tensor(
        (
            -0.5753641449,
            1.2527629685,
            0.8472978604,
            0.4054651081,
            0.30,
            -0.65,
            1.25,
            -2.10,
            0.0,
        )
    )
    with torch.no_grad():
        weight_cpu = torch.empty(
            tuple(head.weight.shape),
            dtype=head.weight.dtype,
            device="cpu",
        )
        weight_cpu.normal_(mean=0.0, std=1.0e-3, generator=generator)
        head.weight.copy_(weight_cpu.to(device=head.weight.device))
        head.bias.copy_(theta.repeat(head.state_channels))
        torch.testing.assert_close(
            head.bias.view(head.state_channels, head.coordinate_count),
            theta.expand(head.state_channels, -1),
            rtol=0.0,
            atol=0.0,
        )


def q9_effective_head_parameters(
    head_weights: tuple[Tensor, ...],
    head_biases: tuple[Tensor, ...],
    *,
    index: int,
    mix: float,
) -> tuple[Tensor, Tensor]:
    """Apply the exact CTK map to functional q9 readout-head tensors."""

    if len(head_weights) != len(head_biases):
        raise ValueError("CTK head weight/bias peer counts must match")
    if len(head_weights) < 2:
        raise ValueError("DCT-CTK requires at least two peer heads")
    if not 0 <= index < len(head_weights):
        raise ValueError("CTK head index is outside the peer group")
    if not 0.0 <= mix < 1.0:
        raise ValueError("CTK mix must lie in [0, 1)")
    if any(weight.shape != head_weights[0].shape for weight in head_weights):
        raise ValueError("CTK peer head weights must have one shape")
    if any(bias.shape != head_biases[0].shape for bias in head_biases):
        raise ValueError("CTK peer head biases must have one shape")
    mean_weight = torch.stack(head_weights, dim=0).mean(dim=0)
    mean_bias = torch.stack(head_biases, dim=0).mean(dim=0)
    private = 1.0 - mix
    return (
        private * head_weights[index] + mix * mean_weight,
        private * head_biases[index] + mix * mean_bias,
    )


class ComplexF9Q9Compiler:
    """A lightweight binding of one ``G`` module and one q9 ``H`` module.

    This class is intentionally not an ``nn.Module``.  The owning network
    registers shared generators/heads exactly once in its q9 banks, while each
    CLPO block keeps only this callable binding.  Independent blocks own their
    generator and head directly on the block, so all optimizer parameters and
    checkpoint keys remain explicit.
    """

    def __init__(
        self,
        generator: ComplexF9Q9Generator,
        head: Q9RealReadoutHead,
        g_adapter: Q9BlockGAdapter | None = None,
        direct_residual: Q9DirectActionResidual | None = None,
        simplex_router: Q9BlockSimplexRouter | None = None,
        ccp_private_head: Q9RealReadoutHead | None = None,
        low_rank_head_residual: Q9BlockLowRankHeadResidual | None = None,
        depth2_head_residual: Q9BlockDepth2HeadResidual | None = None,
        posth_low_rank_residual: Q9PostHLowRankActionResidual | None = None,
        givens_head_residual: Q9BlockGivensHeadResidual | None = None,
        pdr_residual: Q9BlockPDRResidual | None = None,
        *,
        clustered_h_residual: Q9ClusteredHResidual | None = None,
        role_film_residual: Q9RoleFiLMResidual | None = None,
        full_action_residual: Q9FullActionResidual | None = None,
        content_conditioned_action_residual: (
            Q9ContentConditionedActionResidual | None
        ) = None,
        tap_major_output: bool = False,
        centered_state_energy_bias: bool = False,
        rho_max: float = 0.95,
        moment_exponential: bool = False,
        logpolar: bool = False,
        mass_simplex_phase: bool = False,
        center_mass_inverse: bool = False,
        block_role_code: float = 0.0,
        routing_continuation: Tensor | None = None,
        local_family_continuation: Tensor | None = None,
        mn_offdiag_continuation: Tensor | None = None,
    ) -> None:
        if generator.state_channels != head.state_channels:
            raise ValueError("q9 generator/head state widths must match")
        if routing_continuation is not None and routing_continuation.numel() != 1:
            raise ValueError("GLRC9 routing continuation g must be scalar")
        if (
            generator.spatial_operator == "dct_hybrid_routing_continuation"
        ) != (routing_continuation is not None):
            raise ValueError(
                "GLRC9 routing continuation requires exactly one scalar g binding"
            )
        if local_family_continuation is not None and local_family_continuation.numel() != 1:
            raise ValueError("GLRC9 local-family continuation g must be scalar")
        if (
            generator.spatial_operator == "dct_hybrid_local_family_continuation"
        ) != (local_family_continuation is not None):
            raise ValueError(
                "GLRC9 local-family continuation requires exactly one scalar g binding"
            )
        if mn_offdiag_continuation is not None and mn_offdiag_continuation.numel() != 1:
            raise ValueError("GLRC9 M/N continuation g must be scalar")
        if (
            generator.spatial_operator == GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR
        ) != (mn_offdiag_continuation is not None):
            raise ValueError(
                "GLRC9 M/N continuation requires exactly one scalar g binding"
            )
        if (
            g_adapter is not None
            and generator.state_channels != g_adapter.state_channels
        ):
            raise ValueError("q9 generator/adapter state widths must match")
        if (
            direct_residual is not None
            and generator.state_channels != direct_residual.state_channels
        ):
            raise ValueError(
                "q9 generator/direct residual state widths do not match"
            )
        if (
            simplex_router is not None
            and generator.state_channels != simplex_router.state_channels
        ):
            raise ValueError("q9 generator/simplex router widths do not match")
        if (
            generator.spatial_operator == "dct_hybrid_learned_router"
        ) != (simplex_router is not None):
            raise ValueError(
                "the learned DH spatial operator requires exactly one "
                "block-local simplex router"
            )
        if (
            full_action_residual is not None
            and generator.state_channels
            != full_action_residual.state_channels
        ):
            raise ValueError(
                "q9 generator/full action residual widths do not match"
            )
        if (generator.spatial_operator == "gchaf_ar9f") != (
            full_action_residual is not None
        ):
            raise ValueError(
                "GCHAF-AR9F requires exactly one block-private full action residual"
            )
        if (
            content_conditioned_action_residual is not None
            and generator.state_channels
            != content_conditioned_action_residual.state_channels
        ):
            raise ValueError(
                "q9 generator/content-conditioned action residual widths do not match"
            )
        if (generator.spatial_operator == "gchaf_ccar9") != (
            content_conditioned_action_residual is not None
        ):
            raise ValueError(
                "GCHAF-CCAR9 requires exactly one block-private "
                "content-conditioned action residual"
            )
        if (
            full_action_residual is not None
            and content_conditioned_action_residual is not None
        ):
            raise ValueError("GCHAF-AR9F and GCHAF-CCAR9 are separate arms")
        for name, extra_head in (("private", ccp_private_head),):
            if (
                extra_head is not None
                and generator.state_channels != extra_head.state_channels
            ):
                raise ValueError(
                    f"q9 generator/CCP {name} head state widths must match"
                )
        if generator.spatial_operator == "glrc9_ccp":
            heads_enabled = glrc9_ccp_heads_enabled(generator.state_channels)
            if heads_enabled and ccp_private_head is None:
                raise ValueError(
                    "D32/D64 glrc9_ccp requires one private live-f q9 head"
                )
            if not heads_enabled and ccp_private_head is not None:
                raise ValueError("D16 glrc9_ccp must retain the parent DH path")
        elif ccp_private_head is not None:
            raise ValueError(
                "CCP heads are only valid with the glrc9_ccp spatial operator"
            )
        if (
            clustered_h_residual is not None
            and generator.state_channels != clustered_h_residual.state_channels
        ):
            raise ValueError("q9 generator/clustered H residual widths do not match")
        if (
            clustered_h_residual is not None
            and generator.spatial_operator != "dct_hybrid"
        ):
            raise ValueError("clustered H residual is valid only on the GLRC9 latent")
        if (
            role_film_residual is not None
            and generator.state_channels != role_film_residual.state_channels
        ):
            raise ValueError("q9 generator/role-FiLM widths do not match")
        if (
            role_film_residual is not None
            and generator.spatial_operator != "dct_hybrid"
        ):
            raise ValueError("role-FiLM residual is valid only on the GLRC9 latent")
        if role_film_residual is not None and any(
            branch is not None
            for branch in (
                clustered_h_residual,
                full_action_residual,
                content_conditioned_action_residual,
                direct_residual,
                low_rank_head_residual,
                depth2_head_residual,
                givens_head_residual,
                posth_low_rank_residual,
                pdr_residual,
                ccp_private_head,
            )
        ):
            raise ValueError("GLRC9 role-FiLM is a separate candidate arm")
        if (
            low_rank_head_residual is not None
            and generator.state_channels
            != low_rank_head_residual.state_channels
        ):
            raise ValueError("q9 generator/low-rank H widths must match")
        if (
            low_rank_head_residual is not None
            and generator.spatial_operator != "dct_hybrid"
        ):
            raise ValueError(
                "the low-rank H residual is valid only on the DH-9 latent"
            )
        if (
            depth2_head_residual is not None
            and generator.state_channels
            != depth2_head_residual.state_channels
        ):
            raise ValueError("q9 generator/depth-2 H widths must match")
        if (
            depth2_head_residual is not None
            and generator.spatial_operator != "dct_hybrid"
        ):
            raise ValueError(
                "the depth-2 H residual is valid only on the DH-9 latent"
            )
        if (
            givens_head_residual is not None
            and generator.state_channels
            != givens_head_residual.state_channels
        ):
            raise ValueError("q9 generator/Givens H widths must match")
        if (
            givens_head_residual is not None
            and generator.spatial_operator != "dct_hybrid"
        ):
            raise ValueError(
                "the Givens H residual is valid only on the DH-9 latent"
            )
        if (
            low_rank_head_residual is not None
            and (
                depth2_head_residual is not None
                or givens_head_residual is not None
            )
        ):
            raise ValueError("DH-HLR/H2/Givens are separate candidate arms")
        if (
            depth2_head_residual is not None
            and givens_head_residual is not None
        ):
            raise ValueError("DH-H2 and DH-H2-Givens are separate arms")
        if (
            posth_low_rank_residual is not None
            and generator.state_channels
            != posth_low_rank_residual.state_channels
        ):
            raise ValueError("q9 generator/post-H residual widths must match")
        if (
            posth_low_rank_residual is not None
            and generator.spatial_operator != "dct_hybrid"
        ):
            raise ValueError(
                "the post-H residual is valid only on the DH-9 parent"
            )
        if posth_low_rank_residual is not None and any(
            branch is not None
            for branch in (
                direct_residual,
                low_rank_head_residual,
                depth2_head_residual,
                givens_head_residual,
            )
        ):
            raise ValueError(
                "DH-PHR is a separate candidate from direct, HLR, and H2"
            )
        if (
            pdr_residual is not None
            and generator.state_channels != pdr_residual.state_channels
        ):
            raise ValueError("q9 generator/PDR widths must match")
        if pdr_residual is not None and generator.spatial_operator != "dct_hybrid":
            raise ValueError("PDR is valid only on the DH-9 latent")
        if pdr_residual is not None and any(
            branch is not None
            for branch in (
                direct_residual,
                ccp_private_head,
                low_rank_head_residual,
                depth2_head_residual,
                posth_low_rank_residual,
                givens_head_residual,
                full_action_residual,
                content_conditioned_action_residual,
                clustered_h_residual,
            )
        ):
            raise ValueError("PDR is a separate candidate arm")
        if type(tap_major_output) is not bool:
            raise TypeError("tap_major_output must be bool")
        if type(centered_state_energy_bias) is not bool:
            raise TypeError("centered_state_energy_bias must be bool")
        if not 0.0 < rho_max < 1.0:
            raise ValueError("rho_max must lie strictly between zero and one")
        if type(moment_exponential) is not bool:
            raise TypeError("moment_exponential must be bool")
        if type(logpolar) is not bool:
            raise TypeError("logpolar must be bool")
        if type(mass_simplex_phase) is not bool:
            raise TypeError("mass_simplex_phase must be bool")
        if type(center_mass_inverse) is not bool:
            raise TypeError("center_mass_inverse must be bool")
        if not math.isfinite(float(block_role_code)):
            raise ValueError("block_role_code must be finite")
        if generator.spatial_operator == "gchaf_r1":
            if generator.role_gain_real is None or generator.role_gain_imag is None:
                raise ValueError("GCHAF-R1 generator is missing shared role gains")
        elif float(block_role_code) != 0.0:
            raise ValueError("nonzero block role codes are valid only for GCHAF-R1")
        if sum(
            (moment_exponential, logpolar, mass_simplex_phase, center_mass_inverse)
        ) > 1:
            raise ValueError(
                "M2-9, LP-9, AP-S9, and CM-I9 charts are mutually exclusive"
            )
        if (
            moment_exponential
            or logpolar
            or mass_simplex_phase
            or center_mass_inverse
        ) and (
            tap_major_output
            or centered_state_energy_bias
            or any(
                branch is not None
                for branch in (
                    g_adapter,
                    direct_residual,
                    simplex_router,
                    ccp_private_head,
                    low_rank_head_residual,
                    depth2_head_residual,
                    posth_low_rank_residual,
                    givens_head_residual,
                    pdr_residual,
                    full_action_residual,
                    content_conditioned_action_residual,
                    clustered_h_residual,
                )
            )
        ):
            raise ValueError(
                "M2-9 moment chart requires the unchanged shared-head local path"
            )
        if (
            centered_state_energy_bias
            and generator.spatial_operator != "dct_hybrid"
        ):
            raise ValueError(
                "CSEB readout bias is only valid on the DH-9 dct_hybrid latent"
            )
        self.generator = generator
        self.head = head
        self.g_adapter = g_adapter
        self.direct_residual = direct_residual
        self.simplex_router = simplex_router
        self.ccp_private_head = ccp_private_head
        self.low_rank_head_residual = low_rank_head_residual
        self.depth2_head_residual = depth2_head_residual
        self.posth_low_rank_residual = posth_low_rank_residual
        self.givens_head_residual = givens_head_residual
        self.pdr_residual = pdr_residual
        self.clustered_h_residual = clustered_h_residual
        self.role_film_residual = role_film_residual
        self.full_action_residual = full_action_residual
        self.content_conditioned_action_residual = (
            content_conditioned_action_residual
        )
        self.state_channels = generator.state_channels
        self.tap_major_output = tap_major_output
        self.centered_state_energy_bias = centered_state_energy_bias
        self.rho_max = float(rho_max)
        self.moment_exponential = moment_exponential
        self.logpolar = logpolar
        self.mass_simplex_phase = mass_simplex_phase
        self.center_mass_inverse = center_mass_inverse
        self.block_role_code = float(block_role_code)
        self.routing_continuation = routing_continuation
        self.local_family_continuation = local_family_continuation
        self.mn_offdiag_continuation = mn_offdiag_continuation
        # Diagnostic-only toggle used by the CPU Canary to compare the
        # dedicated no-local endpoint with a materialized local=0 reference.
        # Production never enables it and the flag is not part of state_dict.
        self.local_family_force_materialized_reference = False
        self.local_family_no_local_endpoint_static = (
            generator.spatial_operator == "dct_hybrid_local_family_continuation"
        )
        self.mn_offdiag_diagonal_endpoint_static = (
            generator.spatial_operator == GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR
        )
        self.mn_offdiag_force_materialized_reference = False
        self._ctk_heads: tuple[Q9RealReadoutHead, ...] | None = None
        self._ctk_head_index: int | None = None
        self._ctk_mix = 0.0

    def bind_common_tangent_heads(
        self,
        heads: tuple[Q9RealReadoutHead, ...],
        *,
        index: int,
        mix: float,
    ) -> None:
        """Bind an invertible cross-block H parameterization at fixed width."""

        if len(heads) < 2:
            raise ValueError("DCT-CTK requires at least two same-width heads")
        if not 0 <= index < len(heads) or heads[index] is not self.head:
            raise ValueError("DCT-CTK head index does not match this compiler")
        if not 0.0 <= mix < 1.0:
            raise ValueError("DCT-CTK mix must lie in [0, 1)")
        if any(head.state_channels != self.state_channels for head in heads):
            raise ValueError("DCT-CTK heads must share one state width")
        self._ctk_heads = heads
        self._ctk_head_index = index
        self._ctk_mix = float(mix)

    def effective_head_parameters(self) -> tuple[Tensor, Tensor]:
        """Return this block's possibly common-tangent-mixed H tensors."""

        if self._ctk_heads is None:
            return self.head.weight, self.head.bias
        if self._ctk_head_index is None:
            raise RuntimeError("CTK peer heads are bound without an index")
        return q9_effective_head_parameters(
            tuple(head.weight for head in self._ctk_heads),
            tuple(head.bias for head in self._ctk_heads),
            index=self._ctk_head_index,
            mix=self._ctk_mix,
        )

    def reset_parameters(self) -> None:
        self.generator.reset_parameters()
        self.head.reset_parameters()
        if self.direct_residual is not None:
            self.direct_residual.reset_parameters()
        if self.simplex_router is not None:
            self.simplex_router.reset_parameters()
        if self.ccp_private_head is not None:
            nn.init.zeros_(self.ccp_private_head.weight)
            nn.init.zeros_(self.ccp_private_head.bias)
        if self.low_rank_head_residual is not None:
            self.low_rank_head_residual.reset_parameters()
        if self.depth2_head_residual is not None:
            self.depth2_head_residual.reset_parameters()
        if self.givens_head_residual is not None:
            self.givens_head_residual.reset_parameters()
        if self.posth_low_rank_residual is not None:
            self.posth_low_rank_residual.reset_parameters()
        if self.pdr_residual is not None:
            self.pdr_residual.reset_parameters()
        if self.clustered_h_residual is not None:
            self.clustered_h_residual.reset_parameters()
        if self.role_film_residual is not None:
            self.role_film_residual.reset_parameters()
        if self.full_action_residual is not None:
            self.full_action_residual.reset_parameters()
        if self.content_conditioned_action_residual is not None:
            self.content_conditioned_action_residual.reset_parameters()

    def __call__(
        self,
        packed_f: Tensor,
        *,
        use_compiled: bool = True,
        use_custom_vjp: bool = False,
    ) -> Tensor:
        if packed_f.ndim != 4:
            raise ValueError("packed_f must have shape [B,2D,H,W]")
        if packed_f.shape[1] != 2 * self.state_channels:
            raise ValueError(
                "packed_f channel dimension must equal 2 * state_channels"
            )
        if not packed_f.is_floating_point():
            raise TypeError("packed_f must use a real floating dtype")
        generator = self.generator
        if self.g_adapter is not None:
            (
                mix_m_real,
                mix_m_imag,
                mix_n_real,
                mix_n_imag,
            ) = self.g_adapter.effective_matrices(generator)
        elif generator.mix_n_real is None:
            mix_m_real = generator.mix_m_real
            mix_m_imag = generator.mix_m_imag
            mix_n_real = torch.zeros_like(generator.mix_m_real)
            mix_n_imag = torch.zeros_like(generator.mix_m_imag)
        else:
            mix_m_real = generator.mix_m_real
            mix_m_imag = generator.mix_m_imag
            mix_n_real = generator.mix_n_real
            mix_n_imag = generator.mix_n_imag
        depthwise_real = _q9_transform_gain_chart(
            generator.depthwise_real,
            gain_chart=generator.gain_chart,
        )
        depthwise_imag = _q9_transform_gain_chart(
            generator.depthwise_imag,
            gain_chart=generator.gain_chart,
        )
        head_weight, head_bias = self.effective_head_parameters()
        tensors = (
            packed_f,
            mix_m_real,
            mix_m_imag,
            mix_n_real,
            mix_n_imag,
            depthwise_real,
            depthwise_imag,
            head_weight,
            head_bias,
        )
        if generator.spatial_operator == "gchaf_r1":
            if generator.role_gain_real is None or generator.role_gain_imag is None:
                raise RuntimeError("GCHAF-R1 shared role gains are missing")
            tensors = (
                *tensors,
                generator.role_gain_real,
                generator.role_gain_imag,
                depthwise_real.new_tensor(self.block_role_code),
            )
        if generator.spatial_operator == "gchaf_ar9f":
            if self.full_action_residual is None:
                raise RuntimeError("GCHAF-AR9F full action residual is missing")
            tensors = (*tensors, self.full_action_residual.action_weight)
        if generator.spatial_operator == "gchaf_ccar9":
            if self.content_conditioned_action_residual is None:
                raise RuntimeError(
                    "GCHAF-CCAR9 content-conditioned action residual is missing"
                )
            tensors = (
                *tensors,
                self.content_conditioned_action_residual.content_action_weight,
            )
        if generator.spatial_operator == "dct_hybrid_learned_router":
            if self.simplex_router is None:
                raise RuntimeError("learned DH router binding is missing")
            tensors = (
                *tensors,
                self.simplex_router.weight,
                self.simplex_router.bias,
            )
        if generator.spatial_operator in {
            "dh_lms4_additive",
            "dh_lms4_signed",
            "dh_lms4_act4",
            "dh_lms4_dual_route",
            "dh_mwf9_independent",
            "dh_swf9_independent",
            "dh_swf9_scale_tied",
            "dh_swf9_scale_rank2",
        }:
            if generator.gamma_real is None or generator.gamma_imag is None:
                raise RuntimeError("independent DH gamma binding is missing")
            tensors = (
                *tensors,
                generator.gamma_real,
                generator.gamma_imag,
            )
        if generator.spatial_operator in {"dh_lms4_act4", "dh_lms4_dual_route"}:
            if generator.eta_real is None or generator.eta_imag is None:
                raise RuntimeError("DH-LMS4-ACT4 eta binding is missing")
            tensors = (*tensors, generator.eta_real, generator.eta_imag)
        if self.role_film_residual is not None:
            if use_custom_vjp:
                raise ValueError(
                    "GLRC9 role-FiLM uses its exact native autograd graph; "
                    "the local convolution custom VJP is not applicable"
                )
            role_film_tensors = (
                *tensors,
                self.role_film_residual.a,
                self.role_film_residual.b,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_glrc9_role_film_operator(
                    role_film_tensors
                )
            else:
                raw = _fused_q9_glrc9_role_film_operator_graph(
                    *role_film_tensors
                )
        elif self.clustered_h_residual is not None:
            if use_custom_vjp:
                raise ValueError(
                    "GLRC9 clustered H uses its exact native autograd graph; "
                    "the local convolution custom VJP is not applicable"
                )
            clustered_tensors = (*tensors, self.clustered_h_residual.weight)
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_glrc9_clustered_h_operator(
                    clustered_tensors
                )
            else:
                raw = _fused_q9_glrc9_clustered_h_operator_graph(
                    *clustered_tensors
                )
        elif self.pdr_residual is not None:
            if use_custom_vjp:
                raise ValueError(
                    "DH-PDR uses its exact native autograd graph; the local "
                    "convolution custom VJP is not applicable"
                )
            pdr_tensors = (
                *tensors,
                self.pdr_residual.v.weight,
                self.pdr_residual.u.weight,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_dct_hybrid_pdr_operator(
                    pdr_tensors
                )
            else:
                raw = _fused_q9_dct_hybrid_pdr_operator_graph(*pdr_tensors)
        elif self.givens_head_residual is not None:
            if use_custom_vjp:
                raise ValueError(
                    "DH-H2-Givens uses its exact native autograd graph; "
                    "the local convolution custom VJP is not applicable"
                )
            givens_tensors = (
                *tensors,
                self.givens_head_residual.amplitude_gain,
                self.givens_head_residual.horizontal_gain,
                self.givens_head_residual.vertical_gain,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_dct_hybrid_h2_givens_operator(
                    givens_tensors
                )
            else:
                raw = _fused_q9_dct_hybrid_h2_givens_operator_graph(
                    *givens_tensors
                )
        elif self.depth2_head_residual is not None:
            if use_custom_vjp:
                raise ValueError(
                    "DH-H2 uses its exact native autograd graph; the local "
                    "convolution custom VJP is not applicable"
                )
            h2_tensors = (
                *tensors,
                self.depth2_head_residual.refine.weight,
                self.depth2_head_residual.gain,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_dct_hybrid_h2_operator(h2_tensors)
            else:
                raw = _fused_q9_dct_hybrid_h2_operator_graph(*h2_tensors)
        elif self.low_rank_head_residual is not None:
            if use_custom_vjp:
                raise ValueError(
                    "DH-HLR uses its exact native autograd graph; the local "
                    "convolution custom VJP is not applicable"
                )
            hlr_tensors = (
                *tensors,
                self.low_rank_head_residual.down.weight,
                self.low_rank_head_residual.up.weight,
                self.low_rank_head_residual.gate_logit,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_dct_hybrid_hlr_operator(hlr_tensors)
            else:
                raw = _fused_q9_dct_hybrid_hlr_operator_graph(*hlr_tensors)
        elif generator.spatial_operator == "nk9":
            if use_custom_vjp:
                raise ValueError(
                    "nk9 uses its exact native autograd graph; "
                    "the local-convolution custom VJP is not applicable"
                )
            raw = _fused_q9_nk9_operator_graph(*tensors)
        elif generator.spatial_operator == "phase_gauge":
            # The input-dependent unitary gauge changes the exact Jacobian;
            # use its native autograd graph rather than the local hand VJP.
            if use_custom_vjp:
                raise ValueError(
                    "phase_gauge uses its exact native autograd graph; "
                    "the local-convolution custom VJP is not applicable"
                )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_phase_gauge_compiler(tensors)
            else:
                raw = _fused_q9_phase_gauge_compiler_graph(*tensors)
        elif generator.spatial_operator in {
            "cosine_integral",
            "green_resolvent",
            "parseval_resolvent_directional",
            "dct_hybrid",
            "dct_hybrid_routing_continuation",
            "dct_hybrid_local_family_continuation",
            GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR,
            "gchaf_r1",
            "gchaf_ar9f",
            "gchaf_ccar9",
            "dct_global9_conditioned",
            "dh_lms4_additive",
            "dh_lms4_signed",
            "dh_lms4_act4",
            "dh_lms4_dual_route",
            "dh_mwf9",
            "dh_mwf9_independent",
            "dh_swf9_independent",
            "dh_swf9_scale_tied",
            "dh_swf9_scale_rank2",
            "dct_hybrid_learned_router",
            "glrc9_ccp",
            "dct_phase_cycle",
            "dct_dsl8",
            "polyphase_lifting",
        }:
            # The ordinary autograd graph is exact for both non-local
            # operators.  The hand-written local-convolution VJP is
            # deliberately not reused because it would represent a different
            # Jacobian.
            if use_custom_vjp:
                raise ValueError(
                    f"{generator.spatial_operator} uses its exact native "
                    "autograd graph; "
                    "the local-convolution custom VJP is not applicable"
                )
            if (
                generator.spatial_operator
                in {
                    "green_resolvent",
                    "parseval_resolvent_directional",
                    "dct_hybrid",
                    "dct_hybrid_routing_continuation",
                    "dct_hybrid_local_family_continuation",
                    GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR,
                    "gchaf_r1",
                    "gchaf_ar9f",
                    "gchaf_ccar9",
                    "dct_global9_conditioned",
                    "dh_lms4_additive",
                    "dh_lms4_signed",
                    "dh_lms4_act4",
                    "dh_lms4_dual_route",
                    "dh_mwf9",
                    "dh_mwf9_independent",
                    "dh_swf9_independent",
                    "dh_swf9_scale_tied",
                    "dh_swf9_scale_rank2",
                    "dct_hybrid_learned_router",
                    "glrc9_ccp",
                    "dct_phase_cycle",
                    "dct_dsl8",
                    "polyphase_lifting",
                }
                and use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
                and not (
                    generator.spatial_operator == "gchaf_ccar9"
                    and _CCAR9_CONTENT_BATCH_PERMUTATION_CONTEXT is not None
                )
            ):
                if generator.spatial_operator == "glrc9_ccp":
                    if self.ccp_private_head is None:
                        raw = _compiled_fused_q9_dct_hybrid_operator(tensors)
                    else:
                        raw = _compiled_fused_q9_glrc9_ccp_operator(
                            (
                                *tensors,
                                self.ccp_private_head.weight,
                                self.ccp_private_head.bias,
                            )
                        )
                elif generator.spatial_operator == "parseval_resolvent_directional":
                    raw = _compiled_fused_q9_parseval_resolvent_directional_operator(
                        tensors
                    )
                elif generator.spatial_operator == "green_resolvent":
                    raw = _compiled_fused_q9_green_resolvent_operator(tensors)
                elif generator.spatial_operator == "dct_hybrid":
                    if self.centered_state_energy_bias:
                        raw = _compiled_fused_q9_dct_hybrid_cseb_operator(
                            tensors
                        )
                    else:
                        raw = _compiled_fused_q9_dct_hybrid_operator(tensors)
                elif (
                    generator.spatial_operator
                    == "dct_hybrid_routing_continuation"
                ):
                    if self.routing_continuation is None:
                        raise RuntimeError(
                            "GLRC9 routing continuation scalar binding is missing"
                        )
                    # g=1 is the exact GLRC9 parent endpoint.  Dispatch to the
                    # ordinary parent graph instead of evaluating an affine
                    # continuation graph whose mathematically-zero branch can
                    # still accumulate FP32 roundoff through K12.  The scalar
                    # is fixed by the production path and is not optimized.
                    if torch.equal(
                        self.routing_continuation.detach().reshape(()),
                        self.routing_continuation.new_tensor(1.0),
                    ):
                        raw = _compiled_fused_q9_dct_hybrid_operator(tensors)
                    else:
                        raw = _compiled_fused_q9_dct_hybrid_routing_continuation_operator(
                            (*tensors, self.routing_continuation)
                        )
                elif (
                    generator.spatial_operator
                    == "dct_hybrid_local_family_continuation"
                ):
                    if self.local_family_continuation is None:
                        raise RuntimeError(
                            "GLRC9 local-family continuation scalar binding is missing"
                        )
                    scalar = self.local_family_continuation.detach().reshape(())
                    if torch.equal(
                        scalar, self.local_family_continuation.new_tensor(1.0)
                    ):
                        raw = _compiled_fused_q9_dct_hybrid_operator(tensors)
                    elif torch.equal(
                        scalar, self.local_family_continuation.new_tensor(0.0)
                    ) and not self.local_family_force_materialized_reference:
                        raw = _compiled_fused_q9_dct_hybrid_no_local_operator(tensors)
                    else:
                        raw = _compiled_fused_q9_dct_hybrid_local_family_continuation_operator(
                            (*tensors, self.local_family_continuation)
                        )
                elif (
                    generator.spatial_operator
                    == GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR
                ):
                    if self.mn_offdiag_continuation is None:
                        raise RuntimeError(
                            "GLRC9 M/N continuation scalar binding is missing"
                        )
                    scalar = self.mn_offdiag_continuation.detach().reshape(())
                    if torch.equal(
                        scalar, self.mn_offdiag_continuation.new_tensor(1.0)
                    ):
                        raw = _compiled_fused_q9_dct_hybrid_operator(tensors)
                    elif (
                        torch.equal(
                            scalar, self.mn_offdiag_continuation.new_tensor(0.0)
                        )
                        and self.mn_offdiag_force_materialized_reference
                    ):
                        raw = _compiled_fused_q9_dct_hybrid_mn_offdiag_materialized_reference_operator(
                            tensors
                        )
                    elif torch.equal(
                        scalar, self.mn_offdiag_continuation.new_tensor(0.0)
                    ):
                        raw = _compiled_fused_q9_dct_hybrid_mn_offdiag_diagonal_operator(
                            tensors
                        )
                    else:
                        raw = _compiled_fused_q9_dct_hybrid_mn_offdiag_continuation_operator(
                            (*tensors, self.mn_offdiag_continuation)
                        )
                elif generator.spatial_operator == "gchaf_r1":
                    raw = _compiled_fused_q9_gchaf_r1_operator(tensors)
                elif generator.spatial_operator == "gchaf_ar9f":
                    raw = _compiled_fused_q9_gchaf_ar9f_operator(tensors)
                elif generator.spatial_operator == "gchaf_ccar9":
                    raw = _compiled_fused_q9_gchaf_ccar9_operator(tensors)
                elif generator.spatial_operator == "dct_global9_conditioned":
                    raw = _compiled_fused_q9_dct_global9_conditioned_operator(
                        tensors
                    )
                elif generator.spatial_operator == "dh_lms4_additive":
                    raw = _compiled_fused_q9_dh_lms4_additive_operator(
                        tensors
                    )
                elif generator.spatial_operator == "dh_lms4_signed":
                    raw = _compiled_fused_q9_dh_lms4_signed_operator(tensors)
                elif generator.spatial_operator == "dh_lms4_act4":
                    raw = _compiled_fused_q9_dh_lms4_act4_operator(tensors)
                elif generator.spatial_operator == "dh_lms4_dual_route":
                    raw = _compiled_fused_q9_dh_lms4_dual_route_operator(tensors)
                elif generator.spatial_operator == "dh_mwf9":
                    raw = _compiled_fused_q9_dh_mwf9_operator(tensors)
                elif generator.spatial_operator == "dh_mwf9_independent":
                    raw = _compiled_fused_q9_dh_mwf9_independent_operator(
                        tensors
                    )
                elif generator.spatial_operator == "dh_swf9_independent":
                    raw = _compiled_fused_q9_dh_swf9_independent_operator(
                        tensors
                    )
                elif generator.spatial_operator == "dh_swf9_scale_tied":
                    raw = _compiled_fused_q9_dh_swf9_scale_tied_operator(
                        tensors
                    )
                elif generator.spatial_operator == "dh_swf9_scale_rank2":
                    raw = _compiled_fused_q9_dh_swf9_scale_rank2_operator(
                        tensors
                    )
                elif (
                    generator.spatial_operator
                    == "dct_hybrid_learned_router"
                ):
                    raw = (
                        _compiled_fused_q9_dct_hybrid_learned_router_operator(
                            tensors
                        )
                    )
                elif generator.spatial_operator == "dct_phase_cycle":
                    raw = _compiled_fused_q9_dct_phase_cycle_operator(tensors)
                elif generator.spatial_operator == "polyphase_lifting":
                    raw = _compiled_fused_q9_polyphase_lifting_operator(
                        tensors
                    )
                else:
                    raw = _compiled_fused_q9_dct_dsl8_operator(tensors)
            elif generator.spatial_operator == "glrc9_ccp":
                if self.ccp_private_head is None:
                    raw = _fused_q9_dct_hybrid_operator_graph(*tensors)
                else:
                    raw = _fused_q9_glrc9_ccp_operator_graph(
                        *tensors,
                        self.ccp_private_head.weight,
                        self.ccp_private_head.bias,
                    )
            elif (
                generator.spatial_operator == "parseval_resolvent_directional"
            ):
                raw = _fused_q9_parseval_resolvent_directional_operator_graph(
                    *tensors
                )
            elif generator.spatial_operator == "green_resolvent":
                raw = _fused_q9_green_resolvent_operator_graph(*tensors)
            elif generator.spatial_operator == "dct_hybrid":
                if self.centered_state_energy_bias:
                    raw = _fused_q9_dct_hybrid_cseb_operator_graph(*tensors)
                else:
                    raw = _fused_q9_dct_hybrid_operator_graph(*tensors)
            elif (
                generator.spatial_operator
                == "dct_hybrid_routing_continuation"
            ):
                if self.routing_continuation is None:
                    raise RuntimeError(
                        "GLRC9 routing continuation scalar binding is missing"
                    )
                if torch.equal(
                    self.routing_continuation.detach().reshape(()),
                    self.routing_continuation.new_tensor(1.0),
                ):
                    raw = _fused_q9_dct_hybrid_operator_graph(*tensors)
                else:
                    raw = _fused_q9_dct_hybrid_operator_graph(
                        *tensors,
                        self.routing_continuation,
                    )
            elif (
                generator.spatial_operator
                == "dct_hybrid_local_family_continuation"
            ):
                if self.local_family_continuation is None:
                    raise RuntimeError(
                        "GLRC9 local-family continuation scalar binding is missing"
                    )
                scalar = self.local_family_continuation.detach().reshape(())
                if torch.equal(
                    scalar, self.local_family_continuation.new_tensor(1.0)
                ):
                    raw = _fused_q9_dct_hybrid_operator_graph(*tensors)
                elif torch.equal(
                    scalar, self.local_family_continuation.new_tensor(0.0)
                ) and not self.local_family_force_materialized_reference:
                    raw = _fused_q9_dct_hybrid_no_local_operator_graph(*tensors)
                else:
                    raw = _fused_q9_dct_hybrid_local_family_continuation_operator_graph(
                        *tensors, self.local_family_continuation
                    )
            elif (
                generator.spatial_operator
                == GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR
            ):
                if self.mn_offdiag_continuation is None:
                    raise RuntimeError("GLRC9 M/N continuation scalar binding is missing")
                scalar = self.mn_offdiag_continuation.detach().reshape(())
                if torch.equal(
                    scalar, self.mn_offdiag_continuation.new_tensor(1.0)
                ):
                    raw = _fused_q9_dct_hybrid_operator_graph(*tensors)
                elif (
                    torch.equal(
                        scalar, self.mn_offdiag_continuation.new_tensor(0.0)
                    )
                    and self.mn_offdiag_force_materialized_reference
                ):
                    raw = _fused_q9_dct_hybrid_mn_offdiag_materialized_reference_operator_graph(
                        *tensors
                    )
                elif torch.equal(
                    scalar, self.mn_offdiag_continuation.new_tensor(0.0)
                ):
                    raw = _fused_q9_dct_hybrid_mn_offdiag_diagonal_operator_graph(
                        *tensors
                    )
                else:
                    raw = _fused_q9_dct_hybrid_mn_offdiag_continuation_operator_graph(
                        *tensors, self.mn_offdiag_continuation
                    )
            elif generator.spatial_operator == "gchaf_r1":
                raw = _fused_q9_gchaf_r1_operator_graph(*tensors)
            elif generator.spatial_operator == "gchaf_ar9f":
                raw = _fused_q9_gchaf_ar9f_operator_graph(*tensors)
            elif generator.spatial_operator == "gchaf_ccar9":
                raw = _fused_q9_gchaf_ccar9_operator_graph(*tensors)
            elif generator.spatial_operator == "dct_global9_conditioned":
                raw = _fused_q9_dct_global9_conditioned_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_lms4_additive":
                raw = _fused_q9_dh_lms4_additive_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_lms4_signed":
                raw = _fused_q9_dh_lms4_signed_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_lms4_act4":
                raw = _fused_q9_dh_lms4_act4_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_lms4_dual_route":
                raw = _fused_q9_dh_lms4_dual_route_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_mwf9":
                raw = _fused_q9_dh_mwf9_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_mwf9_independent":
                raw = _fused_q9_dh_mwf9_independent_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_swf9_independent":
                raw = _fused_q9_dh_swf9_independent_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_swf9_scale_tied":
                raw = _fused_q9_dh_swf9_scale_tied_operator_graph(*tensors)
            elif generator.spatial_operator == "dh_swf9_scale_rank2":
                raw = _fused_q9_dh_swf9_scale_rank2_operator_graph(*tensors)
            elif generator.spatial_operator == "dct_hybrid_learned_router":
                raw = _fused_q9_dct_hybrid_learned_router_operator_graph(
                    *tensors
                )
            elif generator.spatial_operator == "dct_phase_cycle":
                raw = _fused_q9_dct_phase_cycle_operator_graph(*tensors)
            elif generator.spatial_operator == "dct_dsl8":
                raw = _fused_q9_dct_dsl8_operator_graph(*tensors)
            elif generator.spatial_operator == "polyphase_lifting":
                raw = _fused_q9_polyphase_lifting_operator_graph(*tensors)
            elif (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_fused_q9_cosine_operator(tensors)
            else:
                raw = _fused_q9_cosine_operator_graph(*tensors)
        elif use_custom_vjp:
            raw = _FusedQ9CompilerFunction.apply(*tensors, use_compiled)
        elif (
            use_compiled
            and packed_f.is_cuda
            and packed_f.dtype == torch.float32
        ):
            raw = _compiled_fused_q9_compiler(tensors)
        else:
            raw = _fused_q9_compiler_graph(*tensors)
        if self.direct_residual is not None:
            raw = raw + self.direct_residual(packed_f)
        if self.posth_low_rank_residual is not None:
            raw = raw + self.posth_low_rank_residual(packed_f)
        if self.moment_exponential:
            coordinate_major = _q9_grouped_head_to_coordinate_major(
                raw,
                self.state_channels,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_q9_moment_exponential_raw(
                    coordinate_major,
                    self.rho_max,
                )
            else:
                raw = _q9_moment_exponential_raw(
                    coordinate_major,
                    self.rho_max,
                )
            raw = raw.reshape(
                packed_f.shape[0],
                9 * self.state_channels,
                packed_f.shape[2],
                packed_f.shape[3],
            )
        elif self.logpolar:
            coordinate_major = _q9_grouped_head_to_coordinate_major(
                raw,
                self.state_channels,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_q9_logpolar_raw(coordinate_major)
            else:
                raw = _q9_logpolar_raw(coordinate_major)
            raw = raw.reshape(
                packed_f.shape[0],
                9 * self.state_channels,
                packed_f.shape[2],
                packed_f.shape[3],
            )
        elif self.mass_simplex_phase:
            coordinate_major = _q9_grouped_head_to_coordinate_major(
                raw,
                self.state_channels,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_q9_mass_simplex_phase_raw(coordinate_major)
            else:
                raw = _q9_mass_simplex_phase_raw(coordinate_major)
            raw = raw.reshape(
                packed_f.shape[0],
                9 * self.state_channels,
                packed_f.shape[2],
                packed_f.shape[3],
            )
        elif self.center_mass_inverse:
            coordinate_major = _q9_grouped_head_to_coordinate_major(
                raw,
                self.state_channels,
            )
            if (
                use_compiled
                and packed_f.is_cuda
                and packed_f.dtype == torch.float32
            ):
                raw = _compiled_q9_center_mass_inverse_raw(
                    coordinate_major,
                    self.rho_max,
                )
            else:
                raw = _q9_center_mass_inverse_raw(
                    coordinate_major,
                    self.rho_max,
                )
            raw = raw.reshape(
                packed_f.shape[0],
                9 * self.state_channels,
                packed_f.shape[2],
                packed_f.shape[3],
            )
        elif self.tap_major_output:
            raw = _q9_grouped_head_to_tap_major(raw, self.state_channels)
        return raw
