"""Static-shape compiled F9 step for Selective State-Depth CLPO."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from .fused_stencil import (
    F9_OFFSETS,
    _clone_graph_code,
    _shift_from_padded,
    _tensor_compile_signatures,
    f9_forward_graph,
)


def selective_f9_step_graph(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    batch, channels, height, width = propagated_r.shape
    groups = channels // group_size
    shape = (batch, groups, group_size, height, width)
    real_g = propagated_r.reshape(shape)
    imag_g = propagated_i.reshape(shape)
    strength_g = strength.reshape(shape).mean(dim=2, keepdim=True)
    group_energy = (
        real_g.square() + imag_g.square()
    ).mean(dim=2, keepdim=True)
    scale = torch.rsqrt(1.0 + strength_g * group_energy)
    state_r = drive_r + (real_g * scale).reshape_as(propagated_r)
    state_i = drive_i + (imag_g * scale).reshape_as(propagated_i)
    return f9_forward_graph(state_r, state_i, ar, ai)


def diagonal_lag1_f9_step_graph(
    propagated_r: Tensor,
    propagated_i: Tensor,
    previous_source_r: Tensor,
    previous_source_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Delay only the four diagonal donors by one state-depth step."""

    batch, channels, height, width = propagated_r.shape
    groups = channels // group_size
    shape = (batch, groups, group_size, height, width)
    real_g = propagated_r.reshape(shape)
    imag_g = propagated_i.reshape(shape)
    strength_g = strength.reshape(shape).mean(dim=2, keepdim=True)
    group_energy = (real_g.square() + imag_g.square()).mean(dim=2, keepdim=True)
    scale = torch.rsqrt(1.0 + strength_g * group_energy)
    source_r = drive_r + (real_g * scale).reshape_as(propagated_r)
    source_i = drive_i + (imag_g * scale).reshape_as(propagated_i)

    current_r = F.pad(source_r, (1, 1, 1, 1), value=0.0)
    current_i = F.pad(source_i, (1, 1, 1, 1), value=0.0)
    delayed_r = F.pad(previous_source_r, (1, 1, 1, 1), value=0.0)
    delayed_i = F.pad(previous_source_i, (1, 1, 1, 1), value=0.0)
    out_r, out_i = torch.zeros_like(source_r), torch.zeros_like(source_i)
    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        padded_r = delayed_r if tap in (1, 3, 6, 8) else current_r
        padded_i = delayed_i if tap in (1, 3, 6, 8) else current_i
        shifted_r = _shift_from_padded(padded_r, dy, dx, height, width)
        shifted_i = _shift_from_padded(padded_i, dy, dx, height, width)
        tap_r, tap_i = ar[:, tap], ai[:, tap]
        out_r = out_r + tap_r * shifted_r - tap_i * shifted_i
        out_i = out_i + tap_r * shifted_i + tap_i * shifted_r
    return out_r, out_i, source_r, source_i


def group_channel_radial_f9_step_graph(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """Interpolate group-shared and channel-wise radial memory selection."""

    batch, channels, height, width = propagated_r.shape
    groups = channels // group_size
    shape = (batch, groups, group_size, height, width)
    real_g = propagated_r.reshape(shape)
    imag_g = propagated_i.reshape(shape)
    strength_c = strength.reshape(shape)
    channel_energy = real_g.square() + imag_g.square()
    strength_g = strength_c.mean(dim=2, keepdim=True)
    group_energy = channel_energy.mean(dim=2, keepdim=True)
    group_scale = torch.rsqrt(1.0 + strength_g * group_energy)
    channel_scale = torch.rsqrt(1.0 + strength_c * channel_energy)
    scale = group_scale + eta * (channel_scale - group_scale)
    state_r = drive_r + (real_g * scale).reshape_as(propagated_r)
    state_i = drive_i + (imag_g * scale).reshape_as(propagated_i)
    return f9_forward_graph(state_r, state_i, ar, ai)


def tap_discrepancy_f9_step_graph(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_transition_r: Tensor,
    drive_transition_i: Tensor,
    tap_strength: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """Apply one state-dependent F9 memory transition.

    The center tap uses state RMS as an exponential forget gate.  Each
    neighbour tap uses the complex discrepancy between its shifted donor and
    the current receiver.  All gates lie in (0, 1], so multiplying the already
    projected F9 coefficients cannot increase their row mass.
    """

    batch, channels, height, width = propagated_r.shape
    groups = channels // group_size
    grouped_shape = (batch, groups, group_size, height, width)
    receiver_r = propagated_r.reshape(grouped_shape)
    receiver_i = propagated_i.reshape(grouped_shape)
    padded_r = F.pad(propagated_r, (1, 1, 1, 1), value=0.0)
    padded_i = F.pad(propagated_i, (1, 1, 1, 1), value=0.0)
    out_r, out_i = drive_transition_r, drive_transition_i

    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        shifted_r = _shift_from_padded(
            padded_r, dy, dx, height, width
        ).reshape(grouped_shape)
        shifted_i = _shift_from_padded(
            padded_i, dy, dx, height, width
        ).reshape(grouped_shape)
        if tap == 0:
            measure = (
                receiver_r.square() + receiver_i.square()
            ).mean(dim=2, keepdim=True)
        else:
            measure = (
                (shifted_r - receiver_r).square()
                + (shifted_i - receiver_i).square()
            ).mean(dim=2, keepdim=True)
        strength = tap_strength[:, tap].unsqueeze(2)
        gate = torch.exp(-strength * torch.sqrt(measure + 1.0e-8))
        tap_r = ar[:, tap].reshape(grouped_shape)
        tap_i = ai[:, tap].reshape(grouped_shape)
        out_r = out_r + (
            gate * (tap_r * shifted_r - tap_i * shifted_i)
        ).reshape_as(propagated_r)
        out_i = out_i + (
            gate * (tap_r * shifted_i + tap_i * shifted_r)
        ).reshape_as(propagated_i)
    return out_r, out_i


def _k4_pair_mix(value: Tensor, eta: Tensor, family: int) -> Tensor:
    """Apply one signed, damped K4 perfect matching without a matrix."""

    batch, channels, height, width = value.shape
    groups = channels // 4
    x0, x1, x2, x3 = value.reshape(
        batch, groups, 4, height, width
    ).unbind(dim=2)

    def pair(left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        correction = eta * (left - right)
        return left - correction, right + correction

    if family == 0:  # horizontal: (0,1), (2,3)
        y0, y1 = pair(x0, x1)
        y2, y3 = pair(x2, x3)
    elif family == 1:  # vertical: (0,2), (1,3)
        y0, y2 = pair(x0, x2)
        y1, y3 = pair(x1, x3)
    elif family == 2:  # diagonal: (0,3), (1,2)
        y0, y3 = pair(x0, x3)
        y1, y2 = pair(x1, x2)
    else:
        raise ValueError("K4 paired-Householder family must be 0, 1, or 2")
    return torch.stack((y0, y1, y2, y3), dim=2).reshape_as(value)


def _k4_pair_mix_complex(value: Tensor, eta: Tensor, family: int) -> Tensor:
    """Mix real and imaginary K4 lanes in one packed channel graph."""

    batch, channels, height, width = value.shape
    groups = channels // 8
    grouped = value.reshape(batch, 2, groups, 4, height, width)
    x0, x1, x2, x3 = grouped.unbind(dim=3)
    if family == 0:  # horizontal: (0,1), (2,3)
        partner = torch.stack((x1, x0, x3, x2), dim=3)
    elif family == 1:  # vertical: (0,2), (1,3)
        partner = torch.stack((x2, x3, x0, x1), dim=3)
    elif family == 2:  # diagonal: (0,3), (1,2)
        partner = torch.stack((x3, x2, x1, x0), dim=3)
    else:
        raise ValueError("K4 paired-Householder family must be 0, 1, or 2")
    eta = eta[:, None, :, None]
    return (grouped - eta * (grouped - partner)).reshape_as(value)


_K4_TAP_FAMILIES = (-1, 2, 1, 2, 0, 0, 2, 1, 2)


def _k4_paired_householder_f9_step_graph_packed(
    state: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run the paired F9 graph with real/imag lanes packed for one stencil pass."""

    state_r, state_i = state.chunk(2, dim=1)
    height, width = state.shape[-2:]
    padded = F.pad(state, (1, 1, 1, 1), value=0.0)
    out_r, out_i = torch.zeros_like(state_r), torch.zeros_like(state_i)

    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        shifted = _shift_from_padded(padded, dy, dx, height, width)
        family = _K4_TAP_FAMILIES[tap]
        if family >= 0:
            shifted = _k4_pair_mix_complex(shifted, eta[:, family], family)
        shifted_r, shifted_i = shifted.chunk(2, dim=1)
        tap_r, tap_i = ar[:, tap], ai[:, tap]
        out_r = out_r + tap_r * shifted_r - tap_i * shifted_i
        out_i = out_i + tap_r * shifted_i + tap_i * shifted_r
    return out_r, out_i


def k4_paired_householder_f9_step_graph(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """F9 step with one family-shared damped Householder per K4 matching.

    Eta has shape ``[B,3,D/4,H,W]`` and is compiled once by the caller.  The
    center tap remains diagonal; H/V/diagonal taps use the three commuting
    perfect matchings of K4.  Each pair map has absolute row mass and spectral
    norm one for eta in [0, 1], so the projected F9 budget is unchanged.  The
    complete edge maps remain content-dependent because their diagonal F9
    coefficients generally do not commute with these channel matchings.
    """

    state_r = drive_r + propagated_r
    state_i = drive_i + propagated_i
    height, width = state_r.shape[-2:]
    padded_r = F.pad(state_r, (1, 1, 1, 1), value=0.0)
    padded_i = F.pad(state_i, (1, 1, 1, 1), value=0.0)
    out_r, out_i = torch.zeros_like(state_r), torch.zeros_like(state_i)
    tap_families = (-1, 2, 1, 2, 0, 0, 2, 1, 2)

    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        shifted_r = _shift_from_padded(padded_r, dy, dx, height, width)
        shifted_i = _shift_from_padded(padded_i, dy, dx, height, width)
        family = tap_families[tap]
        if family >= 0:
            family_eta = eta[:, family]
            shifted_r = _k4_pair_mix(shifted_r, family_eta, family)
            shifted_i = _k4_pair_mix(shifted_i, family_eta, family)
        tap_r, tap_i = ar[:, tap], ai[:, tap]
        out_r = out_r + tap_r * shifted_r - tap_i * shifted_i
        out_i = out_i + tap_r * shifted_i + tap_i * shifted_r
    return out_r, out_i


def k4_radial_f9_step_graph(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply radial state selection before the K4 paired F9 transition."""

    batch, channels, height, width = propagated_r.shape
    grouped_shape = (batch, channels // 4, 4, height, width)
    real_g = propagated_r.reshape(grouped_shape)
    imag_g = propagated_i.reshape(grouped_shape)
    strength_g = strength.reshape(grouped_shape).mean(dim=2, keepdim=True)
    group_energy = (real_g.square() + imag_g.square()).mean(dim=2, keepdim=True)
    scale = torch.rsqrt(1.0 + strength_g * group_energy)
    memory_r = (real_g * scale).reshape_as(propagated_r)
    memory_i = (imag_g * scale).reshape_as(propagated_i)
    return k4_paired_householder_f9_step_graph(
        memory_r, memory_i, drive_r, drive_i, eta, ar, ai
    )


def _k4_radial_f9_step_graph_packed(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compiled radial graph with the complex lanes packed through F9."""

    batch, channels, height, width = propagated_r.shape
    grouped_shape = (batch, channels // 4, 4, height, width)
    real_g = propagated_r.reshape(grouped_shape)
    imag_g = propagated_i.reshape(grouped_shape)
    strength_g = strength.reshape(grouped_shape).mean(dim=2, keepdim=True)
    group_energy = (real_g.square() + imag_g.square()).mean(dim=2, keepdim=True)
    scale = torch.rsqrt(1.0 + strength_g * group_energy)
    state = torch.cat(
        (
            drive_r + (real_g * scale).reshape_as(propagated_r),
            drive_i + (imag_g * scale).reshape_as(propagated_i),
        ),
        dim=1,
    )
    return _k4_paired_householder_f9_step_graph_packed(state, eta, ar, ai)


_CACHE: dict[tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]] = {}
_DIAGONAL_LAG1_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_GROUP_CHANNEL_RADIAL_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_TAP_CACHE: dict[tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]] = {}
_K4_PAIR_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_K4_RADIAL_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_IDENTITY_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_REAL_RADIAL_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_STEP_ROLLOUT_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_STATE_DYNAMIC_ROLLOUT_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}
_FEEDFORWARD_ROLLOUT_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}


def identity_memory_f9_step_graph(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """F9 step with identity memory instead of radial state compression."""

    return f9_forward_graph(
        drive_r + propagated_r,
        drive_i + propagated_i,
        ar,
        ai,
    )


def real_pair_radial_f9_step_graph(
    state_a: Tensor,
    state_b: Tensor,
    drive_a: Tensor,
    drive_b: Tensor,
    strength_a: Tensor,
    strength_b: Tensor,
    coeff_a: Tensor,
    coeff_b: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """Two independent real F9 states with the same radial-memory contract."""

    batch, channels, height, width = state_a.shape
    groups = channels // group_size
    shape = (batch, groups, group_size, height, width)

    def selected(value: Tensor, strength: Tensor) -> Tensor:
        value_g = value.reshape(shape)
        strength_g = strength.reshape(shape).mean(dim=2, keepdim=True)
        energy = value_g.square().mean(dim=2, keepdim=True)
        return (value_g * torch.rsqrt(1.0 + strength_g * energy)).reshape_as(value)

    value_a = drive_a + selected(state_a, strength_a)
    value_b = drive_b + selected(state_b, strength_b)
    padded_a = F.pad(value_a, (1, 1, 1, 1), mode="constant", value=0.0)
    padded_b = F.pad(value_b, (1, 1, 1, 1), mode="constant", value=0.0)
    out_a, out_b = torch.zeros_like(value_a), torch.zeros_like(value_b)
    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        shifted_a = _shift_from_padded(padded_a, dy, dx, height, width)
        shifted_b = _shift_from_padded(padded_b, dy, dx, height, width)
        out_a = out_a + coeff_a[:, tap] * shifted_a
        out_b = out_b + coeff_b[:, tap] * shifted_b
    return out_a, out_b


def _step_conditioned_raw_fields(
    hidden: Tensor,
    coefficient_weight: Tensor,
    coefficient_bias: Tensor,
    step_embedding: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    groups = hidden.shape[1] // group_size
    native = F.conv2d(hidden, coefficient_weight, coefficient_bias, groups=groups)
    batch, _, height, width = native.shape
    base = native.view(
        batch, groups, 2, 9, group_size, height, width
    ).permute(0, 2, 3, 1, 4, 5, 6).reshape(
        batch, 18 * hidden.shape[1], height, width
    )
    outputs_per_group = coefficient_weight.shape[0] // groups
    weight = coefficient_weight.reshape(groups, outputs_per_group, group_size)
    embedded = step_embedding.reshape(step_embedding.shape[0], groups, group_size)
    offsets = torch.einsum("sgi,goi->sgo", embedded, weight).reshape(
        step_embedding.shape[0], coefficient_weight.shape[0], 1, 1
    )
    offsets = offsets.view(
        step_embedding.shape[0], groups, 2, 9, group_size, 1, 1
    ).permute(0, 2, 3, 1, 4, 5, 6).reshape(
        step_embedding.shape[0], 18 * hidden.shape[1], 1, 1
    )
    return base, offsets


def _project_cartesian_f9(raw: Tensor, rho_max: float) -> tuple[Tensor, Tensor]:
    batch, _, height, width = raw.shape
    channels = raw.shape[1] // 18
    raw_real, raw_imag = raw.view(
        batch, 2, 9, channels, height, width
    ).unbind(dim=1)
    magnitude = torch.sqrt(raw_real.square() + raw_imag.square() + 1.0e-12)
    scale = rho_max / (1.0 + magnitude.sum(dim=1, keepdim=True))
    return raw_real * scale, raw_imag * scale


def step_conditioned_rollout_graph(
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    hidden: Tensor,
    coefficient_weight: Tensor,
    coefficient_bias: Tensor,
    step_embedding: Tensor,
    depth_weights: Tensor,
    rho_max: float,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    raw_base, raw_offsets = _step_conditioned_raw_fields(
        hidden, coefficient_weight, coefficient_bias, step_embedding, group_size
    )
    state_r, state_i = torch.zeros_like(drive_r), torch.zeros_like(drive_i)
    profile_r, profile_i = torch.zeros_like(drive_r), torch.zeros_like(drive_i)
    for index in range(step_embedding.shape[0]):
        ar, ai = _project_cartesian_f9(
            raw_base + raw_offsets[index : index + 1], rho_max
        )
        next_r, next_i = selective_f9_step_graph(
            state_r, state_i, drive_r, drive_i, strength, ar, ai, group_size
        )
        weight = depth_weights[index]
        profile_r = profile_r + weight * (next_r - state_r)
        profile_i = profile_i + weight * (next_i - state_i)
        state_r, state_i = next_r, next_i
    return profile_r, profile_i


def state_dynamic_rollout_graph(
    packed_f: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    coefficient_input_weight: Tensor,
    coefficient_dw3_weight: Tensor,
    coefficient_dw3_bias: Tensor,
    coefficient_dw7_weight: Tensor,
    coefficient_dw7_bias: Tensor,
    coefficient_gamma: Tensor,
    coefficient_output_weight: Tensor,
    coefficient_output_bias: Tensor,
    depth_weights: Tensor,
    rho_max: float,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    """Recompile the shared local kernel from input plus current state."""

    batch, channels, height, width = drive_r.shape
    groups = channels // group_size
    state_r, state_i = torch.zeros_like(drive_r), torch.zeros_like(drive_i)
    profile_r, profile_i = torch.zeros_like(drive_r), torch.zeros_like(drive_i)
    for index in range(depth_weights.shape[0]):
        source = packed_f + torch.cat((state_r, state_i), dim=1)
        source_r, source_i = source.chunk(2, dim=1)
        source = torch.stack((source_r, source_i), dim=2).flatten(1, 2)
        hidden = F.conv2d(source, coefficient_input_weight, groups=groups)
        primary = F.gelu(
            F.conv2d(
                hidden,
                coefficient_dw3_weight,
                coefficient_dw3_bias,
                padding=1,
                groups=channels,
            )
        )
        secondary = F.gelu(
            F.conv2d(
                hidden,
                coefficient_dw7_weight,
                coefficient_dw7_bias,
                padding=3,
                groups=channels,
            )
        )
        hidden = primary + coefficient_gamma.view(1, -1, 1, 1) * secondary
        raw = F.conv2d(
            hidden,
            coefficient_output_weight,
            coefficient_output_bias,
            groups=groups,
        ).view(
            batch, groups, 2, 9, group_size, height, width
        ).permute(0, 2, 3, 1, 4, 5, 6).reshape(
            batch, 18 * channels, height, width
        )
        ar, ai = _project_cartesian_f9(raw, rho_max)
        next_r, next_i = selective_f9_step_graph(
            state_r, state_i, drive_r, drive_i, strength, ar, ai, group_size
        )
        weight = depth_weights[index]
        profile_r = profile_r + weight * (next_r - state_r)
        profile_i = profile_i + weight * (next_i - state_i)
        state_r, state_i = next_r, next_i
    return profile_r, profile_i


def matched_feedforward_rollout_graph(
    drive_r: Tensor,
    drive_i: Tensor,
    hidden: Tensor,
    coefficient_weight: Tensor,
    coefficient_bias: Tensor,
    step_embedding: Tensor,
    rho_max: float,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    raw_base, raw_offsets = _step_conditioned_raw_fields(
        hidden, coefficient_weight, coefficient_bias, step_embedding, group_size
    )
    state_r, state_i = drive_r, drive_i
    zero_r, zero_i = torch.zeros_like(drive_r), torch.zeros_like(drive_i)
    for index in range(step_embedding.shape[0]):
        ar, ai = _project_cartesian_f9(
            raw_base + raw_offsets[index : index + 1], rho_max
        )
        state_r, state_i = identity_memory_f9_step_graph(
            state_r, state_i, zero_r, zero_i, ar, ai
        )
    return state_r, state_i


def _compiled_conditioned_rollout(
    cache: dict[tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]],
    graph: Callable[..., tuple[Tensor, Tensor]],
    tensors: tuple[Tensor, ...],
    step_count: int,
    rho_max: float,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    key = (
        int(step_count),
        float(rho_max),
        int(group_size),
        _tensor_compile_signatures(tensors),
    )
    entry = cache.get(key)
    if entry is None:
        target = _clone_graph_code(
            graph, name=f"{graph.__name__}__static_signature_{len(cache)}"
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        cache[key] = entry
    return entry[0](*tensors, float(rho_max), int(group_size))


def compiled_step_conditioned_rollout(
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    hidden: Tensor,
    coefficient_weight: Tensor,
    coefficient_bias: Tensor,
    step_embedding: Tensor,
    depth_weights: Tensor,
    rho_max: float,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    tensors = (
        drive_r, drive_i, strength, hidden, coefficient_weight,
        coefficient_bias, step_embedding, depth_weights,
    )
    return _compiled_conditioned_rollout(
        _STEP_ROLLOUT_CACHE, step_conditioned_rollout_graph,
        tensors, step_embedding.shape[0], rho_max, group_size,
    )


def compiled_state_dynamic_rollout(
    packed_f: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    coefficient_input_weight: Tensor,
    coefficient_dw3_weight: Tensor,
    coefficient_dw3_bias: Tensor,
    coefficient_dw7_weight: Tensor,
    coefficient_dw7_bias: Tensor,
    coefficient_gamma: Tensor,
    coefficient_output_weight: Tensor,
    coefficient_output_bias: Tensor,
    depth_weights: Tensor,
    rho_max: float,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    tensors = (
        packed_f,
        drive_r,
        drive_i,
        strength,
        coefficient_input_weight,
        coefficient_dw3_weight,
        coefficient_dw3_bias,
        coefficient_dw7_weight,
        coefficient_dw7_bias,
        coefficient_gamma,
        coefficient_output_weight,
        coefficient_output_bias,
        depth_weights,
    )
    return _compiled_conditioned_rollout(
        _STATE_DYNAMIC_ROLLOUT_CACHE,
        state_dynamic_rollout_graph,
        tensors,
        depth_weights.shape[0],
        rho_max,
        group_size,
    )


def compiled_matched_feedforward_rollout(
    drive_r: Tensor,
    drive_i: Tensor,
    hidden: Tensor,
    coefficient_weight: Tensor,
    coefficient_bias: Tensor,
    step_embedding: Tensor,
    rho_max: float,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    tensors = (
        drive_r, drive_i, hidden, coefficient_weight,
        coefficient_bias, step_embedding,
    )
    return _compiled_conditioned_rollout(
        _FEEDFORWARD_ROLLOUT_CACHE, matched_feedforward_rollout_graph,
        tensors, step_embedding.shape[0], rho_max, group_size,
    )


def compiled_identity_memory_f9_step(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    tensors = (propagated_r, propagated_i, drive_r, drive_i, ar, ai)
    key = _tensor_compile_signatures(tensors)
    entry = _IDENTITY_CACHE.get(key)
    if entry is None:
        ordinal = len(_IDENTITY_CACHE)
        target = _clone_graph_code(
            identity_memory_f9_step_graph,
            name=f"identity_memory_f9_step_graph__static_signature_{ordinal}",
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _IDENTITY_CACHE[key] = entry
    return entry[0](*tensors)


def compiled_real_pair_radial_f9_step(
    state_a: Tensor,
    state_b: Tensor,
    drive_a: Tensor,
    drive_b: Tensor,
    strength_a: Tensor,
    strength_b: Tensor,
    coeff_a: Tensor,
    coeff_b: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    tensors = (
        state_a, state_b, drive_a, drive_b, strength_a, strength_b,
        coeff_a, coeff_b,
    )
    key = (int(group_size), _tensor_compile_signatures(tensors))
    entry = _REAL_RADIAL_CACHE.get(key)
    if entry is None:
        ordinal = len(_REAL_RADIAL_CACHE)
        target = _clone_graph_code(
            real_pair_radial_f9_step_graph,
            name=f"real_pair_radial_f9_step_graph__static_signature_{ordinal}",
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _REAL_RADIAL_CACHE[key] = entry
    return entry[0](*tensors, int(group_size))


def compiled_selective_f9_step(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    tensors = (
        propagated_r,
        propagated_i,
        drive_r,
        drive_i,
        strength,
        ar,
        ai,
    )
    if not tensors[0].is_cuda or any(value.device != tensors[0].device for value in tensors):
        raise RuntimeError("compiled selective F9 requires one CUDA device")
    if any(value.dtype not in {torch.float32, torch.bfloat16} for value in tensors):
        raise RuntimeError("compiled selective F9 requires FP32 or BF16 tensors")
    if group_size < 1 or propagated_r.shape[1] % group_size:
        raise ValueError("invalid selective F9 group_size")
    if any(value.shape != propagated_r.shape for value in tensors[:5]):
        raise ValueError("selective state, drive, and strength shapes must match")
    expected_coefficients = (
        propagated_r.shape[0],
        9,
        propagated_r.shape[1],
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if ar.shape != expected_coefficients or ai.shape != expected_coefficients:
        raise ValueError("selective F9 coefficients must have shape [B,9,D,H,W]")

    key = (int(group_size), _tensor_compile_signatures(tensors))
    entry = _CACHE.get(key)
    if entry is None:
        ordinal = len(_CACHE)
        target = _clone_graph_code(
            selective_f9_step_graph,
            name=f"selective_f9_step_graph__static_signature_{ordinal}",
        )
        entry = (
            torch.compile(target, fullgraph=True, dynamic=False),
            target,
        )
        _CACHE[key] = entry
    return entry[0](*tensors, int(group_size))


def compiled_diagonal_lag1_f9_step(
    propagated_r: Tensor,
    propagated_i: Tensor,
    previous_source_r: Tensor,
    previous_source_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    tensors = (
        propagated_r,
        propagated_i,
        previous_source_r,
        previous_source_i,
        drive_r,
        drive_i,
        strength,
        ar,
        ai,
    )
    if not propagated_r.is_cuda or any(
        value.device != propagated_r.device for value in tensors
    ):
        raise RuntimeError("compiled diagonal-lag F9 requires one CUDA device")
    if any(value.dtype not in {torch.float32, torch.bfloat16} for value in tensors):
        raise RuntimeError("compiled diagonal-lag F9 requires FP32 or BF16")
    if group_size < 1 or propagated_r.shape[1] % group_size:
        raise ValueError("invalid diagonal-lag F9 group_size")
    if any(value.shape != propagated_r.shape for value in tensors[:7]):
        raise ValueError("diagonal-lag state, history, drive, and strength shapes must match")
    expected_coefficients = (
        propagated_r.shape[0],
        9,
        propagated_r.shape[1],
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if ar.shape != expected_coefficients or ai.shape != expected_coefficients:
        raise ValueError("diagonal-lag F9 coefficients must have shape [B,9,D,H,W]")

    key = (int(group_size), _tensor_compile_signatures(tensors))
    entry = _DIAGONAL_LAG1_CACHE.get(key)
    if entry is None:
        ordinal = len(_DIAGONAL_LAG1_CACHE)
        target = _clone_graph_code(
            diagonal_lag1_f9_step_graph,
            name=f"diagonal_lag1_f9_step_graph__static_signature_{ordinal}",
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _DIAGONAL_LAG1_CACHE[key] = entry
    return entry[0](*tensors, int(group_size))


def compiled_group_channel_radial_f9_step(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    tensors = (
        propagated_r,
        propagated_i,
        drive_r,
        drive_i,
        strength,
        eta,
        ar,
        ai,
    )
    if not propagated_r.is_cuda or any(
        value.device != propagated_r.device for value in tensors
    ):
        raise RuntimeError("compiled group-channel radial F9 requires one CUDA device")
    if any(value.dtype not in {torch.float32, torch.bfloat16} for value in tensors):
        raise RuntimeError("compiled group-channel radial F9 requires FP32 or BF16")
    if group_size < 1 or propagated_r.shape[1] % group_size:
        raise ValueError("invalid group-channel radial F9 group_size")
    if any(value.shape != propagated_r.shape for value in tensors[:5]):
        raise ValueError("group-channel radial state, drive, and strength shapes must match")
    if eta.numel() != 1:
        raise ValueError("group-channel radial eta must be a scalar tensor")
    expected_coefficients = (
        propagated_r.shape[0],
        9,
        propagated_r.shape[1],
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if ar.shape != expected_coefficients or ai.shape != expected_coefficients:
        raise ValueError("group-channel radial coefficients must have shape [B,9,D,H,W]")

    key = (int(group_size), _tensor_compile_signatures(tensors))
    entry = _GROUP_CHANNEL_RADIAL_CACHE.get(key)
    if entry is None:
        ordinal = len(_GROUP_CHANNEL_RADIAL_CACHE)
        target = _clone_graph_code(
            group_channel_radial_f9_step_graph,
            name=f"group_channel_radial_f9_step_graph__static_signature_{ordinal}",
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _GROUP_CHANNEL_RADIAL_CACHE[key] = entry
    return entry[0](*tensors, int(group_size))


def compiled_tap_discrepancy_f9_step(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_transition_r: Tensor,
    drive_transition_i: Tensor,
    tap_strength: Tensor,
    ar: Tensor,
    ai: Tensor,
    group_size: int,
) -> tuple[Tensor, Tensor]:
    tensors = (
        propagated_r,
        propagated_i,
        drive_transition_r,
        drive_transition_i,
        tap_strength,
        ar,
        ai,
    )
    if not tensors[0].is_cuda or any(
        value.device != tensors[0].device for value in tensors
    ):
        raise RuntimeError("compiled tap-discrepancy F9 requires one CUDA device")
    if any(value.dtype not in {torch.float32, torch.bfloat16} for value in tensors):
        raise RuntimeError("compiled tap-discrepancy F9 requires FP32 or BF16")
    if group_size < 1 or propagated_r.shape[1] % group_size:
        raise ValueError("invalid tap-discrepancy F9 group_size")
    if any(value.shape != propagated_r.shape for value in tensors[:4]):
        raise ValueError("tap-discrepancy state and drive shapes must match")
    groups = propagated_r.shape[1] // group_size
    expected_strength = (
        propagated_r.shape[0],
        9,
        groups,
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if tap_strength.shape != expected_strength:
        raise ValueError(
            "tap strength must have shape [B,9,D/group_size,H,W]"
        )
    expected_coefficients = (
        propagated_r.shape[0],
        9,
        propagated_r.shape[1],
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if ar.shape != expected_coefficients or ai.shape != expected_coefficients:
        raise ValueError("tap-discrepancy coefficients must have shape [B,9,D,H,W]")

    key = (int(group_size), _tensor_compile_signatures(tensors))
    entry = _TAP_CACHE.get(key)
    if entry is None:
        ordinal = len(_TAP_CACHE)
        target = _clone_graph_code(
            tap_discrepancy_f9_step_graph,
            name=f"tap_discrepancy_f9_step_graph__static_signature_{ordinal}",
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _TAP_CACHE[key] = entry
    return entry[0](*tensors, int(group_size))


def compiled_k4_paired_householder_f9_step(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    tensors = (propagated_r, propagated_i, drive_r, drive_i, eta, ar, ai)
    if not tensors[0].is_cuda or any(
        value.device != tensors[0].device for value in tensors
    ):
        raise RuntimeError("compiled K4 paired-Householder F9 requires one CUDA device")
    if any(value.dtype not in {torch.float32, torch.bfloat16} for value in tensors):
        raise RuntimeError("compiled K4 paired-Householder F9 requires FP32 or BF16")
    if propagated_r.shape[1] % 4:
        raise ValueError("K4 paired-Householder F9 requires D divisible by 4")
    if any(value.shape != propagated_r.shape for value in tensors[:4]):
        raise ValueError("K4 paired-Householder state and drive shapes must match")
    expected_eta = (
        propagated_r.shape[0],
        3,
        propagated_r.shape[1] // 4,
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if eta.shape != expected_eta:
        raise ValueError("K4 paired-Householder eta must have shape [B,3,D/4,H,W]")
    expected_coefficients = (
        propagated_r.shape[0],
        9,
        propagated_r.shape[1],
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if ar.shape != expected_coefficients or ai.shape != expected_coefficients:
        raise ValueError("K4 paired-Householder coefficients must have shape [B,9,D,H,W]")
    key = _tensor_compile_signatures(tensors)
    entry = _K4_PAIR_CACHE.get(key)
    if entry is None:
        ordinal = len(_K4_PAIR_CACHE)
        target = _clone_graph_code(
            _k4_paired_householder_f9_step_graph_packed,
            name=f"k4_paired_householder_f9_step_graph__packed_signature_{ordinal}",
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _K4_PAIR_CACHE[key] = entry
    return entry[0](*tensors)


def compiled_k4_radial_f9_step(
    propagated_r: Tensor,
    propagated_i: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    strength: Tensor,
    eta: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    tensors = (
        propagated_r,
        propagated_i,
        drive_r,
        drive_i,
        strength,
        eta,
        ar,
        ai,
    )
    if not propagated_r.is_cuda or any(
        value.device != propagated_r.device for value in tensors
    ):
        raise RuntimeError("compiled K4 radial F9 requires one CUDA device")
    if any(value.dtype not in {torch.float32, torch.bfloat16} for value in tensors):
        raise RuntimeError("compiled K4 radial F9 requires FP32 or BF16")
    if propagated_r.shape[1] % 4:
        raise ValueError("compiled K4 radial F9 requires D divisible by 4")
    if any(value.shape != propagated_r.shape for value in tensors[:5]):
        raise ValueError("compiled K4 radial state, drive, and strength shapes must match")
    expected_eta = (
        propagated_r.shape[0],
        3,
        propagated_r.shape[1] // 4,
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if eta.shape != expected_eta:
        raise ValueError("compiled K4 radial eta must have shape [B,3,D/4,H,W]")
    expected_coefficients = (
        propagated_r.shape[0],
        9,
        propagated_r.shape[1],
        propagated_r.shape[2],
        propagated_r.shape[3],
    )
    if ar.shape != expected_coefficients or ai.shape != expected_coefficients:
        raise ValueError("compiled K4 radial coefficients must have shape [B,9,D,H,W]")
    key = _tensor_compile_signatures(tensors)
    entry = _K4_RADIAL_CACHE.get(key)
    if entry is None:
        ordinal = len(_K4_RADIAL_CACHE)
        target = _clone_graph_code(
            _k4_radial_f9_step_graph_packed,
            name=f"k4_radial_f9_step_graph__packed_signature_{ordinal}",
        )
        entry = (torch.compile(target, fullgraph=True, dynamic=False), target)
        _K4_RADIAL_CACHE[key] = entry
    return entry[0](*tensors)


def clear_compiled_selective_f9_cache() -> None:
    _CACHE.clear()
    _DIAGONAL_LAG1_CACHE.clear()
    _GROUP_CHANNEL_RADIAL_CACHE.clear()
    _TAP_CACHE.clear()
    _K4_PAIR_CACHE.clear()
    _K4_RADIAL_CACHE.clear()
    _IDENTITY_CACHE.clear()
    _REAL_RADIAL_CACHE.clear()
    _STEP_ROLLOUT_CACHE.clear()
    _STATE_DYNAMIC_ROLLOUT_CACHE.clear()
    _FEEDFORWARD_ROLLOUT_CACHE.clear()


def compiled_selective_f9_cache_info() -> dict[str, int]:
    entries = (
        tuple(_CACHE.values())
        + tuple(_DIAGONAL_LAG1_CACHE.values())
        + tuple(_GROUP_CHANNEL_RADIAL_CACHE.values())
        + tuple(_TAP_CACHE.values())
        + tuple(_K4_PAIR_CACHE.values())
        + tuple(_K4_RADIAL_CACHE.values())
        + tuple(_IDENTITY_CACHE.values())
        + tuple(_REAL_RADIAL_CACHE.values())
        + tuple(_STEP_ROLLOUT_CACHE.values())
        + tuple(_STATE_DYNAMIC_ROLLOUT_CACHE.values())
        + tuple(_FEEDFORWARD_ROLLOUT_CACHE.values())
    )
    return {
        "entries": len(entries),
        "unique_target_code_objects": len(
            {id(target.__code__) for _compiled, target in entries}
        ),
    }
