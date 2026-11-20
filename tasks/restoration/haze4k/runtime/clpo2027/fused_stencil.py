"""Exact finite F9 stencil graphs and their optional Inductor fusion wrappers.

The public ``*_graph`` functions are ordinary PyTorch reference graphs and work
on CPU for contract tests.  The ``compiled_*`` wrappers are deliberately
CUDA/no-grad only and accept FP32 or BF16: they are an implementation backend
for the registered ``finite_exact_adjoint`` path, not a replacement gradient
estimator.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
import types
import threading
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


F9_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)

# A deliberately small remote-only codebook.  One stage-grid offset naturally
# corresponds to a different original-image distance at each U-Net scale, so
# the same eight codes provide multiscale coverage without a dense N x N map.
REMOTE_HOLE_OFFSETS: tuple[tuple[int, int], ...] = (
    (-4, 0), (4, 0), (0, -4), (0, 4),
    (-8, -8), (-8, 8), (8, -8), (8, 8),
)
REMOTE_HOLE_RADIUS = 8


_K12_PAIR_FUSION_STATE = threading.local()
K12_PAIR_FUSION_DEFAULT = True


def k12_pair_fusion_enabled() -> bool:
    """Return the current thread's non-persistent K12 pair switch."""

    return bool(getattr(_K12_PAIR_FUSION_STATE, "enabled", K12_PAIR_FUSION_DEFAULT))


def is_k12_pair_fusion_enabled() -> bool:
    """Explicit-name alias for :func:`k12_pair_fusion_enabled`."""

    return k12_pair_fusion_enabled()


@contextmanager
def use_k12_pair_fusion(enabled: bool):
    """Temporarily enable static K12 pair graphs on the current thread.

    The flag is deliberately neither a module parameter nor a checkpoint/state
    field.  Nested scopes and exceptions restore the prior value exactly.
    """

    if type(enabled) is not bool:
        raise TypeError("K12 pair fusion switch must be a bool")
    had_previous = hasattr(_K12_PAIR_FUSION_STATE, "enabled")
    previous = getattr(
        _K12_PAIR_FUSION_STATE, "enabled", K12_PAIR_FUSION_DEFAULT
    )
    _K12_PAIR_FUSION_STATE.enabled = enabled
    try:
        yield
    finally:
        if had_previous:
            _K12_PAIR_FUSION_STATE.enabled = previous
        else:
            delattr(_K12_PAIR_FUSION_STATE, "enabled")


def _shift_from_padded(
    padded: Tensor, dy: int, dx: int, height: int, width: int,
) -> Tensor:
    y0, x0 = 1 + dy, 1 + dx
    return padded[..., y0:y0 + height, x0:x0 + width]


def _shift_zero(value: Tensor, dy: int, dx: int) -> Tensor:
    height, width = value.shape[-2:]
    return _shift_from_padded(
        F.pad(value, (1, 1, 1, 1), mode="constant", value=0.0),
        dy,
        dx,
        height,
        width,
    )


def _remote_shift_from_padded(
    padded: Tensor, dy: int, dx: int, height: int, width: int,
) -> Tensor:
    y0 = REMOTE_HOLE_RADIUS + dy
    x0 = REMOTE_HOLE_RADIUS + dx
    return padded[..., y0:y0 + height, x0:x0 + width]


def _gather_offset_sources(
    packed_state: Tensor,
    route_indices: Tensor,
) -> Tensor:
    """Gather the two precomputed source pixels; invalid routes read zero."""

    batch, channels, height, width = packed_state.shape
    source_linear = route_indices[:, 1].to(torch.long)
    hole_count = source_linear.shape[1]
    spatial = height * width
    flat_state = packed_state.reshape(batch, channels, spatial)
    flat_state = torch.cat(
        (flat_state, flat_state.new_zeros(batch, channels, 1)),
        dim=2,
    )
    gather_index = source_linear.reshape(
        batch, 1, hole_count, spatial
    ).expand(batch, channels, hole_count, spatial)
    gathered = torch.gather(
        flat_state.unsqueeze(2).expand(
            batch, channels, hole_count, spatial + 1
        ),
        3,
        gather_index,
    )
    return gathered.reshape(
        batch, channels, hole_count, height, width
    )


def f9_forward_graph(
    zr: Tensor, zi: Tensor, ar: Tensor, ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the spatially varying complex F9 stencil with zero boundaries."""

    height, width = zr.shape[-2:]
    padded_r = F.pad(zr, (1, 1, 1, 1), mode="constant", value=0.0)
    padded_i = F.pad(zi, (1, 1, 1, 1), mode="constant", value=0.0)
    out_r, out_i = torch.zeros_like(zr), torch.zeros_like(zi)
    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        sr = _shift_from_padded(padded_r, dy, dx, height, width)
        si = _shift_from_padded(padded_i, dy, dx, height, width)
        tap_r, tap_i = ar[:, tap], ai[:, tap]
        out_r = out_r + tap_r * sr - tap_i * si
        out_i = out_i + tap_r * si + tap_i * sr
    return out_r, out_i


def f9_adjoint_graph(packed_vector: Tensor, ar: Tensor, ai: Tensor) -> Tensor:
    """Apply the exact real-packed transpose of the complex F9 stencil."""

    vr, vi = packed_vector.chunk(2, dim=1)
    out_r, out_i = torch.zeros_like(vr), torch.zeros_like(vi)
    for tap, (dy, dx) in enumerate(F9_OFFSETS):
        tap_r, tap_i = ar[:, tap], ai[:, tap]
        local_r = tap_r * vr + tap_i * vi
        local_i = -tap_i * vr + tap_r * vi
        out_r = out_r + _shift_zero(local_r, -dy, -dx)
        out_i = out_i + _shift_zero(local_i, -dy, -dx)
    return torch.cat((out_r, out_i), dim=1)


def f9_forward_pair_graph(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply two exact driven local F9 transitions in one static graph.

    The returned tuple is ``(p_{t+2}.real, p_{t+2}.imag,
    z_{t+1}.real, z_{t+1}.imag)`` where ``z_t = drive + p_t`` and
    ``p_{t+1}=A z_t``.  Keeping the intermediate ``z_{t+1}`` in the return
    value lets the finite trajectory caller record exactly the same states as
    the scalar recurrence without materialising an extra transition graph.
    """

    first_r, first_i = f9_forward_graph(zr, zi, ar, ai)
    second_state_r = drive_r + first_r
    second_state_i = drive_i + first_i
    second_r, second_i = f9_forward_graph(
        second_state_r, second_state_i, ar, ai
    )
    return second_r, second_i, second_state_r, second_state_i


def f9_adjoint_pair_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply two exact local F9 transpose transitions in one static graph."""

    first = f9_adjoint_graph(packed_vector, ar, ai)
    second = f9_adjoint_graph(first, ar, ai)
    return first, second


def f9_coefficient_gradients_graph(
    trajectory: Tensor, aligned_adjoint: Tensor,
) -> tuple[Tensor, Tensor]:
    """Accumulate exact shared F9 coefficient gradients over finite time."""

    hr, hi = trajectory.chunk(2, dim=2)
    lr, li = aligned_adjoint.chunk(2, dim=2)
    time_steps, batch, channels, height, width = hr.shape
    hr_flat = hr.reshape(time_steps * batch, channels, height, width)
    hi_flat = hi.reshape(time_steps * batch, channels, height, width)
    padded_hr = F.pad(hr_flat, (1, 1, 1, 1), mode="constant", value=0.0)
    padded_hi = F.pad(hi_flat, (1, 1, 1, 1), mode="constant", value=0.0)
    grad_ar, grad_ai = [], []
    for dy, dx in F9_OFFSETS:
        shifted_r = _shift_from_padded(
            padded_hr, dy, dx, height, width
        ).reshape_as(hr)
        shifted_i = _shift_from_padded(
            padded_hi, dy, dx, height, width
        ).reshape_as(hi)
        grad_ar.append((lr * shifted_r + li * shifted_i).sum(dim=0))
        grad_ai.append((-lr * shifted_i + li * shifted_r).sum(dim=0))
    return torch.stack(grad_ar, dim=1), torch.stack(grad_ai, dim=1)


def f9_offset_hole_forward_graph(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor]:
    """Redirect center-tap mass to content-selected discrete remote offsets."""

    local_r, local_i = f9_forward_graph(zr, zi, ar, ai)
    packed = torch.cat((zr, zi), dim=1)
    sampled_r, sampled_i = _gather_offset_sources(
        packed, route_indices
    ).chunk(2, dim=1)
    remote_r = (hole_weights * sampled_r).sum(dim=2)
    remote_i = (hole_weights * sampled_i).sum(dim=2)
    delta_r = remote_r - routed_mass * zr
    delta_i = remote_i - routed_mass * zi
    center_r, center_i = ar[:, 0], ai[:, 0]
    return (
        local_r + center_r * delta_r - center_i * delta_i,
        local_i + center_r * delta_i + center_i * delta_r,
    )


def f9_offset_hole_forward_pair_graph(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply two exact driven discrete-hole transitions in one static graph."""

    first_r, first_i = f9_offset_hole_forward_graph(
        zr, zi, ar, ai, hole_weights, route_indices, routed_mass
    )
    second_state_r = drive_r + first_r
    second_state_i = drive_i + first_i
    second_r, second_i = f9_offset_hole_forward_graph(
        second_state_r,
        second_state_i,
        ar,
        ai,
        hole_weights,
        route_indices,
        routed_mass,
    )
    return second_r, second_i, second_state_r, second_state_i


def f9_offset_hole_adjoint_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> Tensor:
    """Exact deterministic transpose of the discrete-hole transition."""

    vr, vi = packed_vector.chunk(2, dim=1)
    height, width = vr.shape[-2:]
    local = f9_adjoint_graph(packed_vector, ar, ai)
    center_r, center_i = ar[:, 0], ai[:, 0]
    delta_cotangent_r = center_r * vr + center_i * vi
    delta_cotangent_i = -center_i * vr + center_r * vi
    weighted_r = delta_cotangent_r.unsqueeze(2) * hole_weights
    weighted_i = delta_cotangent_i.unsqueeze(2) * hole_weights
    route_codes = route_indices[:, 0].to(torch.long)
    remote_adjoint = torch.zeros_like(packed_vector)
    for index, (dy, dx) in enumerate(REMOTE_HOLE_OFFSETS):
        code_mask = (route_codes == index).unsqueeze(1)
        weighted = torch.cat(
            (
                (weighted_r * code_mask).sum(dim=2),
                (weighted_i * code_mask).sum(dim=2),
            ),
            dim=1,
        )
        padded = F.pad(
            weighted,
            (
                REMOTE_HOLE_RADIUS,
                REMOTE_HOLE_RADIUS,
                REMOTE_HOLE_RADIUS,
                REMOTE_HOLE_RADIUS,
            ),
            mode="constant",
            value=0.0,
        )
        remote_adjoint = remote_adjoint + _remote_shift_from_padded(
            padded,
            -dy,
            -dx,
            height,
            width,
        )
    local_center_adjoint = torch.cat(
        (
            routed_mass * delta_cotangent_r,
            routed_mass * delta_cotangent_i,
        ),
        dim=1,
    )
    return local + remote_adjoint - local_center_adjoint


def f9_offset_hole_adjoint_pair_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply two exact discrete-hole transition transposes."""

    first = f9_offset_hole_adjoint_graph(
        packed_vector, ar, ai, hole_weights, route_indices, routed_mass
    )
    second = f9_offset_hole_adjoint_graph(
        first, ar, ai, hole_weights, route_indices, routed_mass
    )
    return first, second


def f9_offset_hole_parameter_gradients_graph(
    trajectory: Tensor,
    aligned_adjoint: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fuse discrete-hole parameter VJPs over the finite horizon."""

    state_r, state_i = trajectory.chunk(2, dim=2)
    adjoint_r, adjoint_i = aligned_adjoint.chunk(2, dim=2)
    time_steps, batch, packed_channels, height, width = trajectory.shape
    repeated_routes = route_indices.unsqueeze(0).expand(
        time_steps, *route_indices.shape
    ).reshape(
        time_steps * batch,
        *route_indices.shape[1:],
    )
    sampled = _gather_offset_sources(
        trajectory.reshape(
            time_steps * batch, packed_channels, height, width
        ),
        repeated_routes,
    ).reshape(
        time_steps,
        batch,
        packed_channels,
        hole_weights.shape[2],
        height,
        width,
    )
    sampled_r, sampled_i = sampled.chunk(2, dim=2)
    weight = hole_weights.unsqueeze(0)
    remote_r = (weight * sampled_r).sum(dim=3)
    remote_i = (weight * sampled_i).sum(dim=3)
    mass = routed_mass.unsqueeze(0)
    delta_r = remote_r - mass * state_r
    delta_i = remote_i - mass * state_i
    center_r = ar[:, 0].unsqueeze(0)
    center_i = ai[:, 0].unsqueeze(0)
    delta_cotangent_r = center_r * adjoint_r + center_i * adjoint_i
    delta_cotangent_i = -center_i * adjoint_r + center_r * adjoint_i
    center_grad_ar = (
        adjoint_r * delta_r + adjoint_i * delta_i
    ).sum(dim=0)
    center_grad_ai = (
        -adjoint_r * delta_i + adjoint_i * delta_r
    ).sum(dim=0)
    grad_weights = (
        delta_cotangent_r.unsqueeze(3) * sampled_r
        + delta_cotangent_i.unsqueeze(3) * sampled_i
    ).sum(dim=0)
    grad_routed_mass = -(
        delta_cotangent_r * state_r
        + delta_cotangent_i * state_i
    ).sum(dim=0)
    return (
        center_grad_ar,
        center_grad_ai,
        grad_weights,
        grad_routed_mass,
    )


def _factorized_spatial_graph(
    packed_state: Tensor,
    source: Tensor,
    target: Tensor,
    rank_rotation: Tensor | None = None,
) -> Tensor:
    """Apply ``target.T @ source`` with an optional per-state rank rotation."""

    batch, channels, height, width = packed_state.shape
    rank = source.shape[1]
    spatial = height * width
    source_flat = source.reshape(batch, rank, spatial)
    target_flat = target.reshape(batch, rank, spatial)
    state_flat = packed_state.reshape(batch, channels, spatial)
    # R=8 and the spatial reduction can be very long (up to 65,536).  Express
    # both contractions directly so Inductor emits shape-specialized reduction
    # kernels instead of dispatching hundreds of extremely skinny GEMMs.
    memory = (
        source_flat.unsqueeze(2) * state_flat.unsqueeze(1)
    ).sum(dim=3)
    if rank_rotation is not None:
        if channels % 2:
            raise ValueError("rotated complex state must have 2D channels")
        state_channels = channels // 2
        if rank_rotation.ndim == 3:
            valid_shape = (
                rank_rotation.shape == (state_channels, rank, rank)
            )
        elif rank_rotation.ndim == 4:
            valid_shape = (
                rank_rotation.shape
                == (batch, state_channels, rank, rank)
            )
        else:
            valid_shape = False
        if not valid_shape:
            raise ValueError(
                "rank rotation must have shape [D,R,R] or [B,D,R,R]"
            )
        memory_lanes = memory.reshape(
            batch, rank, 2, state_channels
        ).permute(0, 2, 3, 1)
        memory_lanes = (
            torch.einsum(
                "btdi,doi->btdo", memory_lanes, rank_rotation
            )
            if rank_rotation.ndim == 3
            else torch.einsum(
                "btdi,bdoi->btdo", memory_lanes, rank_rotation
            )
        )
        memory = memory_lanes.permute(0, 3, 1, 2).reshape(
            batch, rank, channels
        )
    output = (
        memory.unsqueeze(3) * target_flat.unsqueeze(2)
    ).sum(dim=1)
    return output.reshape(batch, channels, height, width)


def f9_content_forward_graph(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply the exact local-plus-content transition in one static graph."""

    local_r, local_i = f9_forward_graph(zr, zi, ar, ai)
    global_packed = _factorized_spatial_graph(
        torch.cat((zr, zi), dim=1),
        key,
        normalized_query,
    )
    global_r, global_i = global_packed.chunk(2, dim=1)
    local_scale = 1.0 - alpha.abs()
    global_scale = 0.95 * alpha
    return (
        local_scale * local_r + global_scale * global_r,
        local_scale * local_i + global_scale * global_i,
    )


def f9_content_adjoint_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> Tensor:
    """Apply the exact transpose of the local-plus-content transition."""

    if alpha.ndim == 4:
        packed_alpha = (
            torch.cat((alpha, alpha), dim=1)
            if alpha.shape[1] * 2 == packed_vector.shape[1]
            else alpha
        )
        local = f9_adjoint_graph(
            (1.0 - packed_alpha.abs()) * packed_vector, ar, ai
        )
        global_adjoint = _factorized_spatial_graph(
            (0.95 * packed_alpha) * packed_vector,
            normalized_query,
            key,
        )
        return local + global_adjoint
    local = f9_adjoint_graph(packed_vector, ar, ai)
    global_adjoint = _factorized_spatial_graph(
        packed_vector,
        normalized_query,
        key,
    )
    return (
        (1.0 - alpha.abs()) * local
        + (0.95 * alpha) * global_adjoint
    )


def f9_content_rotation_forward_graph(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
    rank_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply local propagation plus per-state rotated Q/K propagation."""

    local_r, local_i = f9_forward_graph(zr, zi, ar, ai)
    global_packed = _factorized_spatial_graph(
        torch.cat((zr, zi), dim=1),
        key,
        normalized_query,
        rank_rotation,
    )
    global_r, global_i = global_packed.chunk(2, dim=1)
    local_scale = 1.0 - alpha.abs()
    global_scale = 0.95 * alpha
    return (
        local_scale * local_r + global_scale * global_r,
        local_scale * local_i + global_scale * global_i,
    )


def f9_content_rotation_adjoint_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
    rank_rotation: Tensor,
) -> Tensor:
    """Exact transpose using the per-state rotation transpose."""

    local = f9_adjoint_graph(packed_vector, ar, ai)
    global_adjoint = _factorized_spatial_graph(
        packed_vector,
        normalized_query,
        key,
        rank_rotation.transpose(-1, -2),
    )
    return (
        (1.0 - alpha.abs()) * local
        + (0.95 * alpha) * global_adjoint
    )


def f9_content_rotation_effective_forward_graph(
    zr: Tensor,
    zi: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply a rotated transition whose constant budget is folded once."""

    local_r, local_i = f9_forward_graph(
        zr, zi, effective_ar, effective_ai
    )
    global_packed = _factorized_spatial_graph(
        torch.cat((zr, zi), dim=1),
        key,
        normalized_query,
        effective_rank_rotation,
    )
    global_r, global_i = global_packed.chunk(2, dim=1)
    return local_r + global_r, local_i + global_i


def f9_content_rotation_effective_adjoint_graph(
    packed_vector: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> Tensor:
    """Exact transpose of the once-folded rotated transition."""

    return f9_adjoint_graph(
        packed_vector, effective_ar, effective_ai
    ) + _factorized_spatial_graph(
        packed_vector,
        normalized_query,
        key,
        effective_rank_rotation.transpose(-1, -2),
    )


def f9_content_rotation_effective_forward_pair_graph(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply two exact driven effective rotated transitions."""

    first_r, first_i = f9_content_rotation_effective_forward_graph(
        zr,
        zi,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    second_state_r = drive_r + first_r
    second_state_i = drive_i + first_i
    second_r, second_i = f9_content_rotation_effective_forward_graph(
        second_state_r,
        second_state_i,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    return second_r, second_i, second_state_r, second_state_i


def f9_content_rotation_effective_adjoint_pair_graph(
    packed_vector: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply two exact effective rotated transition transposes."""

    first = f9_content_rotation_effective_adjoint_graph(
        packed_vector,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    second = f9_content_rotation_effective_adjoint_graph(
        first,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    return first, second


def f9_content_rotation_joint_forward_graph(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply ``c * (D h + eta G h)`` without materialising the graph."""

    local_r, local_i = f9_forward_graph(zr, zi, ar, ai)
    global_packed = _factorized_spatial_graph(
        torch.cat((zr, zi), dim=1),
        key,
        normalized_query,
        rank_rotation,
    )
    global_r, global_i = global_packed.chunk(2, dim=1)
    return (
        projection_scale * (local_r + eta * global_r),
        projection_scale * (local_i + eta * global_i),
    )


def f9_content_rotation_joint_adjoint_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> Tensor:
    """Exact transpose of ``c * (D + eta G)`` for fixed parameters."""

    packed_scale = torch.cat(
        (projection_scale, projection_scale),
        dim=1,
    )
    weighted = packed_scale * packed_vector
    return f9_adjoint_graph(weighted, ar, ai) + eta * (
        _factorized_spatial_graph(
            weighted,
            normalized_query,
            key,
            rank_rotation.transpose(-1, -2),
        )
    )


def f9_content_rotation_joint_forward_pair_graph(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply two exact driven joint-projection transitions."""

    first_r, first_i = f9_content_rotation_joint_forward_graph(
        zr,
        zi,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    second_state_r = drive_r + first_r
    second_state_i = drive_i + first_i
    second_r, second_i = f9_content_rotation_joint_forward_graph(
        second_state_r,
        second_state_i,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    return second_r, second_i, second_state_r, second_state_i


def f9_content_rotation_joint_adjoint_pair_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply two exact joint-projection transition transposes."""

    first = f9_content_rotation_joint_adjoint_graph(
        packed_vector,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    second = f9_content_rotation_joint_adjoint_graph(
        first,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    return first, second


def f9_content_forward_pair_graph(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Apply two exact driven content transitions in one static graph."""

    first_r, first_i = f9_content_forward_graph(
        zr, zi, ar, ai, normalized_query, key, alpha
    )
    second_state_r = drive_r + first_r
    second_state_i = drive_i + first_i
    second_r, second_i = f9_content_forward_graph(
        second_state_r,
        second_state_i,
        ar,
        ai,
        normalized_query,
        key,
        alpha,
    )
    return second_r, second_i, second_state_r, second_state_i


def f9_content_adjoint_pair_graph(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply two exact content-transition transposes in one static graph."""

    first = f9_content_adjoint_graph(
        packed_vector, ar, ai, normalized_query, key, alpha
    )
    second = f9_content_adjoint_graph(
        first, ar, ai, normalized_query, key, alpha
    )
    return first, second


_F9_GRAPHS: dict[str, Callable[..., Any]] = {
    "forward": f9_forward_graph,
    "adjoint": f9_adjoint_graph,
    "forward_pair": f9_forward_pair_graph,
    "adjoint_pair": f9_adjoint_pair_graph,
    "coefficient_gradients": f9_coefficient_gradients_graph,
    "content_forward": f9_content_forward_graph,
    "content_adjoint": f9_content_adjoint_graph,
    "content_rotation_forward": f9_content_rotation_forward_graph,
    "content_rotation_adjoint": f9_content_rotation_adjoint_graph,
    "content_rotation_effective_forward": (
        f9_content_rotation_effective_forward_graph
    ),
    "content_rotation_effective_adjoint": (
        f9_content_rotation_effective_adjoint_graph
    ),
    "content_rotation_effective_forward_pair": (
        f9_content_rotation_effective_forward_pair_graph
    ),
    "content_rotation_effective_adjoint_pair": (
        f9_content_rotation_effective_adjoint_pair_graph
    ),
    "content_rotation_joint_forward": (
        f9_content_rotation_joint_forward_graph
    ),
    "content_rotation_joint_adjoint": (
        f9_content_rotation_joint_adjoint_graph
    ),
    "content_rotation_joint_forward_pair": (
        f9_content_rotation_joint_forward_pair_graph
    ),
    "content_rotation_joint_adjoint_pair": (
        f9_content_rotation_joint_adjoint_pair_graph
    ),
    "content_forward_pair": f9_content_forward_pair_graph,
    "content_adjoint_pair": f9_content_adjoint_pair_graph,
    "offset_hole_forward": f9_offset_hole_forward_graph,
    "offset_hole_forward_pair": f9_offset_hole_forward_pair_graph,
    "offset_hole_adjoint": f9_offset_hole_adjoint_graph,
    "offset_hole_adjoint_pair": f9_offset_hole_adjoint_pair_graph,
    "offset_hole_parameter_gradients": (
        f9_offset_hole_parameter_gradients_graph
    ),
}

# One torch.compile wrapper cannot safely serve every static training/audit
# shape.  Dynamo attaches its recompile budget to the Python code object, not
# to the wrapper instance: constructing several wrappers around the same
# function still reaches the default limit after enough signatures.  Each
# entry therefore owns both a signature-specific code-object clone and its
# compiled callable.
_COMPILED_F9_CACHE: dict[
    tuple[Any, ...], tuple[Callable[..., Any], Callable[..., Any]]
] = {}


def _tensor_compile_signatures(tensors: tuple[Tensor, ...]) -> tuple[Any, ...]:
    """Return the guards that may change a static Inductor specialization."""

    storage_groups: dict[tuple[str, int | None, int], int] = {}
    signatures = []
    for value in tensors:
        if value.layout != torch.strided:
            raise RuntimeError("compiled F9 requires strided dense tensors")
        storage = value.untyped_storage()
        storage_token = (
            value.device.type,
            value.device.index,
            int(storage.data_ptr()),
        )
        alias_group = storage_groups.setdefault(
            storage_token, len(storage_groups)
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


def _compiled_f9_cache_key(
    name: str, tensors: tuple[Tensor, ...],
) -> tuple[Any, ...]:
    if name not in _F9_GRAPHS:
        raise ValueError(f"unknown compiled F9 graph {name!r}")
    return (name, _tensor_compile_signatures(tensors))


def _clone_graph_code(
    graph: Callable[..., Any], *, name: str,
) -> Callable[..., Any]:
    """Clone ``graph`` with a fresh code object and the same exact bytecode."""

    code_fields: dict[str, str] = {"co_name": name}
    if hasattr(graph.__code__, "co_qualname"):
        code_fields["co_qualname"] = name
    clone = types.FunctionType(
        graph.__code__.replace(**code_fields),
        graph.__globals__,
        name=name,
        argdefs=graph.__defaults__,
        closure=graph.__closure__,
    )
    clone.__kwdefaults__ = graph.__kwdefaults__
    clone.__annotations__ = graph.__annotations__
    clone.__doc__ = graph.__doc__
    clone.__module__ = graph.__module__
    return clone


def _compile_for_signature(
    name: str, tensors: tuple[Tensor, ...],
) -> Callable[..., Any]:
    key = _compiled_f9_cache_key(name, tensors)
    entry = _COMPILED_F9_CACHE.get(key)
    if entry is None:
        graph = _F9_GRAPHS[name]
        ordinal = len(_COMPILED_F9_CACHE)
        target = _clone_graph_code(
            graph, name=f"{graph.__name__}__static_signature_{ordinal}"
        )
        compiled = torch.compile(target, fullgraph=True, dynamic=False)
        entry = (compiled, target)
        _COMPILED_F9_CACHE[key] = entry
    return entry[0]


def _clear_compiled_f9_cache() -> None:
    """Clear only this module's wrappers (primarily for isolated Canaries)."""

    _COMPILED_F9_CACHE.clear()


def compiled_f9_cache_info() -> dict[str, Any]:
    """Return a JSON-friendly snapshot without exposing tensor storage IDs."""

    by_operation = {name: 0 for name in _F9_GRAPHS}
    target_code_ids = set()
    for key, (_compiled, target) in _COMPILED_F9_CACHE.items():
        by_operation[key[0]] += 1
        target_code_ids.add(id(target.__code__))
    return {
        "entries": len(_COMPILED_F9_CACHE),
        "by_operation": by_operation,
        "unique_target_code_objects": len(target_code_ids),
    }


def _require_cuda_fp32_no_grad(name: str, tensors: tuple[Tensor, ...]) -> None:
    if torch.is_grad_enabled():
        raise RuntimeError(
            f"{name} is registered only inside the no-grad finite exact backend"
        )
    if not tensors or any(not value.is_cuda for value in tensors):
        raise RuntimeError(f"{name} requires CUDA tensors")
    if any(
        value.is_floating_point()
        and value.dtype not in {torch.float32, torch.bfloat16}
        for value in tensors
    ):
        raise RuntimeError(f"{name} requires FP32 or BF16 tensors")
    if any(
        not value.is_floating_point()
        and value.dtype not in {torch.int32, torch.int64, torch.bool}
        for value in tensors
    ):
        raise RuntimeError(f"{name} received an unsupported index dtype")
    device = tensors[0].device
    if any(value.device != device for value in tensors):
        raise RuntimeError(f"{name} requires all tensors on one CUDA device")


def _validate_state_and_coefficients(
    zr: Tensor, zi: Tensor, ar: Tensor, ai: Tensor,
) -> None:
    if zr.shape != zi.shape or zr.ndim != 4:
        raise ValueError("compiled F9 real/imaginary state shapes differ")
    if ar.shape != ai.shape or ar.ndim != 5 or ar.shape[1] != 9:
        raise ValueError("compiled F9 coefficient shape must be [B,9,D,H,W]")
    expected = (zr.shape[0], zr.shape[1], zr.shape[2], zr.shape[3])
    actual = (ar.shape[0], ar.shape[2], ar.shape[3], ar.shape[4])
    if actual != expected:
        raise ValueError("compiled F9 state/coefficient dimensions differ")


def compiled_f9_forward(
    zr: Tensor, zi: Tensor, ar: Tensor, ai: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    _require_cuda_fp32_no_grad("compiled F9 forward", (zr, zi, ar, ai))
    tensors = (zr, zi, ar, ai)
    return _compile_for_signature("forward", tensors)(*tensors)


def compiled_f9_forward_pair(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compiled two-step exact F9 forward for the finite K12 trajectory."""

    _validate_state_and_coefficients(zr, zi, ar, ai)
    if drive_r.shape != zr.shape or drive_i.shape != zi.shape:
        raise ValueError("compiled F9 pair drive/state dimensions differ")
    tensors = (zr, zi, drive_r, drive_i, ar, ai)
    _require_cuda_fp32_no_grad("compiled F9 forward pair", tensors)
    return _compile_for_signature("forward_pair", tensors)(*tensors)


def compiled_f9_adjoint(packed_vector: Tensor, ar: Tensor, ai: Tensor) -> Tensor:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError("compiled F9 adjoint requires [B,2D,H,W]")
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _require_cuda_fp32_no_grad(
        "compiled F9 adjoint", (packed_vector, ar, ai)
    )
    tensors = (packed_vector, ar, ai)
    return _compile_for_signature("adjoint", tensors)(*tensors)


def compiled_f9_adjoint_pair(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compiled two-step exact F9 transpose for the finite VJP."""

    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError("compiled F9 adjoint pair requires [B,2D,H,W]")
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    tensors = (packed_vector, ar, ai)
    _require_cuda_fp32_no_grad("compiled F9 adjoint pair", tensors)
    return _compile_for_signature("adjoint_pair", tensors)(*tensors)


def compiled_f9_coefficient_gradients(
    trajectory: Tensor, aligned_adjoint: Tensor,
) -> tuple[Tensor, Tensor]:
    if trajectory.shape != aligned_adjoint.shape or trajectory.ndim != 5:
        raise ValueError(
            "compiled F9 coefficient gradients require matching [T,B,2D,H,W] tensors"
        )
    if trajectory.shape[0] < 1 or trajectory.shape[2] % 2:
        raise ValueError("compiled F9 coefficient trajectory is empty or malformed")
    _require_cuda_fp32_no_grad(
        "compiled F9 coefficient gradients", (trajectory, aligned_adjoint)
    )
    tensors = (trajectory, aligned_adjoint)
    return _compile_for_signature("coefficient_gradients", tensors)(*tensors)


def _validate_offset_holes(
    zr: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> None:
    batch, state_channels, height, width = zr.shape
    if hole_weights.ndim != 5 or hole_weights.shape[:2] != (
        batch, state_channels
    ) or hole_weights.shape[-2:] != (height, width):
        raise ValueError("offset hole weights must be [B,D,M,H,W]")
    hole_count = hole_weights.shape[2]
    if route_indices.shape != (
        batch, 2, hole_count, height, width
    ):
        raise ValueError("offset route indices must be [B,2,M,H,W]")
    if routed_mass.shape != (batch, state_channels, height, width):
        raise ValueError("offset hole routed mass must be [B,D,H,W]")


def compiled_f9_offset_hole_forward(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    _validate_offset_holes(zr, hole_weights, route_indices, routed_mass)
    tensors = (zr, zi, ar, ai, hole_weights, route_indices, routed_mass)
    _require_cuda_fp32_no_grad("compiled F9 offset hole forward", tensors)
    return _compile_for_signature("offset_hole_forward", tensors)(*tensors)


def compiled_f9_offset_hole_forward_pair(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    if drive_r.shape != zr.shape or drive_i.shape != zi.shape:
        raise ValueError("offset hole pair drive/state shapes differ")
    _validate_offset_holes(zr, hole_weights, route_indices, routed_mass)
    tensors = (
        zr,
        zi,
        drive_r,
        drive_i,
        ar,
        ai,
        hole_weights,
        route_indices,
        routed_mass,
    )
    _require_cuda_fp32_no_grad(
        "compiled F9 offset hole forward pair", tensors
    )
    return _compile_for_signature(
        "offset_hole_forward_pair", tensors
    )(*tensors)


def compiled_f9_offset_hole_adjoint(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> Tensor:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError("offset hole adjoint requires [B,2D,H,W]")
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _validate_offset_holes(vr, hole_weights, route_indices, routed_mass)
    tensors = (
        packed_vector, ar, ai, hole_weights, route_indices, routed_mass
    )
    _require_cuda_fp32_no_grad("compiled F9 offset hole adjoint", tensors)
    return _compile_for_signature(
        "offset_hole_adjoint", tensors
    )(*tensors)


def compiled_f9_offset_hole_adjoint_pair(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor]:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError("offset hole adjoint pair requires [B,2D,H,W]")
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _validate_offset_holes(vr, hole_weights, route_indices, routed_mass)
    tensors = (
        packed_vector, ar, ai, hole_weights, route_indices, routed_mass
    )
    _require_cuda_fp32_no_grad(
        "compiled F9 offset hole adjoint pair", tensors
    )
    return _compile_for_signature(
        "offset_hole_adjoint_pair", tensors
    )(*tensors)


def compiled_f9_offset_hole_parameter_gradients(
    trajectory: Tensor,
    aligned_adjoint: Tensor,
    ar: Tensor,
    ai: Tensor,
    hole_weights: Tensor,
    route_indices: Tensor,
    routed_mass: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if trajectory.shape != aligned_adjoint.shape or trajectory.ndim != 5:
        raise ValueError(
            "offset-hole parameter gradients require matching "
            "[T,B,2D,H,W] tensors"
        )
    state_r, _state_i = trajectory.chunk(2, dim=2)
    _validate_state_and_coefficients(state_r[0], state_r[0], ar, ai)
    _validate_offset_holes(
        state_r[0],
        hole_weights,
        route_indices,
        routed_mass,
    )
    tensors = (
        trajectory,
        aligned_adjoint,
        ar,
        ai,
        hole_weights,
        route_indices,
        routed_mass,
    )
    _require_cuda_fp32_no_grad(
        "compiled offset-hole parameter gradients", tensors
    )
    return _compile_for_signature(
        "offset_hole_parameter_gradients", tensors
    )(*tensors)


def _validate_content_factors(
    packed_state: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
    *,
    packed_complex: bool = False,
) -> None:
    if normalized_query.shape != key.shape or normalized_query.ndim != 4:
        raise ValueError("compiled content Q/K factors must be matching BCHW")
    if (
        packed_state.shape[0] != key.shape[0]
        or packed_state.shape[-2:] != key.shape[-2:]
    ):
        raise ValueError("compiled content state/factor dimensions differ")
    if alpha.ndim == 0:
        return
    valid_channels = {1}
    valid_channels.add(
        packed_state.shape[1] // 2
        if packed_complex else packed_state.shape[1]
    )
    if (
        alpha.ndim != 4
        or alpha.shape[0] != packed_state.shape[0]
        or alpha.shape[1] not in valid_channels
        or alpha.shape[-2:] not in {(1, 1), packed_state.shape[-2:]}
    ):
        raise ValueError(
            "compiled content budget alpha must be scalar, "
            "[B,1|D,1,1], or [B,1|D,H,W]"
        )


def _validate_content_rank_rotation(
    batch: int,
    state_channels: int,
    rank: int,
    rank_rotation: Tensor,
) -> None:
    if rank_rotation.shape not in {
        (state_channels, rank, rank),
        (batch, state_channels, rank, rank),
    }:
        raise ValueError(
            "compiled content rank rotation must have shape "
            "[D,R,R] or [B,D,R,R]"
        )


def _validate_joint_projection(
    packed_state: Tensor,
    eta: Tensor,
    projection_scale: Tensor,
    *,
    packed_complex: bool,
) -> None:
    state_channels = (
        packed_state.shape[1] // 2
        if packed_complex
        else packed_state.shape[1]
    )
    expected = (
        packed_state.shape[0],
        state_channels,
        packed_state.shape[-2],
        packed_state.shape[-1],
    )
    if eta.ndim != 0:
        raise ValueError("compiled joint projection requires scalar eta")
    if projection_scale.shape != expected:
        raise ValueError(
            "compiled joint projection scale must have shape [B,D,H,W]"
        )


def compiled_f9_content_forward(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    _validate_content_factors(
        zr,
        normalized_query,
        key,
        alpha,
        packed_complex=False,
    )
    tensors = (zr, zi, ar, ai, normalized_query, key, alpha)
    _require_cuda_fp32_no_grad("compiled F9 content forward", tensors)
    return _compile_for_signature("content_forward", tensors)(*tensors)


def compiled_f9_content_adjoint(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> Tensor:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError("compiled F9 content adjoint requires [B,2D,H,W]")
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _validate_content_factors(
        packed_vector,
        normalized_query,
        key,
        alpha,
        packed_complex=True,
    )
    tensors = (
        packed_vector,
        ar,
        ai,
        normalized_query,
        key,
        alpha,
    )
    _require_cuda_fp32_no_grad("compiled F9 content adjoint", tensors)
    return _compile_for_signature("content_adjoint", tensors)(*tensors)


def compiled_f9_content_rotation_forward(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
    rank_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    _validate_content_factors(
        zr,
        normalized_query,
        key,
        alpha,
        packed_complex=False,
    )
    _validate_content_rank_rotation(
        zr.shape[0],
        zr.shape[1],
        normalized_query.shape[1],
        rank_rotation,
    )
    tensors = (
        zr,
        zi,
        ar,
        ai,
        normalized_query,
        key,
        alpha,
        rank_rotation,
    )
    _require_cuda_fp32_no_grad(
        "compiled F9 rotated content forward", tensors
    )
    return _compile_for_signature(
        "content_rotation_forward", tensors
    )(*tensors)


def compiled_f9_content_rotation_adjoint(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
    rank_rotation: Tensor,
) -> Tensor:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError(
            "compiled rotated content adjoint requires [B,2D,H,W]"
        )
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _validate_content_factors(
        packed_vector,
        normalized_query,
        key,
        alpha,
        packed_complex=True,
    )
    _validate_content_rank_rotation(
        vr.shape[0],
        vr.shape[1],
        normalized_query.shape[1],
        rank_rotation,
    )
    tensors = (
        packed_vector,
        ar,
        ai,
        normalized_query,
        key,
        alpha,
        rank_rotation,
    )
    _require_cuda_fp32_no_grad(
        "compiled F9 rotated content adjoint", tensors
    )
    return _compile_for_signature(
        "content_rotation_adjoint", tensors
    )(*tensors)


def compiled_f9_content_rotation_effective_forward(
    zr: Tensor,
    zi: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_state_and_coefficients(
        zr, zi, effective_ar, effective_ai
    )
    _validate_content_factors(
        zr,
        normalized_query,
        key,
        zr.new_zeros(()),
        packed_complex=False,
    )
    _validate_content_rank_rotation(
        zr.shape[0],
        zr.shape[1],
        normalized_query.shape[1],
        effective_rank_rotation,
    )
    tensors = (
        zr,
        zi,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    _require_cuda_fp32_no_grad(
        "compiled effective rotated content forward", tensors
    )
    return _compile_for_signature(
        "content_rotation_effective_forward", tensors
    )(*tensors)


def compiled_f9_content_rotation_effective_adjoint(
    packed_vector: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> Tensor:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError(
            "compiled effective rotated content adjoint requires [B,2D,H,W]"
        )
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(
        vr, vi, effective_ar, effective_ai
    )
    _validate_content_factors(
        packed_vector,
        normalized_query,
        key,
        packed_vector.new_zeros(()),
        packed_complex=True,
    )
    _validate_content_rank_rotation(
        vr.shape[0],
        vr.shape[1],
        normalized_query.shape[1],
        effective_rank_rotation,
    )
    tensors = (
        packed_vector,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    _require_cuda_fp32_no_grad(
        "compiled effective rotated content adjoint", tensors
    )
    return _compile_for_signature(
        "content_rotation_effective_adjoint", tensors
    )(*tensors)


def compiled_f9_content_rotation_effective_forward_pair(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_state_and_coefficients(
        zr, zi, effective_ar, effective_ai
    )
    if drive_r.shape != zr.shape or drive_i.shape != zi.shape:
        raise ValueError(
            "compiled effective rotated pair drive/state dimensions differ"
        )
    _validate_content_factors(
        zr,
        normalized_query,
        key,
        zr.new_zeros(()),
        packed_complex=False,
    )
    _validate_content_rank_rotation(
        zr.shape[0],
        zr.shape[1],
        normalized_query.shape[1],
        effective_rank_rotation,
    )
    tensors = (
        zr,
        zi,
        drive_r,
        drive_i,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    _require_cuda_fp32_no_grad(
        "compiled effective rotated content forward pair", tensors
    )
    return _compile_for_signature(
        "content_rotation_effective_forward_pair", tensors
    )(*tensors)


def compiled_f9_content_rotation_effective_adjoint_pair(
    packed_vector: Tensor,
    effective_ar: Tensor,
    effective_ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    effective_rank_rotation: Tensor,
) -> tuple[Tensor, Tensor]:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError(
            "compiled effective rotated pair adjoint requires [B,2D,H,W]"
        )
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(
        vr, vi, effective_ar, effective_ai
    )
    _validate_content_factors(
        packed_vector,
        normalized_query,
        key,
        packed_vector.new_zeros(()),
        packed_complex=True,
    )
    _validate_content_rank_rotation(
        vr.shape[0],
        vr.shape[1],
        normalized_query.shape[1],
        effective_rank_rotation,
    )
    tensors = (
        packed_vector,
        effective_ar,
        effective_ai,
        normalized_query,
        key,
        effective_rank_rotation,
    )
    _require_cuda_fp32_no_grad(
        "compiled effective rotated content adjoint pair", tensors
    )
    return _compile_for_signature(
        "content_rotation_effective_adjoint_pair", tensors
    )(*tensors)


def compiled_f9_content_rotation_joint_forward(
    zr: Tensor,
    zi: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    _validate_content_factors(
        zr,
        normalized_query,
        key,
        eta,
        packed_complex=False,
    )
    _validate_content_rank_rotation(
        zr.shape[0],
        zr.shape[1],
        normalized_query.shape[1],
        rank_rotation,
    )
    _validate_joint_projection(
        zr,
        eta,
        projection_scale,
        packed_complex=False,
    )
    tensors = (
        zr,
        zi,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    _require_cuda_fp32_no_grad(
        "compiled joint rotated content forward",
        tensors,
    )
    return _compile_for_signature(
        "content_rotation_joint_forward",
        tensors,
    )(*tensors)


def compiled_f9_content_rotation_joint_adjoint(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> Tensor:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError(
            "compiled joint rotated adjoint requires [B,2D,H,W]"
        )
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _validate_content_factors(
        packed_vector,
        normalized_query,
        key,
        eta,
        packed_complex=True,
    )
    _validate_content_rank_rotation(
        vr.shape[0],
        vr.shape[1],
        normalized_query.shape[1],
        rank_rotation,
    )
    _validate_joint_projection(
        packed_vector,
        eta,
        projection_scale,
        packed_complex=True,
    )
    tensors = (
        packed_vector,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    _require_cuda_fp32_no_grad(
        "compiled joint rotated content adjoint",
        tensors,
    )
    return _compile_for_signature(
        "content_rotation_joint_adjoint",
        tensors,
    )(*tensors)


def compiled_f9_content_rotation_joint_forward_pair(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    if drive_r.shape != zr.shape or drive_i.shape != zi.shape:
        raise ValueError(
            "compiled joint pair drive/state dimensions differ"
        )
    _validate_content_factors(
        zr,
        normalized_query,
        key,
        eta,
        packed_complex=False,
    )
    _validate_content_rank_rotation(
        zr.shape[0],
        zr.shape[1],
        normalized_query.shape[1],
        rank_rotation,
    )
    _validate_joint_projection(
        zr,
        eta,
        projection_scale,
        packed_complex=False,
    )
    tensors = (
        zr,
        zi,
        drive_r,
        drive_i,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    _require_cuda_fp32_no_grad(
        "compiled joint rotated content forward pair",
        tensors,
    )
    return _compile_for_signature(
        "content_rotation_joint_forward_pair",
        tensors,
    )(*tensors)


def compiled_f9_content_rotation_joint_adjoint_pair(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    eta: Tensor,
    rank_rotation: Tensor,
    projection_scale: Tensor,
) -> tuple[Tensor, Tensor]:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError(
            "compiled joint rotated adjoint pair requires [B,2D,H,W]"
        )
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _validate_content_factors(
        packed_vector,
        normalized_query,
        key,
        eta,
        packed_complex=True,
    )
    _validate_content_rank_rotation(
        vr.shape[0],
        vr.shape[1],
        normalized_query.shape[1],
        rank_rotation,
    )
    _validate_joint_projection(
        packed_vector,
        eta,
        projection_scale,
        packed_complex=True,
    )
    tensors = (
        packed_vector,
        ar,
        ai,
        normalized_query,
        key,
        eta,
        rank_rotation,
        projection_scale,
    )
    _require_cuda_fp32_no_grad(
        "compiled joint rotated content adjoint pair",
        tensors,
    )
    return _compile_for_signature(
        "content_rotation_joint_adjoint_pair",
        tensors,
    )(*tensors)


def compiled_f9_content_forward_pair(
    zr: Tensor,
    zi: Tensor,
    drive_r: Tensor,
    drive_i: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _validate_state_and_coefficients(zr, zi, ar, ai)
    if drive_r.shape != zr.shape or drive_i.shape != zi.shape:
        raise ValueError("compiled content pair drive/state dimensions differ")
    _validate_content_factors(
        zr, normalized_query, key, alpha, packed_complex=False
    )
    if alpha.ndim != 4:
        raise ValueError(
            "compiled content pair requires alpha [B,1|D,1,1] or "
            "[B,1|D,H,W]"
        )
    tensors = (
        zr,
        zi,
        drive_r,
        drive_i,
        ar,
        ai,
        normalized_query,
        key,
        alpha,
    )
    _require_cuda_fp32_no_grad("compiled F9 content forward pair", tensors)
    return _compile_for_signature("content_forward_pair", tensors)(*tensors)


def compiled_f9_content_adjoint_pair(
    packed_vector: Tensor,
    ar: Tensor,
    ai: Tensor,
    normalized_query: Tensor,
    key: Tensor,
    alpha: Tensor,
) -> tuple[Tensor, Tensor]:
    if packed_vector.ndim != 4 or packed_vector.shape[1] % 2:
        raise ValueError("compiled F9 content adjoint pair requires [B,2D,H,W]")
    vr, vi = packed_vector.chunk(2, dim=1)
    _validate_state_and_coefficients(vr, vi, ar, ai)
    _validate_content_factors(
        packed_vector,
        normalized_query,
        key,
        alpha,
        packed_complex=True,
    )
    if alpha.ndim != 4:
        raise ValueError(
            "compiled content adjoint pair requires alpha [B,1|D,1,1] "
            "or [B,1|D,H,W]"
        )
    tensors = (
        packed_vector,
        ar,
        ai,
        normalized_query,
        key,
        alpha,
    )
    _require_cuda_fp32_no_grad("compiled F9 content adjoint pair", tensors)
    return _compile_for_signature("content_adjoint_pair", tensors)(*tensors)
