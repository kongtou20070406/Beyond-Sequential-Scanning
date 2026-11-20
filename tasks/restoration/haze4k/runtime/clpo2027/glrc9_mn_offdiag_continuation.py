"""GLRC9 widely-linear ``M,N`` off-diagonal continuation.

The RC3 arm keeps the GLRC9 action dictionary, spatial compiler, q9 legalizer,
and recurrent state equation unchanged.  Only the already existing widely
linear state map is continued from its dense parent to a diagonal-only
endpoint.  The endpoint helper deliberately computes the map from diagonal
vectors, so no dense off-diagonal tensor is materialized or consumed at
``g=0``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


GLRC9_MN_OFFDIAG_CONTINUATION_MODE = (
    "q9_shared_head_glrc9_mn_offdiag_continuation_operator"
)
GLRC9_MN_OFFDIAG_CONTINUATION_PARENT_MODE = "q9_shared_head_dct_hybrid_operator"
GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR = (
    "dct_hybrid_mn_offdiag_continuation"
)
GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_KEY = "q9_mn_offdiag_continuation"
GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_COUNT = 1
GLRC9_MN_OFFDIAG_CONTINUATION_PATH = (1.0, 0.75, 0.5, 0.25, 0.0)


def _scalar(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    return value.reshape(())


def _check_mn(
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
) -> None:
    tensors = (mix_m_real, mix_m_imag, mix_n_real, mix_n_imag)
    if any(not isinstance(value, Tensor) for value in tensors):
        raise TypeError("M/N maps must be tensors")
    shape = mix_m_real.shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("M/N maps must be square [D,D] tensors")
    if any(value.shape != shape for value in tensors):
        raise ValueError("M/N real/imag maps must have identical shapes")


def _diag(value: Tensor) -> Tensor:
    """Return a diagonal matrix without changing dtype/device."""

    return torch.diag(torch.diagonal(value, dim1=0, dim2=1))


def mn_offdiag_scaled_maps(
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    g: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return ``diag(A)+g*(A-diag(A))`` for all four real maps.

    This helper is for interior points only.  Production ``g=0`` dispatches
    to :func:`mn_offdiag_diagonal_pointwise`, avoiding dense map use.
    """

    _check_mn(mix_m_real, mix_m_imag, mix_n_real, mix_n_imag)
    scalar = _scalar(g, name="g")
    return tuple(
        _diag(value) + scalar * (value - _diag(value))
        for value in (mix_m_real, mix_m_imag, mix_n_real, mix_n_imag)
    )  # type: ignore[return-value]


def _pointwise_from_maps(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
) -> Tensor:
    """Evaluate the split-half real-linear map and return state-major pairs."""

    _check_mn(mix_m_real, mix_m_imag, mix_n_real, mix_n_imag)
    if packed_f.ndim != 4 or packed_f.shape[1] != 2 * mix_m_real.shape[0]:
        raise ValueError("packed_f must have shape [B,2D,H,W] matching M/N")
    real, imag = packed_f.chunk(2, dim=1)
    real_out = torch.einsum(
        "ij,bjhw->bihw", mix_m_real + mix_n_real, real
    ) + torch.einsum("ij,bjhw->bihw", -mix_m_imag + mix_n_imag, imag)
    imag_out = torch.einsum(
        "ij,bjhw->bihw", mix_m_imag + mix_n_imag, real
    ) + torch.einsum("ij,bjhw->bihw", mix_m_real - mix_n_real, imag)
    return torch.stack((real_out, imag_out), dim=2).flatten(1, 2)


def mn_offdiag_diagonal_pointwise(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
) -> Tensor:
    """Evaluate the ``g=0`` diagonal-only map without dense M/N use.

    Only four diagonal vectors are read.  In particular, a caller may pass a
    dense tensor whose off-diagonal storage is invalid/poisoned; this endpoint
    never indexes, materializes, or multiplies those entries.
    """

    _check_mn(mix_m_real, mix_m_imag, mix_n_real, mix_n_imag)
    d = mix_m_real.shape[0]
    if packed_f.ndim != 4 or packed_f.shape[1] != 2 * d:
        raise ValueError("packed_f must have shape [B,2D,H,W] matching M/N")
    real, imag = packed_f.chunk(2, dim=1)
    m_r = torch.diagonal(mix_m_real, dim1=0, dim2=1)
    m_i = torch.diagonal(mix_m_imag, dim1=0, dim2=1)
    n_r = torch.diagonal(mix_n_real, dim1=0, dim2=1)
    n_i = torch.diagonal(mix_n_imag, dim1=0, dim2=1)
    real_out = (m_r + n_r)[None, :, None, None] * real + (
        -m_i + n_i
    )[None, :, None, None] * imag
    imag_out = (m_i + n_i)[None, :, None, None] * real + (
        m_r - n_r
    )[None, :, None, None] * imag
    return torch.stack((real_out, imag_out), dim=2).flatten(1, 2)


def mn_offdiag_materialized_reference_pointwise(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
) -> Tensor:
    """Independent masked reference for the diagonal endpoint.

    The reference explicitly materializes zero-off-diagonal maps from the
    original dense tensors, then applies the same per-state arithmetic as the
    production vector path.  It is never used by production g=0 dispatch and
    therefore does not weaken the no-dense-offdiagonal guarantee.
    """

    _check_mn(mix_m_real, mix_m_imag, mix_n_real, mix_n_imag)
    d = mix_m_real.shape[0]
    if packed_f.ndim != 4 or packed_f.shape[1] != 2 * d:
        raise ValueError("packed_f must have shape [B,2D,H,W] matching M/N")
    diagonal_maps = tuple(
        torch.diag(torch.diagonal(value, dim1=0, dim2=1))
        for value in (mix_m_real, mix_m_imag, mix_n_real, mix_n_imag)
    )
    m_r, m_i, n_r, n_i = (
        torch.diagonal(value, dim1=0, dim2=1) for value in diagonal_maps
    )
    real, imag = packed_f.chunk(2, dim=1)
    real_out = (m_r + n_r)[None, :, None, None] * real + (
        -m_i + n_i
    )[None, :, None, None] * imag
    imag_out = (m_i + n_i)[None, :, None, None] * real + (
        m_r - n_r
    )[None, :, None, None] * imag
    return torch.stack((real_out, imag_out), dim=2).flatten(1, 2)


def mn_offdiag_continuation_pointwise(
    packed_f: Tensor,
    mix_m_real: Tensor,
    mix_m_imag: Tensor,
    mix_n_real: Tensor,
    mix_n_imag: Tensor,
    g: Tensor,
) -> Tensor:
    """Evaluate an interior ``0 < g < 1`` continuation point.

    Endpoint dispatch is deliberately *not* performed here.  This function is
    used as the body of a ``torch.compile`` graph, so branching on the value of
    the scalar would create a data-dependent Dynamo operator (and would also
    mix the endpoint graphs).  The owning compiler dispatches g=1 to the
    ordinary GLRC9 graph and g=0 to the diagonal-only graph in Python before
    entering this function.
    """

    scalar = _scalar(g, name="g")
    scaled = mn_offdiag_scaled_maps(
        mix_m_real, mix_m_imag, mix_n_real, mix_n_imag, scalar
    )
    return _pointwise_from_maps(packed_f, *scaled)


def parent_to_mn_offdiag_state_dict(
    parent_state_dict: Mapping[str, Tensor],
    *,
    g: float = 1.0,
) -> dict[str, Tensor]:
    """Strict parent transport: clone parent keys and append one fixed scalar."""

    if not isinstance(parent_state_dict, Mapping):
        raise TypeError("parent_state_dict must be a mapping")
    if GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_KEY in parent_state_dict:
        raise ValueError("parent state dict already contains continuation key")
    scalar = float(g)
    if not torch.isfinite(torch.tensor(scalar)):
        raise ValueError("continuation g must be finite")
    candidate = {
        str(key): value.detach().clone() if isinstance(value, Tensor) else value
        for key, value in parent_state_dict.items()
    }
    reference = next(
        (value for value in candidate.values() if isinstance(value, Tensor)), None
    )
    if reference is None:
        raise ValueError("parent state dict contains no tensor values")
    candidate[GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_KEY] = reference.new_tensor(
        scalar
    )
    return candidate


def mn_offdiag_to_parent_state_dict(
    continuation_state_dict: Mapping[str, Tensor], *, strict: bool = True
) -> dict[str, Tensor]:
    """Inverse transport at the exact parent endpoint ``g=1``."""

    if not isinstance(continuation_state_dict, Mapping):
        raise TypeError("continuation_state_dict must be a mapping")
    key = GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_KEY
    if key not in continuation_state_dict:
        raise KeyError(f"missing {key!r} in continuation state dict")
    scalar = _scalar(continuation_state_dict[key], name=key)
    if strict and not torch.equal(scalar.detach(), scalar.new_tensor(1.0)):
        raise ValueError("inverse transport requires the exact parent endpoint g=1")
    return {
        str(name): value.detach().clone() if isinstance(value, Tensor) else value
        for name, value in continuation_state_dict.items()
        if name != key
    }


def assert_mn_offdiag_endpoint(state_dict: Mapping[str, Tensor], g: float) -> None:
    key = GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_KEY
    if key not in state_dict:
        raise ValueError("continuation state dict has no M/N scalar")
    value = _scalar(state_dict[key], name=key)
    if not torch.equal(value.detach(), value.new_tensor(float(g))):
        raise ValueError(f"M/N continuation endpoint is {float(value)}; expected {float(g)}")


# A stable, serializable marker used by the Canary to prove that g=0 selected
# the diagonal graph rather than the materialized reference implementation.
MN_OFFDIAG_DIAGONAL_ENDPOINT_MARKER = "mn_offdiag_diagonal_only_graph"


__all__ = [
    "GLRC9_MN_OFFDIAG_CONTINUATION_MODE",
    "GLRC9_MN_OFFDIAG_CONTINUATION_PARENT_MODE",
    "GLRC9_MN_OFFDIAG_CONTINUATION_SPATIAL_OPERATOR",
    "GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_KEY",
    "GLRC9_MN_OFFDIAG_CONTINUATION_PARAMETER_COUNT",
    "GLRC9_MN_OFFDIAG_CONTINUATION_PATH",
    "MN_OFFDIAG_DIAGONAL_ENDPOINT_MARKER",
    "mn_offdiag_scaled_maps",
    "mn_offdiag_diagonal_pointwise",
    "mn_offdiag_materialized_reference_pointwise",
    "mn_offdiag_continuation_pointwise",
    "parent_to_mn_offdiag_state_dict",
    "mn_offdiag_to_parent_state_dict",
    "assert_mn_offdiag_endpoint",
]
