"""GLRC9 local-family-to-zero continuation contract.

This module contains only the one scalar transport needed by the RC2
subtractive continuation.  The scalar is deliberately a plain tensor binding
owned by the model; the compiler treats it as a fixed endpoint coordinate and
never creates an optimiser parameter of its own.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


GLRC9_LOCAL_FAMILY_CONTINUATION_MODE = (
    "q9_shared_head_glrc9_local_family_continuation_operator"
)
GLRC9_LOCAL_FAMILY_CONTINUATION_PARENT_MODE = "q9_shared_head_dct_hybrid_operator"
GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY = "q9_local_family_continuation"
GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_COUNT = 1
GLRC9_LOCAL_FAMILY_CONTINUATION_PATH = (1.0, 0.75, 0.5, 0.25, 0.0)


def _scalar(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    return value.reshape(())


def scale_local_family_residual(residual: Tensor, g: Tensor) -> Tensor:
    """Apply the continuation only to the already-separated local residual."""

    g = _scalar(g, name="g")
    if not isinstance(residual, Tensor):
        raise TypeError("residual must be a tensor")
    return residual * g


def parent_to_local_family_state_dict(
    parent_state: Mapping[str, Tensor],
    *,
    strict: bool = True,
) -> dict[str, Tensor]:
    """Transport a GLRC9 parent state into the RC2 candidate by adding ``g=1``."""

    if not isinstance(parent_state, Mapping):
        raise TypeError("parent_state must be a mapping")
    candidate = {key: value.detach().clone() for key, value in parent_state.items()}
    candidate[GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY] = torch.ones(())
    if strict and GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY in parent_state:
        raise ValueError("parent state already contains the RC2 continuation scalar")
    return candidate


def local_family_to_parent_state_dict(
    candidate_state: Mapping[str, Tensor],
    *,
    strict: bool = True,
) -> dict[str, Tensor]:
    """Inverse transport; only the fixed scalar is removed."""

    if not isinstance(candidate_state, Mapping):
        raise TypeError("candidate_state must be a mapping")
    if GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY not in candidate_state:
        raise KeyError(
            f"missing {GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY!r} in candidate state"
        )
    scalar = candidate_state[GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY]
    _scalar(scalar, name=GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY)
    if strict and not torch.equal(
        scalar.detach().reshape(()), scalar.detach().new_tensor(1.0)
    ):
        raise ValueError("inverse transport requires the exact parent endpoint g=1")
    return {
        key: value.detach().clone()
        for key, value in candidate_state.items()
        if key != GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY
    }


def assert_local_family_endpoint(value: Tensor, expected: float = 1.0) -> None:
    value = _scalar(value, name="g")
    if not torch.equal(value.detach(), value.detach().new_tensor(float(expected))):
        raise ValueError(f"GLRC9 local-family endpoint must equal {expected:g}")


__all__ = [
    "GLRC9_LOCAL_FAMILY_CONTINUATION_MODE",
    "GLRC9_LOCAL_FAMILY_CONTINUATION_PARENT_MODE",
    "GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_KEY",
    "GLRC9_LOCAL_FAMILY_CONTINUATION_PARAMETER_COUNT",
    "GLRC9_LOCAL_FAMILY_CONTINUATION_PATH",
    "scale_local_family_residual",
    "parent_to_local_family_state_dict",
    "local_family_to_parent_state_dict",
    "assert_local_family_endpoint",
]
