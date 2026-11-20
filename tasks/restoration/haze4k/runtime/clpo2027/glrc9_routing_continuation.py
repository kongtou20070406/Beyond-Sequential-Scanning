"""GLRC9 routing continuation contract and parent-preserving transport helpers.

The production continuation changes only the parameter-free GLRC9 action
conditioning.  ``g=1`` is the ordinary GLRC9 parent and ``g=0`` is the exact
uniform nine-action endpoint.  The scalar is intentionally explicit so a
checkpoint can freeze and audit the endpoint without introducing a router or
another generator.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


GLRC9_ROUTING_CONTINUATION_MODE = (
    "q9_shared_head_glrc9_routing_continuation_operator"
)
GLRC9_ROUTING_CONTINUATION_PARENT_MODE = "q9_shared_head_dct_hybrid_operator"
GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY = "q9_routing_continuation"
GLRC9_ROUTING_CONTINUATION_PARAMETER_COUNT = 1


def glrc9_routing_conditioning(conditioning: Tensor, g: Tensor) -> Tensor:
    """Blend GLRC9 conditioning with uniform routing.

    ``conditioning = 0.5 + 4.5*p`` for the parent.  The returned value is
    therefore ``1 + g*(conditioning-1)``.  Explicit endpoint selection keeps
    the two frozen endpoints exact in eager and compiled graphs.
    """

    if not isinstance(conditioning, Tensor) or not isinstance(g, Tensor):
        raise TypeError("conditioning and g must be tensors")
    if conditioning.ndim != 3 or conditioning.shape[-1] != 9:
        raise ValueError("GLRC9 conditioning must have shape [B,D,9]")
    if g.numel() != 1:
        raise ValueError("GLRC9 routing continuation g must be scalar")
    scalar = g.reshape(())
    # Explicit endpoint branches make both frozen endpoints bitwise exact in
    # eager and compiled graphs.  The scalar is a fixed production coordinate,
    # not an optimizer parameter, so the zero tangent at g=1 is intentional;
    # reachability is probed at the preregistered interior point g=0.5.
    blended = conditioning + (scalar - 1.0) * (conditioning - 1.0)
    return torch.where(
        scalar == 1.0,
        conditioning,
        torch.where(scalar == 0.0, torch.ones_like(conditioning), blended),
    )


def parent_to_routing_continuation_state_dict(
    parent_state_dict: Mapping[str, Tensor],
    *,
    g: float = 1.0,
) -> dict[str, Tensor]:
    """Transport a strict GLRC9 parent state dict to the continuation arm.

    The parent has no continuation key.  All parent tensors are cloned and a
    single scalar key is appended.  This keeps the operation deterministic and
    avoids silently mutating a caller-owned checkpoint mapping.
    """

    if not isinstance(parent_state_dict, Mapping):
        raise TypeError("parent_state_dict must be a mapping")
    if not torch.isfinite(torch.tensor(float(g))):
        raise ValueError("routing continuation g must be finite")
    if GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY in parent_state_dict:
        raise ValueError("parent state dict already contains continuation key")
    transported: dict[str, Tensor] = {
        str(key): value.detach().clone() if isinstance(value, Tensor) else value
        for key, value in parent_state_dict.items()
    }
    reference = next(
        (value for value in transported.values() if isinstance(value, Tensor)),
        None,
    )
    if reference is None:
        raise ValueError("parent state dict contains no tensor values")
    transported[GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY] = reference.new_tensor(
        float(g)
    )
    return transported


def routing_continuation_to_parent_state_dict(
    continuation_state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Remove the explicit continuation scalar and recover parent keys."""

    if not isinstance(continuation_state_dict, Mapping):
        raise TypeError("continuation_state_dict must be a mapping")
    if GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY not in continuation_state_dict:
        raise ValueError("continuation state dict has no routing scalar")
    transported = {
        str(key): value.detach().clone() if isinstance(value, Tensor) else value
        for key, value in continuation_state_dict.items()
        if key != GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY
    }
    return transported


def assert_routing_endpoint(state_dict: Mapping[str, Tensor], g: float) -> None:
    """Fail closed unless a continuation state dict stores the requested g."""

    if GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY not in state_dict:
        raise ValueError("continuation state dict has no routing scalar")
    value = state_dict[GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY]
    if not isinstance(value, Tensor) or value.numel() != 1:
        raise ValueError("routing continuation state value must be scalar")
    if not torch.equal(value.detach().reshape(()), value.new_tensor(float(g))):
        raise ValueError(
            f"routing continuation endpoint is {float(value)}; expected {float(g)}"
        )


def freeze_routing_endpoint(module: torch.nn.Module, g: float) -> None:
    """Set and freeze every GLRC9 routing coordinate owned by ``module``."""

    parameters = {
        id(parameter): parameter
        for name, parameter in module.named_parameters()
        if name == GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY
        or name.endswith("." + GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY)
    }
    if not parameters or any(
        parameter.numel() != 1 for parameter in parameters.values()
    ):
        raise ValueError("module has no scalar GLRC9 routing continuation")
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.fill_(float(g))
            parameter.requires_grad_(False)
    values = [
        value
        for name, value in module.state_dict().items()
        if name == GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY
        or name.endswith("." + GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY)
    ]
    if len(values) != len(parameters) or any(
        not torch.equal(value.reshape(()), value.new_tensor(float(g)))
        for value in values
    ):
        raise RuntimeError("failed to freeze the requested GLRC9 routing endpoint")


__all__ = [
    "GLRC9_ROUTING_CONTINUATION_MODE",
    "GLRC9_ROUTING_CONTINUATION_PARENT_MODE",
    "GLRC9_ROUTING_CONTINUATION_PARAMETER_KEY",
    "GLRC9_ROUTING_CONTINUATION_PARAMETER_COUNT",
    "glrc9_routing_conditioning",
    "freeze_routing_endpoint",
    "parent_to_routing_continuation_state_dict",
    "routing_continuation_to_parent_state_dict",
    "assert_routing_endpoint",
]
