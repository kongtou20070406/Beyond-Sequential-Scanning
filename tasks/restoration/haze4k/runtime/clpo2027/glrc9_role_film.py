"""Frozen C08 GLRC9 latent role-FiLM candidate and strict parent transport.

The candidate keeps the complete GLRC9 shared generator and private readout
heads.  Six zero-initialized, width/role-shared affine sections are inserted
between the live GLRC9 latent and the unchanged private ``H`` readout.  D16 is
intentionally absent from the bank, so it remains exactly on the GLRC9 parent
path.
"""

from __future__ import annotations

import copy
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


GLRC9_ROLE_FILM_MODE = "q9_shared_head_glrc9_role_film_d32d64_operator"
GLRC9_ROLE_FILM_BANK_PREFIX = "q9_role_film_bank."
GLRC9_ROLE_FILM_CLUSTER_SPECS = (
    ("d32_enc2", 32),
    ("d64_enc3", 64),
    ("d64_dec3", 64),
    ("d32_dec2", 32),
    ("d32_dec1", 32),
    ("d32_refine", 32),
)
GLRC9_ROLE_FILM_CLUSTER_COUNT = 6
GLRC9_ROLE_FILM_EXTRA_KEY_COUNT = 12
GLRC9_ROLE_FILM_REAL_PARAMETER_DELTA = 1024
GLRC9_ROLE_FILM_PARAMETER_COUNT = 1_589_471
GLRC9_ROLE_FILM_EPSILON = 1.0e-12


class Q9RoleFiLMResidual(nn.Module):
    """Zero-initialized, bounded affine latent modulation for one cluster.

    ``a`` and ``b`` each have one entry for every real channel in the packed
    complex latent.  The forward law itself is kept in the compiler graph so
    eager and compiled paths share exactly the same native autograd semantics;
    this module owns only the six trainable parameter pairs.
    """

    def __init__(self, state_channels: int) -> None:
        super().__init__()
        if type(state_channels) is not int or state_channels not in {32, 64}:
            raise ValueError("role-FiLM state width must be D32 or D64")
        self.state_channels = state_channels
        self.a = nn.Parameter(torch.zeros(2 * state_channels))
        self.b = nn.Parameter(torch.zeros(2 * state_channels))

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.a)
        nn.init.zeros_(self.b)

    def forward(self, packed_f: Tensor, latent: Tensor) -> Tensor:
        """Apply the fixed C08 law to a GLRC9 latent.

        This method is intentionally a small reference implementation used by
        CPU tests.  Production compiler execution calls the equivalent graph
        in :mod:`clpo2027.coefficient_compilers` to preserve the existing
        cache/dispatch boundary.
        """

        if packed_f.ndim != 4 or latent.ndim != 4:
            raise ValueError("packed_f and latent must have shape [B,C,H,W]")
        expected_channels = 2 * self.state_channels
        if packed_f.shape[1] != expected_channels:
            raise ValueError("packed_f channel count does not match state width")
        if latent.shape[1] != expected_channels:
            raise ValueError("latent channel count does not match state width")
        if packed_f.shape[0] != latent.shape[0] or packed_f.shape[2:] != latent.shape[2:]:
            raise ValueError("packed_f and latent spatial shapes must match")
        real, imag = packed_f.chunk(2, dim=1)
        energy = torch.stack((real, imag), dim=2)
        energy_dtype = (
            torch.float32
            if packed_f.dtype in {torch.float16, torch.bfloat16}
            else packed_f.dtype
        )
        phi_state = 0.25 + 0.5 * torch.sigmoid(
            energy.to(dtype=energy_dtype).square().mean(dim=(2, 3, 4))
        )
        phi = phi_state.to(dtype=latent.dtype).repeat_interleave(2, dim=1)
        phi = phi.unsqueeze(-1).unsqueeze(-1)
        scale = (0.25 * torch.tanh(self.a)).view(1, expected_channels, 1, 1)
        shift = (0.25 * torch.tanh(self.b)).view(1, expected_channels, 1, 1)
        latent_rms = torch.sqrt(
            latent.square().mean(dim=1, keepdim=True)
            + torch.finfo(latent.dtype).eps
        )
        return (1.0 + phi * scale) * latent + phi * shift * latent_rms


@dataclass(frozen=True)
class GLRC9RoleFiLMTransportRecord:
    inserted_keys: tuple[str, ...]
    parent_keys: tuple[str, ...]
    candidate_keys: tuple[str, ...]
    source_key_count: int
    candidate_key_count: int


def expected_role_film_state_keys() -> tuple[str, ...]:
    keys: list[str] = []
    for name, _ in GLRC9_ROLE_FILM_CLUSTER_SPECS:
        keys.extend(
            (
                f"{GLRC9_ROLE_FILM_BANK_PREFIX}{name}.a",
                f"{GLRC9_ROLE_FILM_BANK_PREFIX}{name}.b",
            )
        )
    return tuple(keys)


def _require_tensor(key: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"state entry {key!r} is not a tensor")
    return value


def _validate_role_film_section(
    state_dict: Mapping[str, Tensor],
    inserted_keys: tuple[str, ...],
    *,
    require_parent_section: bool,
) -> None:
    expected_keys = expected_role_film_state_keys()
    if inserted_keys != expected_keys:
        raise ValueError(
            "GLRC9 role-FiLM extras must be the six frozen cluster a/b pairs "
            f"in order; found {inserted_keys}"
        )
    width_counts: Counter[int] = Counter()
    parameter_count = 0
    for cluster_name, state_channels in GLRC9_ROLE_FILM_CLUSTER_SPECS:
        for suffix in ("a", "b"):
            key = f"{GLRC9_ROLE_FILM_BANK_PREFIX}{cluster_name}.{suffix}"
            value = _require_tensor(key, state_dict[key])
            expected_shape = (2 * state_channels,)
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"role-FiLM tensor {key!r} must have shape {expected_shape}; "
                    f"found {tuple(value.shape)}"
                )
            if require_parent_section and int(torch.count_nonzero(value)) != 0:
                raise ValueError(
                    f"nonzero role-FiLM tensor {key!r} is outside the exact parent section"
                )
            parameter_count += value.numel()
        width_counts[state_channels] += 1
    if width_counts != Counter({32: 4, 64: 2}):
        raise ValueError(f"role-FiLM width multiplicities changed: {dict(width_counts)}")
    if parameter_count != GLRC9_ROLE_FILM_REAL_PARAMETER_DELTA:
        raise ValueError(
            "GLRC9 role-FiLM parameter delta must be "
            f"{GLRC9_ROLE_FILM_REAL_PARAMETER_DELTA}; found {parameter_count}"
        )


def transport_glrc9_state_dict_to_role_film(
    parent_state_dict: Mapping[str, Tensor],
    candidate_template_state_dict: Mapping[str, Tensor],
) -> tuple[OrderedDict[str, Tensor], GLRC9RoleFiLMTransportRecord]:
    """Insert the canonical zero role-FiLM bank without changing GLRC9 tensors."""

    parent_keys = tuple(parent_state_dict)
    candidate_keys = tuple(candidate_template_state_dict)
    parent_key_set = set(parent_keys)
    if missing := parent_key_set - set(candidate_keys):
        raise KeyError(f"candidate template is missing GLRC9 keys: {sorted(missing)}")
    inserted_keys = tuple(key for key in candidate_keys if key not in parent_key_set)
    if tuple(key for key in candidate_keys if key in parent_key_set) != parent_keys:
        raise ValueError("candidate template reorders the GLRC9 parent layout")
    for key in parent_keys:
        parent = _require_tensor(key, parent_state_dict[key])
        template = _require_tensor(key, candidate_template_state_dict[key])
        if parent.shape != template.shape:
            raise ValueError(
                f"candidate changes parent shape at {key!r}: "
                f"{tuple(parent.shape)} != {tuple(template.shape)}"
            )
    _validate_role_film_section(
        candidate_template_state_dict,
        inserted_keys,
        require_parent_section=True,
    )
    inserted = set(inserted_keys)
    transported: OrderedDict[str, Tensor] = OrderedDict()
    for key in candidate_keys:
        source = (
            candidate_template_state_dict[key]
            if key in inserted
            else parent_state_dict[key]
        )
        transported[key] = _require_tensor(key, source).detach().clone()
    metadata = getattr(candidate_template_state_dict, "_metadata", None)
    if metadata is not None:
        transported._metadata = copy.deepcopy(metadata)  # type: ignore[attr-defined]
    return transported, GLRC9RoleFiLMTransportRecord(
        inserted_keys=inserted_keys,
        parent_keys=parent_keys,
        candidate_keys=candidate_keys,
        source_key_count=len(parent_keys),
        candidate_key_count=len(candidate_keys),
    )


def inverse_role_film_parent_state_dict(
    state_dict: Mapping[str, Tensor],
    record: GLRC9RoleFiLMTransportRecord,
) -> OrderedDict[str, Tensor]:
    """Recover GLRC9 only from the canonical all-zero role-FiLM section."""

    if tuple(state_dict) != record.candidate_keys:
        raise ValueError("candidate state keys or order differ from transport")
    _validate_role_film_section(
        state_dict,
        record.inserted_keys,
        require_parent_section=True,
    )
    inserted = set(record.inserted_keys)
    if tuple(key for key in state_dict if key not in inserted) != record.parent_keys:
        raise ValueError("inverse did not recover the GLRC9 parent key order")
    parent: OrderedDict[str, Tensor] = OrderedDict(
        (key, _require_tensor(key, state_dict[key]).detach().clone())
        for key in record.parent_keys
    )
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        parent._metadata = copy.deepcopy(metadata)  # type: ignore[attr-defined]
    return parent


__all__ = [
    "GLRC9_ROLE_FILM_BANK_PREFIX",
    "GLRC9_ROLE_FILM_CLUSTER_COUNT",
    "GLRC9_ROLE_FILM_CLUSTER_SPECS",
    "GLRC9_ROLE_FILM_EPSILON",
    "GLRC9_ROLE_FILM_EXTRA_KEY_COUNT",
    "GLRC9_ROLE_FILM_MODE",
    "GLRC9_ROLE_FILM_PARAMETER_COUNT",
    "GLRC9_ROLE_FILM_REAL_PARAMETER_DELTA",
    "GLRC9RoleFiLMTransportRecord",
    "Q9RoleFiLMResidual",
    "expected_role_film_state_keys",
    "inverse_role_film_parent_state_dict",
    "transport_glrc9_state_dict_to_role_film",
]
