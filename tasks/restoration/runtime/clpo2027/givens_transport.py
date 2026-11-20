"""Pure Givens-pair Strang transport prototype.

The prototype operates on real-packed complex feature maps with shape
``[B, D, 2, H, W]``.  A fixed per-channel gain is modulated by a bounded,
non-negative magnitude gate computed from the live conditioning tensor.  Each
transport stage applies independent two-point Givens rotations to one of two
offset pairings; offset one leaves the first boundary element unpaired.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


__all__ = ["givens_strang_transport"]


def _validate_inputs(
    value: Tensor,
    conditioning: Tensor,
    gain_x: Tensor,
    gain_y: Tensor,
    eta: float,
) -> tuple[int, int, float]:
    if not isinstance(value, Tensor) or not isinstance(conditioning, Tensor):
        raise TypeError("value and conditioning must be torch.Tensor objects")
    if not isinstance(gain_x, Tensor) or not isinstance(gain_y, Tensor):
        raise TypeError("gain_x and gain_y must be torch.Tensor objects")
    if value.ndim != 5 or value.shape[2] != 2:
        raise ValueError("value must have shape [B, D, 2, H, W]")
    if conditioning.shape != value.shape:
        raise ValueError("conditioning must have the same shape as value")
    if not torch.is_floating_point(value):
        raise TypeError("value must use a floating-point dtype")
    if not torch.is_floating_point(conditioning):
        raise TypeError("conditioning must use a floating-point dtype")
    if value.device != conditioning.device:
        raise ValueError("value and conditioning must be on the same device")

    batch, channels, _, height, width = value.shape
    del batch  # The dimension is validated by the shape contract above.
    if gain_x.ndim != 1 or gain_x.shape[0] != channels:
        raise ValueError("gain_x must have shape [D]")
    if gain_y.ndim != 1 or gain_y.shape[0] != channels:
        raise ValueError("gain_y must have shape [D]")
    if not torch.is_floating_point(gain_x) or not torch.is_floating_point(gain_y):
        raise TypeError("gain_x and gain_y must use floating-point dtypes")
    if gain_x.device != value.device or gain_y.device != value.device:
        raise ValueError("all tensors must be on the same device")

    if type(eta) is not float:
        raise TypeError("eta must be a fixed Python float in [0, 1]")
    eta_value = eta
    if not math.isfinite(eta_value) or not 0.0 <= eta_value <= 1.0:
        raise ValueError("eta must be a finite scalar in [0, 1]")
    return height, width, eta_value


def _magnitude_gate(conditioning: Tensor) -> Tensor:
    """Return a non-negative, bounded live-conditioning magnitude gate."""

    # The real/imaginary pair is the reduction axis.  ``tanh`` maps the
    # non-negative norm into [0, 1), avoiding an unbounded angle multiplier.
    magnitude = torch.linalg.vector_norm(conditioning, dim=2)
    return torch.tanh(magnitude)


def _rotate_horizontal(value: Tensor, angle: Tensor, *, offset: int) -> Tensor:
    """Rotate non-overlapping column pairs for offset zero or one.

    Offset zero pairs ``(0, 1), (2, 3), ...``.  Offset one preserves column
    zero and pairs ``(1, 2), (3, 4), ...``.  The angle for each pair is the
    arithmetic mean of the two endpoint angle fields.
    """

    if offset not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    _, _, _, height, width = value.shape
    pair_count = (width - offset) // 2
    if pair_count == 0:
        return value
    pair_end = offset + 2 * pair_count

    left = value[..., offset:pair_end:2]
    right = value[..., offset + 1:pair_end:2]
    pair_angle = 0.5 * (
        angle[..., offset:pair_end:2] + angle[..., offset + 1:pair_end:2]
    )
    cosine = torch.cos(pair_angle).unsqueeze(2)
    sine = torch.sin(pair_angle).unsqueeze(2)
    rotated_left = cosine * left - sine * right
    rotated_right = sine * left + cosine * right
    paired = torch.stack((rotated_left, rotated_right), dim=-1)
    paired = paired.reshape(value.shape[0], value.shape[1], 2, height, 2 * pair_count)
    pieces = []
    if offset:
        pieces.append(value[..., :offset])
    pieces.append(paired)
    if pair_end < width:
        pieces.append(value[..., pair_end:])
    return torch.cat(pieces, dim=-1)


def _rotate_vertical(value: Tensor, angle: Tensor, *, offset: int) -> Tensor:
    """Rotate non-overlapping row pairs for offset zero or one.

    Offset zero pairs ``(0, 1), (2, 3), ...``.  Offset one preserves row zero
    and pairs ``(1, 2), (3, 4), ...``.  The angle for each pair is the
    arithmetic mean of the two endpoint angle fields.
    """

    if offset not in (0, 1):
        raise ValueError("offset must be 0 or 1")
    _, _, _, height, width = value.shape
    pair_count = (height - offset) // 2
    if pair_count == 0:
        return value
    pair_end = offset + 2 * pair_count

    top = value[:, :, :, offset:pair_end:2, :]
    bottom = value[:, :, :, offset + 1:pair_end:2, :]
    pair_angle = 0.5 * (
        angle[:, :, offset:pair_end:2, :]
        + angle[:, :, offset + 1:pair_end:2, :]
    )
    cosine = torch.cos(pair_angle).unsqueeze(2)
    sine = torch.sin(pair_angle).unsqueeze(2)
    rotated_top = cosine * top - sine * bottom
    rotated_bottom = sine * top + cosine * bottom
    # Keep the pair index before the inserted top/bottom member axis so the
    # reshape below interleaves rows as (top_0, bottom_0, top_1, bottom_1, …).
    paired = torch.stack((rotated_top, rotated_bottom), dim=4)
    paired = paired.reshape(
        value.shape[0], value.shape[1], 2, 2 * pair_count, width
    )
    pieces = []
    if offset:
        pieces.append(value[:, :, :, :offset, :])
    pieces.append(paired)
    if pair_end < height:
        pieces.append(value[:, :, :, pair_end:, :])
    return torch.cat(pieces, dim=3)


def _horizontal_half(value: Tensor, angle: Tensor) -> Tensor:
    """Apply E(x/4) -> O(x/2) -> E(x/4)."""

    value = _rotate_horizontal(value, 0.25 * angle, offset=0)
    value = _rotate_horizontal(value, 0.50 * angle, offset=1)
    return _rotate_horizontal(value, 0.25 * angle, offset=0)


def _vertical_full(value: Tensor, angle: Tensor) -> Tensor:
    """Apply E(y/2) -> O(y) -> E(y/2)."""

    value = _rotate_vertical(value, 0.50 * angle, offset=0)
    value = _rotate_vertical(value, 1.00 * angle, offset=1)
    return _rotate_vertical(value, 0.50 * angle, offset=0)


def givens_strang_transport(
    value: Tensor,
    conditioning: Tensor,
    gain_x: Tensor,
    gain_y: Tensor,
    *,
    eta: float = 1.0,
) -> Tensor:
    """Apply a bounded live-conditioned Givens Strang transport and blend.

    Parameters
    ----------
    value, conditioning:
        Real-packed complex tensors with shape ``[B, D, 2, H, W]``.
    gain_x, gain_y:
        Shared per-channel gains with shape ``[D]``.  ``gain_x`` drives the
        horizontal stages and ``gain_y`` the vertical stages.
    eta:
        A fixed Python ``float`` in ``[0, 1]``.  ``eta=1`` returns the full
        product transport and intermediate values are norm-preserving up to
        roundoff; tensor eta values are rejected.

    Notes
    -----
    The inputs are required to be finite; no runtime finite check is performed
    here to keep this pure function graph-friendly.  The angle field is
    ``tanh(gain) * tanh(||conditioning||_2)``.  The product is nine stages:
    ``T = Hx_half -> Vy_full -> Hx_half`` where
    ``Hx_half = E(x/4) -> O(x/2) -> E(x/4)`` and
    ``Vy_full = E(y/2) -> O(y) -> E(y/2)``.  The blend is evaluated as a
    residual update so an exact zero-angle product remains an exact identity
    without cutting the gain tangent at the zero initialization.
    """

    _, _, eta_value = _validate_inputs(value, conditioning, gain_x, gain_y, eta)

    # Cast only the shared gains/conditioning used for arithmetic.  ``to`` is
    # differentiable, so a caller retaining higher-precision gains still gets
    # a valid first-order gradient while the output follows value's dtype.
    gate = _magnitude_gate(conditioning.to(dtype=value.dtype))
    gain_x_value = gain_x.to(dtype=value.dtype).reshape(1, -1, 1, 1)
    gain_y_value = gain_y.to(dtype=value.dtype).reshape(1, -1, 1, 1)
    angle_x = torch.tanh(gain_x_value) * gate
    angle_y = torch.tanh(gain_y_value) * gate

    transported = _horizontal_half(value, angle_x)
    transported = _vertical_full(transported, angle_y)
    transported = _horizontal_half(transported, angle_x)

    if eta_value == 1.0:
        return transported
    if eta_value == 0.0:
        return value
    # Algebraically this is (1-eta)*value + eta*T(value).  The residual form
    # avoids cancellation/rounding at zero angle and preserves the gain
    # derivative there, which is required for a zero-initialized trainable gain.
    return value + eta_value * (transported - value)
