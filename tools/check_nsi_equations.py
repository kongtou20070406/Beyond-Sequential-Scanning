"""Compare a five-step NSI rollout with explicit grid equations on CPU.

This is a formula-to-code smoke check, not a proof of the theorem.
"""

import sys
from pathlib import Path

import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tasks/restoration/runtime"))
from token_mixers.ours.implementation import RadialProfiledOperator  # noqa: E402
from clpo2027.model import N9_OFFSETS  # noqa: E402

torch.manual_seed(20260923)
GROUP_SIZE = 4  # The released GoPro and O-HAZE configurations use four pairs.
STEPS = 5
op = RadialProfiledOperator(8, K=STEPS, group_size=GROUP_SIZE, rho_max=0.95)
B, D, H, W = 1, 4, 3, 4
dr = torch.randn(B, D, H, W) * 0.2
di = torch.randn_like(dr) * 0.2
strength = torch.rand(B, D, H, W) + 0.1
raw_r = torch.randn(B, 9, D, H, W) * 0.1
raw_i = torch.randn_like(raw_r) * 0.1
# Normalize all nine offsets at each receiving position by their total gain.
mass = (raw_r.square() + raw_i.square()).sqrt().sum(dim=1, keepdim=True)
ar = 0.95 * raw_r / (1 + mass)
ai = 0.95 * raw_i / (1 + mass)
w = op.depth_weights(dr)
with torch.no_grad():
    actual_r, actual_i = op._profiled_rollout(dr, di, strength, ar, ai, w)
hr = torch.zeros_like(dr)
hi = torch.zeros_like(di)
zr = torch.zeros_like(dr)
zi = torch.zeros_like(di)
for weight in w:
    # One retention coefficient is shared by the pairs in each group.
    sr = strength.reshape(B, D // GROUP_SIZE, GROUP_SIZE, H, W).mean(
        dim=2, keepdim=True
    )
    energy = (
        hr.reshape(B, D // GROUP_SIZE, GROUP_SIZE, H, W).square()
        + hi.reshape(B, D // GROUP_SIZE, GROUP_SIZE, H, W).square()
    ).mean(dim=2, keepdim=True)
    eta = (
        torch.rsqrt(1 + sr * energy)
        .expand(-1, -1, GROUP_SIZE, -1, -1)
        .reshape(B, D, H, W)
    )
    vr = dr + eta * hr
    vi = di + eta * hi
    nr = torch.zeros_like(hr)
    ni = torch.zeros_like(hi)
    for t, (dy, dx) in enumerate(N9_OFFSETS):
        for y in range(H):
            for x in range(W):
                sy = y + dy
                sx = x + dx
                if 0 <= sy < H and 0 <= sx < W:
                    nr[:, :, y, x] += (
                        ar[:, t, :, y, x] * vr[:, :, sy, sx]
                        - ai[:, t, :, y, x] * vi[:, :, sy, sx]
                    )
                    ni[:, :, y, x] += (
                        ar[:, t, :, y, x] * vi[:, :, sy, sx]
                        + ai[:, t, :, y, x] * vr[:, :, sy, sx]
                    )
    zr += weight * (nr - hr)
    zi += weight * (ni - hi)
    hr, hi = nr, ni
err = max((zr - actual_r).abs().max().item(), (zi - actual_i).abs().max().item())
q = (ar.square() + ai.square()).sqrt().sum(dim=1).max().item()


def N(r, i):
    return (r.square() + i.square()).sqrt().max().item()


bound = q / (1 - q) * N(dr, di)
print(
    {
        "formula_vs_code_max_abs": err,
        "q_f": q,
        "rho": 0.95,
        "group_size": GROUP_SIZE,
        "sum_weights": w.sum().item(),
        "weights_nondecreasing": bool(torch.all(w[1:] >= w[:-1])),
        "N_hK": N(hr, hi),
        "N_z": N(zr, zi),
        "uniform_bound": bound,
    }
)
if not (
    err < 2e-7
    and q < 0.95
    and abs(w.sum().item() - STEPS) < 1e-6
    and bool(torch.all(w[1:] >= w[:-1]))
    and N(hr, hi) <= bound + 1e-7
    and N(zr, zi) <= bound + 1e-7
):
    raise AssertionError("NSI formula check failed")
