"""Check the saved ERF identities, maps, metrics, and figures."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from plot_erf import GROUPS, RESULTS, read_group
from run_erf import CODE, HERE, enclosed_radius


inventory = json.loads((CODE / "weights" / "checksums.json").read_text(encoding="utf-8"))["files"]
lock = json.loads((HERE.parent / "GOPRO_SEED42_LOCK.json").read_text(encoding="utf-8"))
seen = set()
for name, (dataset, arms) in GROUPS.items():
    for meta, mass in read_group(dataset, arms):
        seen.add(meta["id"])
        relative = Path(meta["checkpoint"]).as_posix()
        assert meta["checkpoint_sha256"] == inventory[relative]["sha256"]
        assert 0 <= meta["r50_px"] <= meta["r90_px"] <= meta["r95_px"]
        assert abs(enclosed_radius(mass.astype("float32"), tuple(meta["center_yx"]), 0.9)
                   - meta["r90_px"]) < 1e-6
        if dataset == "gopro" and meta["arm"] in lock:
            assert meta["checkpoint_sha256"] == lock[meta["arm"]]["sha256"]
        if meta["arm"] in {"spatial_mamba", "spatial_native_blocks"}:
            assert meta["spatial_full_forward_max_abs"] < 1e-5
    for kind in ("heatmaps", "radial"):
        for extension in ("png", "svg"):
            path = RESULTS / f"{name}_{kind}.{extension}"
            assert path.is_file() and path.stat().st_size > 1000, path
        with Image.open(RESULTS / f"{name}_{kind}.png") as image:
            assert image.width >= 1500 and image.height >= 700

assert len(seen) == 20, len(seen)
for dataset, arm in (("gopro", "ours"), ("ohaze", "ours"), ("haze4k", "full")):
    meta = json.loads((RESULTS / dataset / f"{arm}.json").read_text(encoding="utf-8"))
    assert meta["eval_forward_max_abs"] < 2e-6
print("PASS: 20 unique formal checkpoints, 20 ERF maps, 8 PNG and 8 SVG figures")
