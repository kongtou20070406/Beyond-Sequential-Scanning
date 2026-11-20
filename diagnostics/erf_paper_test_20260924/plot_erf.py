"""Plot comparable ERF maps and enclosed-gradient curves for paper models."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_erf import APPENDIX, MAIN


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GROUPS = {
    "gopro_main": ("gopro", MAIN["gopro"]),
    "ohaze_main": ("ohaze", MAIN["ohaze"]),
    "haze4k_main": ("haze4k", MAIN["haze4k"]),
    "ohaze_appendix": ("ohaze", ("ss2d", "ours", *APPENDIX)),
}
DISPLAY_DATASET = {"gopro": "GoPro", "ohaze": "O-HAZE", "haze4k": "Haze4K"}


def read_group(dataset: str, arms: tuple[str, ...]) -> list[tuple[dict, np.ndarray]]:
    rows = []
    for arm in arms:
        root = RESULTS / dataset
        meta = json.loads((root / f"{arm}.json").read_text(encoding="utf-8"))
        mass = np.load(root / f"{arm}.npz")["gradient"].astype(np.float64)
        assert mass.shape == tuple(meta["input_shape"])
        assert np.isfinite(mass).all() and (mass >= 0).all() and mass.sum() > 0
        rows.append((meta, mass))
    first = rows[0][0]
    assert all(row["input"] == first["input"] and row["center_yx"] == first["center_yx"]
               for row, _ in rows), "models must use identical test input and target pixel"
    return rows


def radial_cdf(mass: np.ndarray, center: list[int]) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:mass.shape[0], :mass.shape[1]]
    radius = np.rint(np.hypot(yy - center[0], xx - center[1])).astype(np.int32)
    weights = np.bincount(radius.ravel(), weights=mass.ravel())
    return np.arange(len(weights)), np.cumsum(weights) / mass.sum()


def heatmaps(name: str, rows: list[tuple[dict, np.ndarray]]) -> None:
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(3.25 * n, 6.9), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    for i, (meta, mass) in enumerate(rows):
        cy, cx = meta["center_yx"]
        shown = np.log10(np.maximum(mass / mass.max(), 1e-6))
        full = axes[0, i]
        im = full.imshow(shown, cmap=cmap, vmin=-6, vmax=0, interpolation="nearest")
        full.plot(cx, cy, marker="+", color="cyan", markersize=7, markeredgewidth=1.0)
        full.set_title(f"{meta['label']}\n$r_{{90}}$ = {meta['r90_px']:.1f} px", fontsize=11)
        full.set_xticks([])
        full.set_yticks([])

        zoom = axes[1, i]
        y0, y1 = max(0, cy - 128), min(mass.shape[0], cy + 128)
        x0, x1 = max(0, cx - 128), min(mass.shape[1], cx + 128)
        zoom.imshow(shown[y0:y1, x0:x1], cmap=cmap, vmin=-6, vmax=0, interpolation="nearest")
        zoom.plot(cx - x0, cy - y0, marker="+", color="cyan", markersize=7, markeredgewidth=1.0)
        zoom.set_title("Center 256 × 256", fontsize=10)
        zoom.set_xticks([])
        zoom.set_yticks([])

    dataset = rows[0][0]["dataset"]
    fig.suptitle(f"{DISPLAY_DATASET[dataset]} · full-model ERF at the center output residual pixel",
                 fontsize=15, fontweight="bold")
    bar = fig.colorbar(im, ax=axes.ravel().tolist(), location="bottom", fraction=0.035, pad=0.035)
    bar.set_label(r"$\log_{10}(g / \max g)$, $g=\sum_c |\partial y / \partial x_c|$; per-model normalization")
    fig.savefig(RESULTS / f"{name}_heatmaps.png", dpi=200, facecolor="white")
    fig.savefig(RESULTS / f"{name}_heatmaps.svg", facecolor="white")
    plt.close(fig)


def curves(name: str, rows: list[tuple[dict, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.5), constrained_layout=True)
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf")
    for i, (meta, mass) in enumerate(rows):
        x, y = radial_cdf(mass, meta["center_yx"])
        color = "#d62728" if meta["label"] == "NSI" else colors[i % len(colors)]
        ax.plot(x, y, lw=2.2 if meta["label"] == "NSI" else 1.8,
                color=color, label=f"{meta['label']}  ($r_{{90}}$={meta['r90_px']:.1f} px)")
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.axhline(0.9, color="0.6", lw=0.8, ls="--")
    ax.set(xlabel="Distance from center output pixel (input pixels)",
           ylabel="Cumulative absolute input-gradient mass", ylim=(0, 1.02),
           xlim=(0, None), title=f"{DISPLAY_DATASET[rows[0][0]['dataset']]} · enclosed ERF gradient")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9, ncol=2, loc="lower right")
    fig.savefig(RESULTS / f"{name}_radial.png", dpi=200, facecolor="white")
    fig.savefig(RESULTS / f"{name}_radial.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    records = {}
    for name, (dataset, arms) in GROUPS.items():
        rows = read_group(dataset, arms)
        heatmaps(name, rows)
        curves(name, rows)
        for meta, _ in rows:
            records[meta["id"]] = meta
        print(f"{name}: {len(rows)} models, figures saved", flush=True)
    columns = ("dataset", "id", "label", "input_shape", "r50_px", "r90_px", "r95_px",
               "gradient_sum", "gradient_peak", "checkpoint_sha256")
    with (RESULTS / "ERF_SUMMARY.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for meta in records.values():
            writer.writerow({key: meta[key] for key in columns})


if __name__ == "__main__":
    main()
