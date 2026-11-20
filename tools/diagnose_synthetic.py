"""Run the fixed synthetic path probe on CPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


DIRECT = ((0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
DELAYED_1 = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2))
METHODS = {"ours": "main_ours", "ss2d": "main_ss2d", "cspn_style": "main_cspn_style", "dwconv": "main_dwconv"}
PAPER_RUNS = {
    "ours": "ohaze-ours-seed42",
    "ss2d": "ohaze-ss2d-seed42",
    "cspn_style": "ohaze-cspn_style-seed42",
    "dwconv": "ohaze-dwconv-seed42",
}


def fields(count: int, blur_steps: int, grid: int, generator: torch.Generator) -> torch.Tensor:
    """Make the deterministic Gaussian fields used by the source probe."""

    value = torch.randn(count, 1, grid, grid, generator=generator, dtype=torch.float64)
    kernel = torch.tensor([1.0, 2.0, 1.0], dtype=value.dtype) / 4.0
    for _ in range(blur_steps):
        value = F.conv2d(F.pad(value, (1, 1, 0, 0), mode="circular"), kernel.view(1, 1, 1, 3))
        value = F.conv2d(F.pad(value, (0, 0, 1, 1), mode="circular"), kernel.view(1, 1, 3, 1))
    return (value - value.mean()) / value.std()


def design(value: torch.Tensor, offsets: tuple[tuple[int, int], ...], margin: int) -> torch.Tensor:
    """Build the local feature matrix for one offset set."""

    size = value.shape[-1] - 2 * margin
    return torch.stack(
        [value[:, 0, margin + dy : margin + dy + size, margin + dx : margin + dx + size] for dy, dx in offsets],
        dim=-1,
    ).reshape(-1, len(offsets))


def target(value: torch.Tensor, offset: tuple[int, int], margin: int) -> torch.Tensor:
    """Build the target column for one offset."""

    dy, dx = offset
    size = value.shape[-1] - 2 * margin
    return value[:, 0, margin + dy : margin + dy + size, margin + dx : margin + dx + size].reshape(-1, 1)


def ridge_nmse(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor) -> float:
    """Fit the small ridge model and return normalized test error."""

    gram = train_x.T @ train_x
    identity = torch.eye(gram.shape[0], dtype=gram.dtype)
    weight = torch.linalg.solve(gram + 1e-6 * identity, train_x.T @ train_y)
    mse = (test_x @ weight - test_y).square().mean()
    return float(mse / test_y.var(unbiased=False).clamp_min(1e-30))


def routing_probe(seed: int, train_count: int, test_count: int, grid: int, blur_levels: tuple[int, ...]) -> dict[str, Any]:
    """Run the controlled direct versus delayed routing test."""

    margin = 2
    if grid <= 2 * margin:
        raise ValueError("grid must be greater than four")
    rows: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(seed)
    for blur_steps in blur_levels:
        train = fields(train_count, blur_steps, grid, generator)
        test = fields(test_count, blur_steps, grid, generator)
        direct_train = design(train, DIRECT, margin)
        direct_test = design(test, DIRECT, margin)
        delayed_train = design(train, DELAYED_1, margin)
        delayed_test = design(test, DELAYED_1, margin)
        for name, offset in (("diagonal", (1, 1)), ("axis", (0, 1))):
            train_y = target(train, offset, margin)
            test_y = target(test, offset, margin)
            direct = ridge_nmse(direct_train, train_y, direct_test, test_y)
            delayed_1 = ridge_nmse(delayed_train, train_y, delayed_test, test_y)
            rows.append(
                {
                    "blur_steps": blur_steps,
                    "target": name,
                    "direct_k1_nmse": direct,
                    "delayed_k1_nmse": delayed_1,
                    "delayed_k2_nmse": direct,
                }
            )
    diagonal = [row for row in rows if row["target"] == "diagonal"]
    axis = [row for row in rows if row["target"] == "axis"]
    passed = all(
        row["delayed_k1_nmse"] - row["direct_k1_nmse"] >= 1e-3
        and abs(row["delayed_k2_nmse"] - row["direct_k1_nmse"]) <= 1e-8
        for row in diagonal
    ) and all(abs(row["delayed_k1_nmse"] - row["direct_k1_nmse"]) <= 1e-6 for row in axis)
    return {
        "seed": seed,
        "train_fields": train_count,
        "test_fields": test_count,
        "grid": grid,
        "blur_levels": list(blur_levels),
        "rows": rows,
        "verdict": "SUPPORTED" if passed else "NOT_SUPPORTED",
    }


def image_tensor(path: Path) -> torch.Tensor:
    """Load an RGB image in the [0, 1] range."""

    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div_(255)


def resized_gt(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """Resize ground truth to the saved output size."""

    return F.interpolate(image_tensor(path).unsqueeze(0), size=size, mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)


def masks(gt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the frozen diagonal, axial, and flat masks."""

    gray = (gt * gt.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)[None, None]
    sx = gray.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    sy = sx.transpose(-1, -2)
    gx, gy = F.conv2d(gray, sx, padding=1)[0, 0], F.conv2d(gray, sy, padding=1)[0, 0]
    magnitude = torch.sqrt(gx.square() + gy.square())
    inner = torch.zeros_like(magnitude, dtype=torch.bool)
    inner[1:-1, 1:-1] = True
    values = magnitude[inner]
    edge = inner & (magnitude >= torch.quantile(values, 0.75))
    flat = inner & (magnitude <= torch.quantile(values, 0.50))
    angle = torch.atan2(gy, gx)
    diagonal_angle = torch.abs(torch.cos(2.0 * angle)) < math.sqrt(0.5)
    return {"diagonal_edge": edge & diagonal_angle, "axial_edge": edge & ~diagonal_angle, "flat": flat}


def masked_mse(prediction: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float:
    """Measure MSE inside one frozen mask."""

    return float((prediction - gt).square()[:, mask].mean())


def result_dir(suite_root: Path, method: str) -> Path:
    """Find a saved output folder from a paper run or source suite."""

    candidates = (suite_root / PAPER_RUNS[method] / "results", suite_root / METHODS[method] / "results")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"saved output folder not found for {method} under {suite_root}")


def orientation_probe(suite_root: Path, data_root: Path, names: list[str]) -> dict[str, Any]:
    """Measure the saved-output orientation association."""

    gt_dir = data_root / "test" / "gt"
    per_image: list[dict[str, Any]] = []
    output_dirs = {method: result_dir(suite_root, method) for method in METHODS}
    for name in names:
        predictions = {method: image_tensor(output_dirs[method] / name) for method in METHODS}
        size = predictions["ours"].shape[-2:]
        if any(tuple(value.shape[-2:]) != tuple(size) for value in predictions.values()):
            raise RuntimeError(f"prediction geometry mismatch for {name}")
        gt = resized_gt(gt_dir / name, tuple(size))
        region_masks = masks(gt)
        scores = {
            method: {region: masked_mse(prediction, gt, mask) for region, mask in region_masks.items()}
            for method, prediction in predictions.items()
        }
        gains = {
            method: {
                region: 1.0 - scores[method][region] / scores["ss2d"][region]
                for region in region_masks
            }
            for method in ("ours", "cspn_style", "dwconv")
        }
        contrast = gains["ours"]["diagonal_edge"] - gains["ours"]["axial_edge"]
        per_image.append(
            {
                "image": name,
                "pixel_counts": {key: int(value.sum()) for key, value in region_masks.items()},
                "mse": scores,
                "relative_mse_reduction_vs_ss2d": gains,
                "ours_diagonal_minus_axial_gain": contrast,
            }
        )
    contrasts = [row["ours_diagonal_minus_axial_gain"] for row in per_image]
    mean_contrast = sum(contrasts) / len(contrasts)
    positives = sum(value > 0 for value in contrasts)
    return {
        "suite_root": str(suite_root),
        "data_root": str(data_root),
        "images": names,
        "per_image": per_image,
        "mean_ours_diagonal_minus_axial_gain": mean_contrast,
        "positive_images": positives,
        "verdict": "SUPPORTED" if mean_contrast > 0 and positives >= 4 else "NOT_SUPPORTED",
    }


def markdown(result: dict[str, Any]) -> str:
    """Write a short report for the JSON result."""

    lines = ["# Synthetic path diagnostic", "", "## Controlled routing", "", "| blur | target | direct K1 NMSE | delayed K1 NMSE | delayed K2 NMSE |", "|---:|---|---:|---:|---:|"]
    for row in result["controlled_routing"]["rows"]:
        lines.append(f"| {row['blur_steps']} | {row['target']} | {row['direct_k1_nmse']:.8g} | {row['delayed_k1_nmse']:.8g} | {row['delayed_k2_nmse']:.8g} |")
    lines += ["", f"Verdict: **{result['controlled_routing']['verdict']}**."]
    if "ohaze_orientation" in result:
        orientation = result["ohaze_orientation"]
        lines += ["", "## Saved O-HAZE outputs", "", "| image | diagonal gain | axial gain | diagonal minus axial |", "|---|---:|---:|---:|"]
        for row in orientation["per_image"]:
            gain = row["relative_mse_reduction_vs_ss2d"]["ours"]
            lines.append(f"| {row['image']} | {gain['diagonal_edge']:.4%} | {gain['axial_edge']:.4%} | {row['ours_diagonal_minus_axial_gain']:.4%} |")
        lines += ["", f"Mean contrast: **{orientation['mean_ours_diagonal_minus_axial_gain']:.4%}**; positive images: **{orientation['positive_images']}/{len(orientation['images'])}**.", f"Verdict: **{orientation['verdict']}**."]
    lines += ["", "The controlled probe is a mechanism check. The saved-output probe is a fixed-output association check. Neither is a matched retraining result."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the synthetic path diagnostic on CPU.")
    parser.add_argument("--mode", choices=("synthetic", "orientation", "all"), default="synthetic")
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("diagnostic_outputs/synthetic_path.json"))
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--train-fields", type=int, default=192)
    parser.add_argument("--test-fields", type=int, default=64)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--blur-levels", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--suite-root", type=Path, default=None, help="paper.py output root; default is runs/paper")
    parser.add_argument("--data-root", type=Path, default=None, help="O-HAZE root; default is data/ohaze")
    parser.add_argument("--images", nargs="+", default=["41.jpg", "42.jpg", "43.jpg", "44.jpg", "45.jpg"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.train_fields <= 0 or args.test_fields <= 0:
        raise SystemExit("field counts must be positive")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    result: dict[str, Any] = {
        "device": "cpu",
        "training": False,
        "method_reference": "research_outputs/direct_path_performance_cpu_20260904/run_probe.py",
        "controlled_routing": routing_probe(args.seed, args.train_fields, args.test_fields, args.grid, tuple(args.blur_levels)),
    }
    if args.mode in {"orientation", "all"}:
        names = args.images[: args.limit or None]
        release_root = args.release_root.expanduser().resolve()
        suite_root = (args.suite_root or release_root / "runs" / "paper").expanduser().resolve()
        data_root = (args.data_root or release_root / "data" / "ohaze").expanduser().resolve()
        output_dirs = {method: result_dir(suite_root, method) for method in METHODS}
        for name in names:
            if any(not (folder / name).is_file() for folder in output_dirs.values()):
                raise FileNotFoundError(f"saved output not found for {name} under {suite_root}")
            if not (data_root / "test" / "gt" / name).is_file():
                raise FileNotFoundError(f"ground truth not found: {data_root / 'test' / 'gt' / name}")
        result["ohaze_orientation"] = orientation_probe(suite_root, data_root, names)
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({key: value.get("verdict") for key, value in result.items() if isinstance(value, dict) and "verdict" in value}, indent=2))


if __name__ == "__main__":
    main()
