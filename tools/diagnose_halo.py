"""Run the fixed-weight context halo probes on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


CORE = 32
HALOS = (32, 64, 96, 128)


def image_tensor(path: Path) -> torch.Tensor:
    """Load an RGB image in the [0, 1] range."""

    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div_(255).unsqueeze(0)


def resize_long_side(image: torch.Tensor, side: int = 512) -> torch.Tensor:
    """Resize the long image side to the frozen LS512 value."""

    height, width = image.shape[-2:]
    scale = side / max(height, width)
    return F.interpolate(image, size=(round(height * scale), round(width * scale)), mode="bicubic", align_corners=False, antialias=True)


def pad32(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pad an image to a multiple of 32."""

    height, width = image.shape[-2:]
    return F.pad(image, (0, (-width) % 32, 0, (-height) % 32), mode="reflect"), height, width


def predict(model: torch.nn.Module, source: torch.Tensor) -> torch.Tensor:
    """Run one model pass and crop padding."""

    padded, height, width = pad32(source)
    value = model(padded)
    value = value[-1] if isinstance(value, list) else value
    return value[:, :, :height, :width].clamp(0, 1)


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Return PSNR for tensors in the [0, 1] range."""

    return float(10 * torch.log10(1 / F.mse_loss(prediction, target)))


def aligned_core_start(size: int) -> int:
    """Pick a centered core with every frozen halo inside the image."""

    lower, upper = max(HALOS), size - CORE - max(HALOS)
    candidates = [value for value in range(0, size - CORE + 1, 32) if lower <= value <= upper]
    if not candidates:
        raise ValueError(f"resized dimension {size} cannot contain the frozen core and halo")
    center = (size - CORE) / 2
    return min(candidates, key=lambda value: abs(value - center))


def origins(height: int, width: int) -> list[tuple[int, int]]:
    """Return all non-overlap core origins with a 64-pixel halo."""

    return [(y, x) for y in range(64, height - CORE - 64 + 1, CORE) for x in range(64, width - CORE - 64 + 1, CORE)]


def _load_model(release_root: Path, config: Path, checkpoint: Path) -> torch.nn.Module:
    """Load the published model on CPU."""

    runtime = release_root / "tasks" / "restoration" / "runtime"
    sys.path.insert(0, str(runtime.resolve()))
    from basicsr.models.archs import define_network

    network = yaml.safe_load(config.read_text(encoding="utf-8"))["network_g"]
    model = define_network(dict(network)).float().eval()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("params_ema", saved.get("params", saved)), strict=True)
    return model


def _names(data_root: Path, limit: int) -> list[str]:
    """Return matching O-HAZE test names."""

    hazy = data_root / "test" / "hazy"
    gt = data_root / "test" / "gt"
    if not hazy.is_dir() or not gt.is_dir():
        raise FileNotFoundError(f"test/hazy and test/gt are required under {data_root}")
    names = sorted(path.name for path in hazy.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if limit:
        names = names[:limit]
    if not names or any(not (gt / name).is_file() for name in names):
        raise RuntimeError("the selected O-HAZE test pairs are empty or unmatched")
    return names


def center_probe(model: torch.nn.Module, data_root: Path, names: list[str]) -> dict[str, Any]:
    """Measure one centered core for each context size."""

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name in names:
            source = resize_long_side(image_tensor(data_root / "test" / "hazy" / name))
            target = resize_long_side(image_tensor(data_root / "test" / "gt" / name))
            height, width = source.shape[-2:]
            y0, x0 = aligned_core_start(height), aligned_core_start(width)
            target_core = target[:, :, y0 : y0 + CORE, x0 : x0 + CORE]
            started = time.time()
            full_core = predict(model, source)[:, :, y0 : y0 + CORE, x0 : x0 + CORE]
            scores = {"full": psnr(full_core, target_core)}
            for halo in HALOS:
                crop = source[:, :, y0 - halo : y0 + CORE + halo, x0 - halo : x0 + CORE + halo]
                crop_core = predict(model, crop)[:, :, halo : halo + CORE, halo : halo + CORE]
                scores[str(halo)] = psnr(crop_core, target_core)
            records.append(
                {
                    "image": name,
                    "resized_size": [height, width],
                    "core_origin": [y0, x0],
                    "psnr": scores,
                    "full_minus_halo64": scores["full"] - scores["64"],
                    "full_minus_halo128": scores["full"] - scores["128"],
                    "seconds": time.time() - started,
                }
            )
    mean = lambda key: sum(row[key] for row in records) / len(records)
    delta64 = mean("full_minus_halo64")
    return {
        "protocol": "frozen K5; O-HAZE LS512; CPU FP32; aligned 32x32 core",
        "halos": list(HALOS),
        "records": records,
        "mean_psnr": {setting: sum(row["psnr"][setting] for row in records) / len(records) for setting in ("full", *(str(value) for value in HALOS))},
        "mean_full_minus_halo64": delta64,
        "mean_full_minus_halo128": mean("full_minus_halo128"),
        "positive_full_minus_halo64": sum(row["full_minus_halo64"] > 0 for row in records),
        "verdict": "SUPPORTED" if delta64 >= 0.10 and sum(row["full_minus_halo64"] > 0 for row in records) >= 4 else "NOT_SUPPORTED",
    }


def grid_probe(model: torch.nn.Module, data_root: Path, names: list[str], batch_size: int) -> dict[str, Any]:
    """Measure every non-overlap core with a 64-pixel halo."""

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name in names:
            source = resize_long_side(image_tensor(data_root / "test" / "hazy" / name))
            target = resize_long_side(image_tensor(data_root / "test" / "gt" / name))
            full = predict(model, source)
            points = origins(*source.shape[-2:])
            core_rows: list[dict[str, Any]] = []
            for start in range(0, len(points), batch_size):
                batch_points = points[start : start + batch_size]
                crops = torch.cat([source[:, :, y - 64 : y + CORE + 64, x - 64 : x + CORE + 64] for y, x in batch_points])
                cropped = predict(model, crops)[:, :, 64 : 64 + CORE, 64 : 64 + CORE]
                for index, (y, x) in enumerate(batch_points):
                    target_core = target[:, :, y : y + CORE, x : x + CORE]
                    full_core = full[:, :, y : y + CORE, x : x + CORE]
                    crop_core = cropped[index : index + 1]
                    full_score = psnr(full_core, target_core)
                    crop_score = psnr(crop_core, target_core)
                    core_rows.append(
                        {
                            "origin": [y, x],
                            "full_psnr": full_score,
                            "halo64_psnr": crop_score,
                            "delta": full_score - crop_score,
                            "prediction_rmse": float(F.mse_loss(full_core, crop_core).sqrt()),
                        }
                    )
            mean_delta = sum(row["delta"] for row in core_rows) / len(core_rows)
            records.append(
                {
                    "image": name,
                    "cores": len(core_rows),
                    "mean_delta": mean_delta,
                    "median_delta": float(torch.tensor([row["delta"] for row in core_rows]).median()),
                    "positive_fraction": sum(row["delta"] > 0 for row in core_rows) / len(core_rows),
                    "mean_prediction_rmse": sum(row["prediction_rmse"] for row in core_rows) / len(core_rows),
                    "core_records": core_rows,
                }
            )
    grand = sum(row["mean_delta"] for row in records) / len(records)
    positive_images = sum(row["mean_delta"] > 0 for row in records)
    return {
        "protocol": "post-result spatial average; all aligned non-overlapping 32x32 cores; halo=64; CPU FP32",
        "records": records,
        "total_cores": sum(row["cores"] for row in records),
        "mean_of_image_mean_deltas": grand,
        "positive_images": positive_images,
        "verdict": "SUPPORTED" if grand >= 0.10 and positive_images >= 4 else "NOT_SUPPORTED",
    }


def markdown(results: dict[str, dict[str, Any]]) -> str:
    """Write a short halo report."""

    lines = ["# Context halo diagnostic", ""]
    if "center" in results:
        result = results["center"]
        lines += ["## Centered core", "", "| context | mean core PSNR |", "|---|---:|"]
        for setting, value in result["mean_psnr"].items():
            lines.append(f"| {setting} | {value:.4f} dB |")
        lines += ["", f"Full minus halo64: **{result['mean_full_minus_halo64']:+.4f} dB**; positive images: **{result['positive_full_minus_halo64']}/{len(result['records'])}**.", f"Verdict: **{result['verdict']}**."]
    if "grid" in results:
        result = results["grid"]
        lines += ["", "## All non-overlap cores", "", "| image | cores | full minus halo64 |", "|---|---:|---:|"]
        for row in result["records"]:
            lines.append(f"| {row['image']} | {row['cores']} | {row['mean_delta']:+.4f} dB |")
        lines += ["", f"Total cores: **{result['total_cores']}**; mean: **{result['mean_of_image_mean_deltas']:+.4f} dB**; positive images: **{result['positive_images']}/{len(result['records'])}**.", f"Verdict: **{result['verdict']}**."]
    lines += ["", "The centered and grid probes are fixed-weight diagnostics. They do not train the model."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run context halo diagnostics on CPU.")
    parser.add_argument("--mode", choices=("center", "grid", "all"), default="center")
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=5)
    args = parser.parse_args()
    if args.threads <= 0 or args.batch_size <= 0 or args.max_images < 0:
        raise SystemExit("threads and batch size must be positive; max images must be non-negative")
    release_root = args.release_root.expanduser().resolve()
    config = (args.config or release_root / "configs" / "restoration" / "ohaze" / "O-HAZE_VIVNet_RadialProfiled_K5_P256.yml").expanduser().resolve()
    checkpoint = (args.checkpoint or release_root / "weights" / "ohaze" / "ours" / "seed42" / "last.pth").expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (args.output_dir or release_root / "diagnostic_outputs" / "context_halo").expanduser().resolve()
    for path, label in ((config, "config"), (checkpoint, "checkpoint"), (data_root, "data root")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    torch.set_num_threads(args.threads)
    names = _names(data_root, args.max_images)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for mode in ("center", "grid") if args.mode == "all" else (args.mode,):
        model = _load_model(release_root, config, checkpoint)
        results[mode] = center_probe(model, data_root, names) if mode == "center" else grid_probe(model, data_root, names, args.batch_size)
        del model
        results[mode]["release_root"] = str(release_root)
        results[mode]["config"] = str(config)
        results[mode]["checkpoint"] = str(checkpoint)
        results[mode]["data_root"] = str(data_root)
        results[mode]["method_reference"] = "research_outputs/context_halo_cpu_20260904/run_probe.py"
        results[mode]["training"] = False
        (output_dir / f"{mode}.json").write_text(json.dumps(results[mode], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run.json").write_text(json.dumps({"mode": args.mode, "images": names, "device": "cpu", "training": False}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "results.md").write_text(markdown(results), encoding="utf-8")
    print(json.dumps({mode: result["verdict"] for mode, result in results.items()}, indent=2))


if __name__ == "__main__":
    main()
