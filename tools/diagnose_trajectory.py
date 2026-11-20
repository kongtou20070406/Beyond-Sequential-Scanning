"""Run the fixed-weight trajectory readout probe on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import torch
import yaml


def _default_release_root() -> Path:
    """Return the reproduction code directory."""

    return Path(__file__).resolve().parents[1]


def _default_config(release_root: Path) -> Path:
    """Return the published O-HAZE K5 config."""

    return release_root / "configs" / "restoration" / "ohaze" / "O-HAZE_VIVNet_RadialProfiled_K5_P256.yml"


def _default_checkpoint(release_root: Path) -> Path:
    """Return the published O-HAZE checkpoint."""

    return release_root / "weights" / "ohaze" / "ours" / "seed42" / "last.pth"


def operators(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Find the state operators in the release model."""

    return [module for module in model.modules() if hasattr(module, "_state_depth_step") and hasattr(module, "depth_weights")]


def load_model(define_network: Callable, config: Path, checkpoint: Path) -> torch.nn.Module:
    """Load one release checkpoint on CPU."""

    network = yaml.safe_load(config.read_text(encoding="utf-8"))["network_g"]
    if network.get("type") == "VIVNetRadialStateDepthProfiledCLPO":
        network["real_pair_state"] = False
    model = define_network(dict(network))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("params_ema", saved.get("params", saved)), strict=True)
    model = model.float().eval()
    for operator in operators(model):
        operator.selective_backend = "eager"
        operator.set_transition_backend("eager")
    return model


def controlled_depth_weights(self, reference: torch.Tensor) -> torch.Tensor:
    """Select one readout weight order."""

    weights = self._trajectory_original_depth_weights(reference)
    mode = self._trajectory_mode
    if mode == "original":
        selected = weights
    elif mode == "reverse":
        selected = weights.flip(0)
    elif mode == "cyclic":
        selected = weights.roll(-1, 0)
    elif mode == "uniform":
        selected = torch.ones_like(weights)
    else:
        raise ValueError(mode)
    if mode in {"reverse", "cyclic"}:
        if not torch.equal(weights.sort().values, selected.sort().values):
            raise RuntimeError("temporal permutation changed the weight multiset")
        if not torch.allclose(weights.sum(), selected.sum(), rtol=1.0e-7, atol=1.0e-7):
            raise RuntimeError("temporal permutation changed the weight sum")
    return selected


def complex_rms(real: torch.Tensor, imag: torch.Tensor) -> float:
    """Return the RMS of a complex tensor pair."""

    return float(torch.sqrt((real.double().square() + imag.double().square()).mean()))


def complex_cosine(ar: torch.Tensor, ai: torch.Tensor, br: torch.Tensor, bi: torch.Tensor) -> float:
    """Return the cosine between two complex tensor pairs."""

    numerator = (ar.double() * br.double() + ai.double() * bi.double()).sum()
    denominator = torch.sqrt(
        (ar.double().square() + ai.double().square()).sum()
        * (br.double().square() + bi.double().square()).sum()
    ).clamp_min(1.0e-30)
    return float(numerator / denominator)


def local_variation_ratio(real: torch.Tensor, imag: torch.Tensor) -> float:
    """Return the local variation ratio."""

    power = (real.double().square() + imag.double().square()).mean().clamp_min(1.0e-30)
    dx = (real[..., 1:] - real[..., :-1]).double().square() + (imag[..., 1:] - imag[..., :-1]).double().square()
    dy = (real[..., 1:, :] - real[..., :-1, :]).double().square() + (imag[..., 1:, :] - imag[..., :-1, :]).double().square()
    return float((dx.mean() + dy.mean()) / (2.0 * power))


def controlled_state_step(self, propagated_r, propagated_i, drive_r, drive_i, strength, ar, ai):
    """Run one state step and record its increment."""

    next_r, next_i = self._trajectory_original_state_step(
        propagated_r, propagated_i, drive_r, drive_i, strength, ar, ai
    )
    if self._trajectory_capture:
        delta_r = next_r - propagated_r
        delta_i = next_i - propagated_i
        step = self._trajectory_step
        if step == 0:
            self._trajectory_first_delta = (delta_r.detach(), delta_i.detach())
        first_r, first_i = self._trajectory_first_delta
        self._trajectory_records.append(
            {
                "step": step,
                "delta_rms": complex_rms(delta_r, delta_i),
                "delta_to_first_cosine": complex_cosine(delta_r, delta_i, first_r, first_i),
                "delta_local_variation_ratio": local_variation_ratio(delta_r, delta_i),
            }
        )
        if step == self.K - 1:
            self._trajectory_first_delta = None
    self._trajectory_step += 1
    return next_r, next_i


def install_controls(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Install readout and state probes on the model."""

    result = operators(model)
    for operator in result:
        operator._trajectory_original_depth_weights = operator.depth_weights
        operator._trajectory_original_state_step = operator._state_depth_step
        operator._trajectory_mode = "original"
        operator._trajectory_capture = False
        operator._trajectory_step = 0
        operator._trajectory_records = []
        operator._trajectory_first_delta = None
        operator.depth_weights = types.MethodType(controlled_depth_weights, operator)
        operator._state_depth_step = types.MethodType(controlled_state_step, operator)
    return result


def set_mode(items: list[torch.nn.Module], mode: str, capture: bool = False) -> None:
    """Set one readout mode and clear captured records."""

    for operator in items:
        operator._trajectory_mode = mode
        operator._trajectory_capture = capture
        operator._trajectory_step = 0
        operator._trajectory_records = []
        operator._trajectory_first_delta = None


def prediction(model: torch.nn.Module, source: torch.Tensor, pad32: Callable) -> torch.Tensor:
    """Run the model and remove padding."""

    padded, height, width = pad32(source)
    value = model(padded)
    return (value[-1] if isinstance(value, list) else value)[:, :, :height, :width]


def oracle_channel_affine(prediction_value: torch.Tensor, target: torch.Tensor):
    """Fit a per-image, per-channel affine control."""

    prediction64, target64 = prediction_value.double(), target.double()
    prediction_mean = prediction64.mean(dim=(-2, -1), keepdim=True)
    target_mean = target64.mean(dim=(-2, -1), keepdim=True)
    centered_prediction = prediction64 - prediction_mean
    centered_target = target64 - target_mean
    denominator = centered_prediction.square().sum(dim=(-2, -1), keepdim=True)
    slope = torch.where(
        denominator > 1.0e-30,
        (centered_prediction * centered_target).sum(dim=(-2, -1), keepdim=True) / denominator,
        torch.zeros_like(denominator),
    )
    bias = target_mean - slope * prediction_mean
    calibrated = (slope * prediction64 + bias).to(prediction_value.dtype)
    return calibrated, slope.flatten().tolist(), bias.flatten().tolist()


def summarize_records(records: list[dict[str, Any]], steps: int) -> list[dict[str, Any]]:
    """Summarize state increments by depth."""

    output = []
    for step in range(steps):
        selected = [row for row in records if row["step"] == step]
        row: dict[str, Any] = {"step": step + 1, "count": len(selected)}
        for key in ("delta_rms", "delta_to_first_cosine", "delta_local_variation_ratio"):
            values = torch.tensor([item[key] for item in selected], dtype=torch.float64)
            row[key] = {"median": float(values.median()), "mean": float(values.mean())}
        output.append(row)
    return output


def summarize_weights(items: list[torch.nn.Module]) -> list[dict[str, Any]]:
    """Summarize the learned depth weights."""

    all_weights = []
    for operator in items:
        reference = next(operator.parameters())
        all_weights.append(operator._trajectory_original_depth_weights(reference).detach().double())
    matrix = torch.stack(all_weights)
    return [
        {
            "step": step + 1,
            "mean": float(matrix[:, step].mean()),
            "min": float(matrix[:, step].min()),
            "max": float(matrix[:, step].max()),
        }
        for step in range(matrix.shape[1])
    ]


def evaluate(
    define_network: Callable,
    image_tensor: Callable,
    resize_long_side: Callable,
    pad32: Callable,
    metric: Callable,
    config: Path,
    checkpoint: Path,
    data_root: Path,
    limit: int,
    affine_control: bool = False,
) -> dict[str, Any]:
    """Evaluate readout permutations on the selected test pairs."""

    model = load_model(define_network, config, checkpoint)
    hazy = data_root / "test" / "hazy"
    gt = data_root / "test" / "gt"
    names = sorted(path.name for path in hazy.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[: limit or None]
    baseline_predictions: dict[str, torch.Tensor] = {}
    baseline_scores: dict[str, tuple[float, float]] = {}
    with torch.inference_mode():
        for name in names:
            source = resize_long_side(image_tensor(hazy / name))
            target = resize_long_side(image_tensor(gt / name))
            value = prediction(model, source, pad32)
            baseline_predictions[name] = value.detach().clone()
            baseline_scores[name] = metric(value, target)

    items = install_controls(model)
    weight_profile = summarize_weights(items)
    records: list[dict[str, Any]] = []
    replay_errors: list[float] = []
    scores = {mode: [] for mode in ("baseline", "original", "reverse", "cyclic", "uniform")}
    oracle_scores = {mode: [] for mode in ("reverse", "cyclic", "uniform")} if affine_control else None
    for name in names:
        scores["baseline"].append({"image": name, "psnr": baseline_scores[name][0], "ssim": baseline_scores[name][1]})
    with torch.inference_mode():
        for name in names:
            source = resize_long_side(image_tensor(hazy / name))
            target = resize_long_side(image_tensor(gt / name))
            for mode in ("original", "reverse", "cyclic", "uniform"):
                set_mode(items, mode, capture=(mode == "original"))
                value = prediction(model, source, pad32)
                psnr, ssim = metric(value, target)
                scores[mode].append({"image": name, "psnr": psnr, "ssim": ssim})
                if oracle_scores is not None:
                    calibrated, slope, bias = oracle_channel_affine(value, target)
                    calibrated_psnr, calibrated_ssim = metric(calibrated, target)
                    oracle_scores[mode].append({"image": name, "psnr": calibrated_psnr, "ssim": calibrated_ssim, "slope_rgb": slope, "bias_rgb": bias})
                if mode == "original":
                    replay_errors.append(float((value - baseline_predictions[name]).abs().max()))
                    for block, operator in enumerate(items):
                        if len(operator._trajectory_records) != operator.K:
                            raise RuntimeError(f"block {block}: captured {len(operator._trajectory_records)} of {operator.K} increments")
                        records.extend({"image": name, "block": block, **row} for row in operator._trajectory_records)
    variants = {
        mode: {
            "images": rows,
            "psnr": sum(row["psnr"] for row in rows) / len(rows),
            "ssim": sum(row["ssim"] for row in rows) / len(rows),
        }
        for mode, rows in scores.items()
    }
    base = variants["baseline"]["psnr"]
    control_delta = variants["original"]["psnr"] - base
    control_pass = max(replay_errors) <= 1.0e-6 and abs(control_delta) <= 1.0e-5
    per_image = []
    for index, name in enumerate(names):
        image_base = variants["baseline"]["images"][index]["psnr"]
        delta = sum(variants[mode]["images"][index]["psnr"] - image_base for mode in ("reverse", "cyclic")) / 2.0
        per_image.append({"image": name, "mean_temporal_permutation_delta_psnr": delta})
    mean_delta = sum(row["mean_temporal_permutation_delta_psnr"] for row in per_image) / len(per_image)
    negative = sum(row["mean_temporal_permutation_delta_psnr"] < 0.0 for row in per_image)
    primary_verdict = "CANARY_ONLY" if len(per_image) < 5 else "SUPPORTED" if control_pass and mean_delta <= -0.10 and negative >= 4 else "NOT_SUPPORTED"
    result: dict[str, Any] = {
        "evidence_label": "FIXED_WEIGHT_CAUSAL_DIAGNOSTIC",
        "limit": limit,
        "operator_count": len(items),
        "weight_profile": weight_profile,
        "increment_profile": summarize_records(records, items[0].K),
        "variants": variants,
        "negative_control": {"max_abs_output_error": max(replay_errors), "mean_psnr_delta": control_delta, "pass": control_pass},
        "primary": {"mean_temporal_permutation_delta_psnr": mean_delta, "negative_images": negative, "per_image": per_image, "verdict": primary_verdict},
        "uniform_terminal_delta_psnr": variants["uniform"]["psnr"] - base,
    }
    if oracle_scores is not None:
        oracle_variants = {
            mode: {"images": rows, "psnr": sum(row["psnr"] for row in rows) / len(rows), "ssim": sum(row["ssim"] for row in rows) / len(rows)}
            for mode, rows in oracle_scores.items()
        }
        cyclic_deltas = [row["psnr"] - variants["baseline"]["images"][index]["psnr"] for index, row in enumerate(oracle_variants["cyclic"]["images"])]
        cyclic_delta = oracle_variants["cyclic"]["psnr"] - base
        cyclic_negative = sum(delta < 0.0 for delta in cyclic_deltas)
        oracle_verdict = "CANARY_ONLY" if len(cyclic_deltas) < 5 else "SUPPORTED" if cyclic_delta <= -0.10 and cyclic_negative >= 4 else "NOT_SUPPORTED"
        result["oracle_affine_control"] = {
            "fit": "per-image per-RGB-channel least-squares affine fit to ground truth",
            "baseline_reference_psnr": base,
            "variants": oracle_variants,
            "primary": {"variant": "cyclic", "delta_psnr": cyclic_delta, "negative_images": cyclic_negative, "per_image_delta_psnr": cyclic_deltas, "verdict": oracle_verdict},
        }
    del model
    return result


def markdown(result: dict[str, Any]) -> str:
    """Write a short trajectory report."""

    base = result["variants"]["baseline"]["psnr"]
    lines = [
        "# Trajectory-readout temporal alignment",
        "",
        "**No training was started.** Reverse and cyclic variants keep each learned readout weight and its sum. They change which state increment receives each weight.",
        "",
        "| variant | PSNR | SSIM | delta PSNR |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("baseline", "original", "reverse", "cyclic", "uniform"):
        row = result["variants"][mode]
        lines.append(f"| {mode} | {row['psnr']:.4f} | {row['ssim']:.5f} | {row['psnr'] - base:+.4f} |")
    control = result["negative_control"]
    primary = result["primary"]
    lines += [
        "",
        f"Negative-control instrumentation: **{'PASS' if control['pass'] else 'FAIL'}**; max output error {control['max_abs_output_error']:.3g}; PSNR delta {control['mean_psnr_delta']:+.6f} dB.",
        "",
        f"Primary verdict: **{primary['verdict']}**; two-permutation mean {primary['mean_temporal_permutation_delta_psnr']:+.4f} dB; negative on {primary['negative_images']}/{len(primary['per_image'])} images.",
        "",
        "## Learned depth weights and state increments",
        "",
        "| depth | mean weight | median increment RMS | median cosine to first increment | median local-variation ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for weight, increment in zip(result["weight_profile"], result["increment_profile"]):
        lines.append(f"| {weight['step']} | {weight['mean']:.4f} | {increment['delta_rms']['median']:.5f} | {increment['delta_to_first_cosine']['median']:.4f} | {increment['delta_local_variation_ratio']['median']:.4f} |")
    lines += [
        "",
        "Uniform weights equal one and telescope to the terminal state. The permutations keep the same coefficients but attach them to different increments. This is a fixed-weight diagnostic. It does not prove a retrained effect.",
    ]
    if "oracle_affine_control" in result:
        control = result["oracle_affine_control"]
        lines += [
            "",
            "## Oracle gain/bias control",
            "",
            "Each permuted prediction receives a per-image, per-channel affine fit to ground truth. This control removes global color gain and bias.",
            "",
            "| variant after affine fit | PSNR | SSIM | delta PSNR |",
            "|---|---:|---:|---:|",
        ]
        for mode in ("reverse", "cyclic", "uniform"):
            row = control["variants"][mode]
            lines.append(f"| {mode} | {row['psnr']:.4f} | {row['ssim']:.5f} | {row['psnr'] - base:+.4f} |")
        primary = control["primary"]
        lines += ["", f"Oracle cyclic verdict: **{primary['verdict']}**; delta {primary['delta_psnr']:+.4f} dB; negative on {primary['negative_images']}/{len(primary['per_image_delta_psnr'])} images."]
    return "\n".join(lines) + "\n"


def self_check() -> None:
    """Check the permutation and affine controls."""

    weights = torch.tensor([0.2, 0.3, 0.5, 1.0, 3.0])
    for changed in (weights.flip(0), weights.roll(-1, 0)):
        assert torch.equal(weights.sort().values, changed.sort().values)
        assert torch.allclose(weights.sum(), changed.sum(), rtol=1.0e-7, atol=1.0e-7)
    states = torch.randn(6, 3)
    increments = states[1:] - states[:-1]
    assert torch.allclose(increments.sum(0), states[-1] - states[0])
    target = torch.randn(1, 3, 7, 9)
    calibrated, _, _ = oracle_channel_affine((target - 0.2) / 1.7, target)
    assert torch.allclose(calibrated, target, rtol=1.0e-5, atol=1.0e-5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the trajectory readout diagnostic on CPU.")
    parser.add_argument("--release-root", type=Path, default=_default_release_root())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--affine-control", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    release_root = args.release_root.expanduser().resolve()
    config = (args.config or _default_config(release_root)).expanduser().resolve()
    checkpoint = (args.checkpoint or _default_checkpoint(release_root)).expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (args.output_dir or release_root / "diagnostic_outputs" / "trajectory_readout").expanduser().resolve()
    for path, label in ((config, "config"), (checkpoint, "checkpoint"), (data_root, "data root")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.threads:
        torch.set_num_threads(args.threads)
    self_check()
    runtime = release_root / "tasks" / "restoration" / "runtime"
    task_dir = release_root / "tasks" / "restoration"
    sys.path[:0] = [str(runtime.resolve()), str(task_dir.resolve())]
    from basicsr.models.archs import define_network
    from evaluate_ohaze import image_tensor, metric, pad32, resize_long_side

    result = evaluate(define_network, image_tensor, resize_long_side, pad32, metric, config, checkpoint, data_root, args.limit, args.affine_control)
    result.update({"release_root": str(release_root), "config": str(config), "checkpoint": str(checkpoint), "data_root": str(data_root), "method_reference": "research_outputs/trajectory_readout_temporal_alignment_cpu_20260904/run_probe.py", "device": "cpu", "training": False})
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_limit{args.limit}" if args.limit else "") + ("_affine_control" if args.affine_control else "")
    (output_dir / f"result{suffix}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"result{suffix}.md").write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "verdict": result["primary"]["verdict"], "limit": args.limit}, indent=2))


if __name__ == "__main__":
    main()
