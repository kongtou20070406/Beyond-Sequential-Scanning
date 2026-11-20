"""Run the published O-HAZE fixed-weight diagnostics on CPU."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import types
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


DIAGONAL = frozenset((1, 3, 6, 8))
AXIAL = frozenset((2, 4, 5, 7))
VARIANTS = ("row_mass", "global_mean_mass", "kernel_rmsnorm", "kernel_layernorm")


def _default_release_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_config(release_root: Path) -> Path:
    return release_root / "configs" / "restoration" / "ohaze" / "O-HAZE_VIVNet_RadialProfiled_K5_P256.yml"


def _default_checkpoint(release_root: Path) -> Path:
    return release_root / "weights" / "ohaze" / "ours" / "seed42" / "last.pth"


def _load_runtime(release_root: Path):
    """Load the network and metric code."""

    runtime = release_root / "tasks" / "restoration" / "runtime"
    task_dir = release_root / "tasks" / "restoration"
    if not runtime.is_dir() or not task_dir.is_dir():
        raise FileNotFoundError(
            f"release runtime is missing under {release_root}; "
            "pass --release-root to the reproduction code directory"
        )
    sys.path[:0] = [str(runtime.resolve()), str(task_dir.resolve())]
    import yaml

    from basicsr.models.archs import define_network
    from evaluate_ohaze import image_tensor, metric, pad32, resize_long_side

    return define_network, yaml, image_tensor, metric, pad32, resize_long_side


def _load_model(define_network: Callable, yaml: Any, config: Path, checkpoint: Path) -> torch.nn.Module:
    network = yaml.safe_load(config.read_text(encoding="utf-8"))["network_g"]
    if network.get("type") == "VIVNetRadialStateDepthProfiledCLPO":
        network["real_pair_state"] = False
    model = define_network(dict(network))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("params_ema", saved.get("params", saved))
    model.load_state_dict(state, strict=True)
    return model.float().eval()


def _image_names(data_root: Path, limit: int) -> list[str]:
    hazy_dir = data_root / "test" / "hazy"
    gt_dir = data_root / "test" / "gt"
    if not hazy_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(
            f"O-HAZE test folders are missing under {data_root}; "
            "expected test/hazy and test/gt"
        )
    names = sorted(
        path.name
        for path in hazy_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if limit:
        names = names[:limit]
    if not names or any(not (gt_dir / name).is_file() for name in names):
        raise RuntimeError("the selected O-HAZE test pairs are empty or unmatched")
    return names


def _evaluate_model(
    model: torch.nn.Module,
    names: list[str],
    data_root: Path,
    image_tensor: Callable,
    metric: Callable,
    pad32: Callable,
    resize_long_side: Callable,
) -> list[dict[str, Any]]:
    hazy_dir, gt_dir = data_root / "test" / "hazy", data_root / "test" / "gt"
    scores: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name in names:
            source = resize_long_side(image_tensor(hazy_dir / name))
            target = resize_long_side(image_tensor(gt_dir / name))
            padded, height, width = pad32(source)
            value = model(padded)
            prediction = (value[-1] if isinstance(value, list) else value)[:, :, :height, :width]
            psnr, ssim = metric(prediction, target)
            scores.append({"image": name, "psnr": psnr, "ssim": ssim})
    return scores


def _reset_timing(operators: list[torch.nn.Module]) -> None:
    for operator in operators:
        operator._timing_previous = None


def _timed_transition(delayed_taps: frozenset[int]):
    """Build the timing test transition."""

    def apply(self, zr, zi, ar, ai, *args, **kwargs):
        if args or any(value is not None for value in kwargs.values()):
            raise RuntimeError("timing diagnostic supports the local F9 transition only")
        from clpo2027.fused_stencil import F9_OFFSETS, _shift_from_padded

        previous = self._timing_previous
        self._timing_previous = (zr, zi)
        delayed_r, delayed_i = previous if previous is not None else (zr, zi)
        height, width = zr.shape[-2:]
        current_r = F.pad(zr, (1, 1, 1, 1))
        current_i = F.pad(zi, (1, 1, 1, 1))
        delayed_r = F.pad(delayed_r, (1, 1, 1, 1))
        delayed_i = F.pad(delayed_i, (1, 1, 1, 1))
        out_r, out_i = torch.zeros_like(zr), torch.zeros_like(zi)
        for tap, (dy, dx) in enumerate(F9_OFFSETS):
            source_r = delayed_r if tap in delayed_taps else current_r
            source_i = delayed_i if tap in delayed_taps else current_i
            sr = _shift_from_padded(source_r, dy, dx, height, width)
            si = _shift_from_padded(source_i, dy, dx, height, width)
            tap_r, tap_i = ar[:, tap], ai[:, tap]
            out_r.add_(tap_r * sr - tap_i * si)
            out_i.add_(tap_r * si + tap_i * sr)
        return out_r, out_i

    return apply


def _ours_operators(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [
        module
        for module in model.modules()
        if hasattr(module, "_state_depth_step") and hasattr(module, "depth_weights")
    ]


def _patch_timing(model: torch.nn.Module, delayed_taps: frozenset[int]) -> list[torch.nn.Module]:
    operators = _ours_operators(model)
    for operator in operators:
        operator.selective_backend = "eager"
        operator.set_transition_backend("eager")
        operator._timing_previous = None
        operator.apply_transition = types.MethodType(_timed_transition(delayed_taps), operator)
    return operators


def _run_delay(
    define_network: Callable,
    yaml: Any,
    config: Path,
    checkpoint: Path,
    data_root: Path,
    names: list[str],
    image_tensor: Callable,
    metric: Callable,
    pad32: Callable,
    resize_long_side: Callable,
) -> dict[str, Any]:
    variants = (
        ("baseline", frozenset()),
        ("diagonal_recurrent_lag1", DIAGONAL),
        ("axial_recurrent_lag1", AXIAL),
    )
    rows: list[dict[str, Any]] = []
    for label, delayed_taps in variants:
        model = _load_model(define_network, yaml, config, checkpoint)
        operators = _patch_timing(model, delayed_taps)
        scores: list[dict[str, Any]] = []
        with torch.inference_mode():
            for name in names:
                _reset_timing(operators)
                source = resize_long_side(image_tensor(data_root / "test" / "hazy" / name))
                target = resize_long_side(image_tensor(data_root / "test" / "gt" / name))
                padded, height, width = pad32(source)
                value = model(padded)
                prediction = (value[-1] if isinstance(value, list) else value)[:, :, :height, :width]
                psnr, ssim = metric(prediction, target)
                scores.append({"image": name, "psnr": psnr, "ssim": ssim})
        rows.append(
            {
                "variant": label,
                "images": scores,
                "psnr": sum(item["psnr"] for item in scores) / len(scores),
                "ssim": sum(item["ssim"] for item in scores) / len(scores),
            }
        )
        del model

    baseline = rows[0]
    diagonal = rows[1]
    deltas = [
        diagonal["images"][index]["psnr"] - baseline["images"][index]["psnr"]
        for index in range(len(names))
    ]
    verdict = "SUPPORTED" if sum(deltas) / len(deltas) <= -0.10 and sum(delta < 0 for delta in deltas) >= 4 else "NOT_SUPPORTED"
    return {
        "evidence_label": "FIXED_WEIGHT_CAUSAL_DIAGNOSTIC",
        "protocol": "first_step_preserved_recurrent_neighbor_timing",
        "device": "cpu",
        "training": False,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "images": names,
        "invariants": [
            "same learned weights and row-mass projection",
            "same K=5 and depth readout",
            "same first transition",
            "same tap count and arithmetic count",
        ],
        "results": rows,
        "diagonal_per_image_delta_psnr": deltas,
        "diagonal_negative_images": sum(delta < 0 for delta in deltas),
        "primary_verdict": verdict,
    }


def _group_major_raw(operator: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(operator, "_group_major_raw"):
        return operator._group_major_raw(hidden)
    raw = operator.coeff_out(hidden)
    batch, _, height, width = raw.shape
    return raw.view(
        batch,
        operator.selective_group_count,
        2,
        operator.tap_count,
        operator.selective_group_size,
        height,
        width,
    ).permute(0, 2, 3, 1, 4, 5, 6).reshape(
        batch, 2 * operator.tap_count * operator.state_channels, height, width
    )


def _raw_coefficients(operator: torch.nn.Module, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw = _group_major_raw(operator, hidden)
    batch, _, height, width = raw.shape
    raw = raw.view(batch, 2, operator.tap_count, operator.state_channels, height, width)
    return raw[:, 0], raw[:, 1]


def _coefficient_sets(
    operator: torch.nn.Module,
    hidden: torch.Tensor,
    original: Callable,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    raw_r, raw_i = _raw_coefficients(operator, hidden)
    projected_r, projected_i = original(hidden)
    eps = torch.finfo(raw_r.dtype).eps
    target_mean = torch.sqrt(projected_r.square() + projected_i.square()).sum(dim=1).mean()

    raw_mass = torch.sqrt(raw_r.square() + raw_i.square()).sum(dim=1, keepdim=True)
    global_scale = target_mean / raw_mass.mean().clamp_min(eps)

    rms = torch.sqrt((raw_r.square() + raw_i.square()).mean(dim=1, keepdim=True) + eps)
    rms_r, rms_i = raw_r / rms, raw_i / rms
    rms_scale = target_mean / torch.sqrt(rms_r.square() + rms_i.square()).sum(dim=1).mean().clamp_min(eps)

    real_coordinates = torch.cat((raw_r, raw_i), dim=1)
    centered = real_coordinates - real_coordinates.mean(dim=1, keepdim=True)
    normalized = centered / torch.sqrt(centered.square().mean(dim=1, keepdim=True) + eps)
    layer_r, layer_i = normalized[:, : operator.tap_count], normalized[:, operator.tap_count :]
    layer_scale = target_mean / torch.sqrt(layer_r.square() + layer_i.square()).sum(dim=1).mean().clamp_min(eps)

    return {
        "row_mass": (projected_r, projected_i),
        "global_mean_mass": (raw_r * global_scale, raw_i * global_scale),
        "kernel_rmsnorm": (rms_r * rms_scale, rms_i * rms_scale),
        "kernel_layernorm": (layer_r * layer_scale, layer_i * layer_scale),
    }


def _patch_variant(operator: torch.nn.Module, name: str) -> None:
    original = operator._lightweight_coefficients

    def coefficients(self, hidden):
        return _coefficient_sets(self, hidden, original)[name]

    operator._lightweight_coefficients = types.MethodType(coefficients, operator)


def _matrix_metrics(operator: torch.nn.Module, ar: torch.Tensor, ai: torch.Tensor, channel: int, crop: int = 7) -> dict[str, Any]:
    height, width = ar.shape[-2:]
    y0, x0 = (height - crop) // 2, (width - crop) // 2
    coefficients = torch.complex(
        ar[0, :, channel, y0 : y0 + crop, x0 : x0 + crop],
        ai[0, :, channel, y0 : y0 + crop, x0 : x0 + crop],
    ).to(torch.complex64)
    matrix = torch.zeros((crop * crop, crop * crop), dtype=torch.complex64)
    for y in range(crop):
        for x in range(crop):
            row = y * crop + x
            for tap, (dy, dx) in enumerate(operator.offsets):
                source_y, source_x = y + dy, x + dx
                if 0 <= source_y < crop and 0 <= source_x < crop:
                    matrix[row, source_y * crop + source_x] += coefficients[tap, y, x]
    power = torch.eye(crop * crop, dtype=torch.complex64)
    norms: dict[str, float] = {}
    for depth in range(1, 8):
        power = power @ matrix
        if depth in {1, 3, 5, 7}:
            norms[str(depth)] = float(power.abs().sum(dim=1).max())
    return {
        "channel": channel,
        "spectral_radius": float(torch.linalg.eigvals(matrix).abs().max()),
        "power_infinity_norm": norms,
    }


def _capture_operator_probe(model: torch.nn.Module, source: torch.Tensor) -> list[dict[str, Any]]:
    operators = _ours_operators(model)
    records: list[dict[str, Any]] = []
    handles = []
    for block_index, operator in enumerate(operators):
        original = operator._lightweight_coefficients

        def hook(module, inputs, _output, index=block_index, bound_original=original):
            packed, _fr, _fi, _attention = module._drive_with_attention(inputs[0])
            hidden = module._coefficient_hidden(inputs[0], packed)
            variants = _coefficient_sets(module, hidden, bound_original)
            block = {"block": index, "rho_max": float(module.rho_max), "variants": {}}
            channels = sorted({0, module.state_channels // 2})
            for name, (ar, ai) in variants.items():
                mass = torch.sqrt(ar.square() + ai.square()).sum(dim=1)
                block["variants"][name] = {
                    "row_mass_mean": float(mass.mean()),
                    "row_mass_p99": float(torch.quantile(mass.flatten(), 0.99)),
                    "row_mass_max": float(mass.max()),
                    "fraction_above_rho": float((mass > module.rho_max).float().mean()),
                    "matrices": [_matrix_metrics(module, ar, ai, channel) for channel in channels],
                }
            records.append(block)

        handles.append(operator.register_forward_hook(hook))
    with torch.inference_mode():
        model(source)
    for handle in handles:
        handle.remove()
    return records


def _summarize_operator_probe(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in VARIANTS:
        fractions = [block["variants"][name]["fraction_above_rho"] for block in blocks]
        masses = [block["variants"][name]["row_mass_max"] for block in blocks]
        powers = {depth: [] for depth in ("1", "3", "5", "7")}
        for block in blocks:
            for matrix in block["variants"][name]["matrices"]:
                for depth, value in matrix["power_infinity_norm"].items():
                    powers[depth].append(value)
        summary[name] = {
            "mean_fraction_above_rho": sum(fractions) / len(fractions),
            "max_row_mass": max(masses),
            "median_power_infinity_norm": {
                depth: statistics.median(values) for depth, values in powers.items()
            },
        }
    for name in VARIANTS[1:]:
        ratios: dict[str, float] = {}
        for depth in ("1", "3", "5", "7"):
            paired = []
            for block in blocks:
                for index, matrix in enumerate(block["variants"][name]["matrices"]):
                    baseline = block["variants"]["row_mass"]["matrices"][index]["power_infinity_norm"][depth]
                    paired.append(matrix["power_infinity_norm"][depth] / max(baseline, 1e-30))
            ratios[depth] = statistics.median(paired)
        summary[name]["median_matched_ratio_vs_row_mass"] = ratios
    supported = all(
        summary[name]["median_matched_ratio_vs_row_mass"]["5"]
        > summary[name]["median_matched_ratio_vs_row_mass"]["1"]
        for name in ("global_mean_mass", "kernel_rmsnorm")
    )
    return {"variants": summary, "reuse_verdict": "SUPPORTED" if supported else "NOT_SUPPORTED"}


def _run_normalization(
    define_network: Callable,
    yaml: Any,
    config: Path,
    checkpoint: Path,
    data_root: Path,
    names: list[str],
    image_tensor: Callable,
    metric: Callable,
    pad32: Callable,
    resize_long_side: Callable,
) -> dict[str, Any]:
    performances: list[dict[str, Any]] = []
    operator_blocks: list[dict[str, Any]] | None = None
    for variant in VARIANTS:
        model = _load_model(define_network, yaml, config, checkpoint)
        operators = _ours_operators(model)
        if variant != "row_mass":
            for operator in operators:
                _patch_variant(operator, variant)
        scores: list[dict[str, Any]] = []
        with torch.inference_mode():
            for image_index, name in enumerate(names):
                source = resize_long_side(image_tensor(data_root / "test" / "hazy" / name))
                target = resize_long_side(image_tensor(data_root / "test" / "gt" / name))
                padded, height, width = pad32(source)
                if variant == "row_mass" and image_index == 0:
                    operator_blocks = _capture_operator_probe(model, padded)
                value = model(padded)
                prediction = (value[-1] if isinstance(value, list) else value)[:, :, :height, :width]
                psnr, ssim = metric(prediction, target)
                scores.append({"image": name, "psnr": psnr, "ssim": ssim})
        performances.append(
            {
                "variant": variant,
                "images": scores,
                "psnr": sum(item["psnr"] for item in scores) / len(scores),
                "ssim": sum(item["ssim"] for item in scores) / len(scores),
            }
        )
        del model
    if operator_blocks is None:
        raise RuntimeError("operator probe was not captured")
    return {
        "evidence_label": "FIXED_WEIGHT_DIAGNOSTIC",
        "protocol": "row_mass_alternatives_and_cropped_operator_probe",
        "device": "cpu",
        "training": False,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "images": names,
        "variants": {
            "row_mass": "trained per-row complex L1 projection",
            "global_mean_mass": "one positive scalar per operator and input",
            "kernel_rmsnorm": "RMS-normalize the nine complex taps per row",
            "kernel_layernorm": "LayerNorm the 18 real coordinates per row",
        },
        "performance": performances,
        "operator_probe": {
            "blocks": operator_blocks,
            "summary": _summarize_operator_probe(operator_blocks),
        },
        "interpretation_boundary": [
            "This is a fixed-weight intervention, not a retrained architecture comparison.",
            "Kernel RMSNorm and Kernel LayerNorm are coefficient-space substitutions.",
            "Kernel RMSNorm independent retraining is not included in the paper evidence.",
        ],
    }


def _delay_markdown(result: dict[str, Any]) -> str:
    rows = result["results"]
    baseline = rows[0]["psnr"]
    finding = result["primary_verdict"]
    lines = [
        "# O-HAZE fixed-weight recurrent timing",
        "",
        "**CPU only. No training was started.** The first transition is unchanged.",
        "",
        "| variant | images | PSNR | SSIM | delta PSNR |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['variant']} | {len(row['images'])} | {row['psnr']:.4f} | {row['ssim']:.5f} | {row['psnr'] - baseline:+.4f} |"
        for row in rows
    )
    lines += [
        "",
        f"Diagonal lag-1 verdict: **{finding}**; negative on {result['diagonal_negative_images']}/{len(result['images'])} images.",
        "",
        "The intervention delays diagonal or axial neighbor reads from step 2 onward. It keeps the learned weights, row-mass projection, K=5, and tap arithmetic. It is not SS2D emulation and not a retrained comparison.",
    ]
    return "\n".join(lines) + "\n"


def _normalization_markdown(result: dict[str, Any]) -> str:
    rows = result["performance"]
    lines = [
        "# O-HAZE fixed-weight coefficient normalization",
        "",
        "**CPU only. No training was started.** These are checkpoint interventions.",
        "",
        "| variant | images | PSNR | SSIM |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['variant']} | {len(row['images'])} | {row['psnr']:.4f} | {row['ssim']:.5f} |"
        for row in rows
    )
    summary = result["operator_probe"]["summary"]
    lines += [
        "",
        "## Cropped operator probe",
        "",
        "| variant | mean rows above rho | max row mass | median ||A||inf | median ||A^5||inf | matched K1 | matched K5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["variants"].items():
        ratios = row.get("median_matched_ratio_vs_row_mass", {"1": 1.0, "5": 1.0})
        lines.append(
            f"| {name} | {row['mean_fraction_above_rho']:.4%} | {row['max_row_mass']:.4f} | "
            f"{row['median_power_infinity_norm']['1']:.5f} | {row['median_power_infinity_norm']['5']:.5f} | "
            f"{ratios['1']:.3f} | {ratios['5']:.3f} |"
        )
    lines += [
        "",
        f"Repeated-reuse probe verdict: **{summary['reuse_verdict']}**.",
        "",
        "Kernel RMSNorm and Kernel LayerNorm change coefficient space in a fixed trained model. They do not represent independent retraining.",
    ]
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-weight O-HAZE delay and normalization diagnostics on CPU.")
    parser.add_argument("--mode", choices=("all", "delay", "normalization"), default="all")
    parser.add_argument("--release-root", type=Path, default=_default_release_root())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Use the first N test pairs; 0 uses all pairs.")
    parser.add_argument("--threads", type=int, default=0, help="Set CPU thread count; 0 keeps the PyTorch default.")
    args = parser.parse_args()

    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    release_root = args.release_root.expanduser().resolve()
    config = (args.config or _default_config(release_root)).expanduser().resolve()
    checkpoint = (args.checkpoint or _default_checkpoint(release_root)).expanduser().resolve()
    data_root = (args.data_root or (release_root / "data" / "ohaze")).expanduser().resolve()
    output_dir = (args.output_dir or (release_root / "diagnostic_outputs" / "ohaze_fixed_weight")).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if args.threads:
        torch.set_num_threads(args.threads)
    names = _image_names(data_root, args.limit)
    define_network, yaml, image_tensor, metric, pad32, resize_long_side = _load_runtime(release_root)

    run_info = {
        "mode": args.mode,
        "device": "cpu",
        "release_root": str(release_root),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "limit": args.limit,
        "images": names,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "run.json", run_info)

    if args.mode in {"all", "delay"}:
        delay = _run_delay(
            define_network, yaml, config, checkpoint, data_root, names,
            image_tensor, metric, pad32, resize_long_side,
        )
        _write_json(output_dir / "delay.json", delay)
        (output_dir / "delay.md").write_text(_delay_markdown(delay), encoding="utf-8")
    if args.mode in {"all", "normalization"}:
        normalization = _run_normalization(
            define_network, yaml, config, checkpoint, data_root, names,
            image_tensor, metric, pad32, resize_long_side,
        )
        _write_json(output_dir / "normalization.json", normalization)
        (output_dir / "normalization.md").write_text(_normalization_markdown(normalization), encoding="utf-8")
    print(json.dumps(run_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
