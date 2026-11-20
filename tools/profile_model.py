"""Count parameters and MACs for a complete paper model."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeAlias

import torch
import yaml
from fvcore.nn import FlopCountAnalysis
from torch import nn


LogitsFunction: TypeAlias = Callable[[nn.Module, Any, torch.Tensor], torch.Tensor]
ProfileRecord: TypeAlias = dict[str, int | str]

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tasks" / "restoration" / "runtime"
sys.path[:0] = [str(ROOT), str(RUNTIME)]


class SegmentationForward(nn.Module):
    def __init__(self, model: nn.Module, cfg: Any, logits: LogitsFunction) -> None:
        super().__init__()
        self.model, self.cfg, self.logits = model, cfg, logits

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.logits(self.model, self.cfg, image)


def restoration_model(config_path: Path, steps: int) -> nn.Module:
    sys.path[:0] = [str(ROOT), str(RUNTIME)]
    from basicsr.models.archs import define_network

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if "K" in config["network_g"]:
        config["network_g"]["K"] = steps
    with contextlib.redirect_stdout(io.StringIO()):
        return define_network(dict(config["network_g"]))


def classification_model(config_path: Path, mixer: str, steps: int) -> nn.Module:
    task_root = ROOT / "tasks" / "classification"
    sys.path[:0] = [str(ROOT), str(task_root)]
    from carrier.poolformer_s12 import build_metaformer_s12

    config = json.loads(config_path.read_text(encoding="utf-8"))
    arm = {"native": "poolformerv2", "cspn_style": "cspn_style_k5", "ours": "radial_profiled"}.get(mixer, mixer)
    if arm not in config["arms"]:
        raise ValueError(f"unknown classification mixer: {mixer}")
    arm_config = dict(config["arms"][arm])
    if mixer in {"ours", "cspn_style"}:
        arm_config["K"] = steps
    return build_metaformer_s12(arm_config)


def segmentation_model(mixer: str, steps: int) -> nn.Module:
    task_root = ROOT / "tasks" / "segmentation"
    sys.path[:0] = [str(ROOT), str(task_root)]
    from train import build_model, logits

    repo = task_root / "carrier" / "BPNet"
    cfg, model, _, _ = build_model(repo, 896, mixer, steps)
    return SegmentationForward(model, cfg, logits)


def propagation_macs(
    model: nn.Module, sample: torch.Tensor
) -> tuple[int, list[ProfileRecord]]:
    """Count local propagation MACs that fvcore does not trace."""
    records: list[ProfileRecord] = []
    hooks = []
    for module in model.modules():
        if type(module).__name__ in {"CSPNStyleMixer", "CSPNOriginalStyleMixer"}:
            def record_cspn(current, inputs, _output):
                height, width = inputs[0].shape[-2:]
                records.append(
                    {
                        "kind": "real_cspn_3x3",
                        "channels": int(current.channels),
                        "steps": int(current.steps),
                        "height": int(height),
                        "width": int(width),
                        "macs_per_channel_step": 9,
                    }
                )

            hooks.append(module.register_forward_hook(record_cspn))
            continue
        if not hasattr(module, "state_channels") or not hasattr(module, "K"):
            continue
        group_count = getattr(
            module, "radial_group_count", getattr(module, "selective_group_count", None)
        )
        if group_count is None:
            continue

        def record(current, inputs, _output):
            height, width = inputs[0].shape[-2:]
            records.append(
                {
                    "kind": "real_pair_3x3" if hasattr(current, "pair_layout_version") else "complex_3x3",
                    "channels": int(current.state_channels),
                    "groups": int(
                        current.radial_group_count
                        if hasattr(current, "radial_group_count")
                        else current.selective_group_count
                    ),
                    "steps": int(current.K),
                    "height": int(height),
                    "width": int(width),
                    "macs_per_channel_step": 36,
                }
            )

        hooks.append(module.register_forward_hook(record))
    with torch.inference_mode():
        model(sample)
    for hook in hooks:
        hook.remove()
    total = 0
    for row in records:
        d, steps = row["channels"], row["steps"]
        pixels = row["height"] * row["width"]
        total += pixels * steps * row["macs_per_channel_step"] * d
    return total, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("restoration", "segmentation", "classification"), required=True)
    parser.add_argument("--mixer", choices=("native", "cspn_style", "dwconv", "mdta", "swin", "ss2d", "ours", "mair"), required=True)
    parser.add_argument("--config", type=Path, help="restoration or classification configuration")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--propagation-steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task == "restoration":
        if args.config is None:
            raise ValueError("restoration requires --config")
        model = restoration_model(args.config, args.propagation_steps)
    elif args.task == "classification":
        if args.config is None:
            raise ValueError("classification requires --config")
        model = classification_model(args.config, args.mixer, args.propagation_steps)
    else:
        model = segmentation_model(args.mixer, args.propagation_steps)
    device = torch.device(args.device)
    model.to(device).eval()
    sample = torch.randn(1, 3, args.height, args.width, device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analysis = FlopCountAnalysis(model, sample)
        if args.mixer in {"ss2d", "mair"}:
            from third_party.VMamba.classification.models.csms6s import selective_scan_flop_jit

            analysis.set_op_handle(
                "prim::PythonOp.SelectiveScanCuda",
                partial(selective_scan_flop_jit, backend="prefixsum", verbose=False),
            )
            analysis.set_op_handle(
                "prim::PythonOp._SelectiveScanOflex",
                partial(selective_scan_flop_jit, backend="prefixsum", verbose=False),
            )
            analysis.set_op_handle(
                "aten::silu", None, "aten::neg", None, "aten::exp", None, "aten::flip", None
            )
        analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False).tracer_warnings("none")
        raw_traced = int(round(analysis.total()))
        by_operator = {name: int(round(count)) for name, count in analysis.by_operator().items()}
        # Use the restoration paper's arithmetic-only count. The fvcore
        # defaults also estimate some normalization, pooling, and resize ops.
        excluded = {}
        if args.task == "restoration":
            excluded_names = {
                "layer_norm", "batch_norm", "group_norm", "instance_norm",
                "adaptive_avg_pool2d", "upsample_bilinear2d", "upsample_nearest2d",
            }
            excluded = {name: count for name, count in by_operator.items() if name in excluded_names}
        traced = raw_traced - sum(excluded.values())
        unsupported = {name: int(count) for name, count in analysis.unsupported_ops().items()}
    custom, records = propagation_macs(model, sample)
    result = {
        "task": args.task,
        "input": [1, 3, args.height, args.width],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        "fvcore_macs": traced,
        "fvcore_supported_ops_raw": raw_traced,
        "by_operator_raw": by_operator,
        "excluded_supported_ops": excluded,
        "propagation_macs": custom,
        "macs": traced + custom,
        "macs_g": (traced + custom) / 1e9,
        "propagation_records": records,
        "unsupported_ops": unsupported,
        "convention": (
            "one multiply-accumulate is one MAC; Conv2d/Linear/matmul are counted by fvcore; "
            "SS2D uses the official VMamba selective-scan handler; "
            "scalar and two-channel 3x3 propagation are added explicitly; "
            + (
                "restoration excludes normalization, pooling, interpolation, activation, gating, padding and rearrangement"
                if args.task == "restoration"
                else "other tasks retain fvcore-supported normalization, pooling and interpolation; unsupported elementwise operations are excluded"
            )
        ),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
