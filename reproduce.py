"""Unified entry point for the paper experiments.

Examples:
python reproduce.py --task restoration --dataset ohaze --mixer ours --data-root data/ohaze
python reproduce.py --task segmentation --mixer ours --data-root data/cityscapes --seed 304
python reproduce.py --task classification --mixer ours --data-root data/imagenet100
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, TypeAlias

import yaml


Config: TypeAlias = dict[str, Any]
AblationSpec: TypeAlias = tuple[str, str | None, int, float]

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "tasks" / "restoration" / "runtime"
MIXERS = ("native", "cspn_style", "dwconv", "mdta", "swin", "ss2d", "ours", "mair")
OHAZE_CONFIGS: dict[str, str] = {
    "mair": "O-HAZE_VIVNet_Matched_MaIR_P256.yml",
    "native": "O-HAZE_VIVNet_Baseline.yml",
    "cspn_style": "O-HAZE_VIVNet_Matched_CSPN_STYLE_P256.yml",
    "dwconv": "O-HAZE_VIVNet_Matched_DWConv_P256.yml",
    "mdta": "O-HAZE_VIVNet_Matched_MDTA_P256.yml",
    "swin": "O-HAZE_VIVNet_Matched_Swin_P256.yml",
    "ss2d": "O-HAZE_VIVNet_Matched_SS2D_P256.yml",
    "ours": "O-HAZE_VIVNet_RadialProfiled_K5_P256.yml",
}
OHAZE_ABLATIONS: dict[str, AblationSpec] = {
    "p0t_diagonal_delay": ("VIVNetRadialProfiledAblation", "diagonal_lag1_transition", 5, 0.95),
    "p0s_state_dynamic": ("VIVNetRadialProfiledAblation", "state_dynamic_transition", 5, 0.95),
    "matched_feedforward": ("VIVNetRadialProfiledAblation", "matched_feedforward", 5, 0.95),
    "step_conditioned_rules": ("VIVNetRadialProfiledAblation", "step_conditioned_transition", 5, 0.95),
    "first_step_drive": ("VIVNetRadialProfiledAblation", "first_drive", 5, 0.95),
    "identity_retention": ("VIVNetRadialProfiledAblation", "identity_memory", 5, 0.95),
    "terminal_readout": ("VIVNetRadialProfiledAblation", "uniform_readout", 5, 0.95),
    "identity_retention_terminal_readout": ("VIVNetRadialProfiledAblation", "identity_memory_uniform_readout", 5, 0.95),
    "static_rules": ("VIVNetRadialProfiledAblation", "static_transition", 5, 0.95),
    "equal_parameter_real": ("VIVNetRadialProfiledAblation", "real_f9", 5, 0.95),
    "no_row_mass_projection": ("VIVNetRadialProfiledAblation", "no_projection", 5, 0.95),
    "alternative_generator": ("VIVNetRadialProfiledAblation", "compiler_3x3x2", 5, 0.95),
    "conjugate_symmetric": ("VIVNetPhysicalQ9RadialProfiledCLPO", None, 5, 0.95),
    "depth_k1": ("VIVNetRadialStateDepthProfiledCLPO", None, 1, 0.95),
    "depth_k3": ("VIVNetRadialStateDepthProfiledCLPO", None, 3, 0.95),
    "depth_k7": ("VIVNetRadialStateDepthProfiledCLPO", None, 7, 0.95),
}
GOPRO_CONFIGS: dict[str, str] = {
    "native": "original.yml",
    "cspn_style": "cspn_style.yml",
    "ss2d": "ss2d.yml",
    "ours": "ours.yml",
}
CLASSIFICATION_ARMS: dict[str, str] = {
    "native": "poolformerv2",
    "cspn_style": "cspn_style_k5",
    "dwconv": "dwconv",
    "mdta": "mdta",
    "swin": "swin",
    "ss2d": "ss2d",
    "ours": "radial_profiled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, and profile the paper models")
    parser.add_argument("--task", choices=("restoration", "segmentation", "classification"), required=True)
    parser.add_argument("--dataset", choices=("ohaze", "gopro", "cityscapes", "imagenet100"))
    parser.add_argument("--mixer", choices=MIXERS, default="ours")
    parser.add_argument("--ablation", choices=tuple(OHAZE_ABLATIONS))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--source-root", type=Path, help="official dataset root used by prepare")
    parser.add_argument("--copy-data", action="store_true", help="copy data instead of creating hard links")
    parser.add_argument("--eval-data-root", type=Path, help="nested official GoPro test root; defaults to data-root")
    parser.add_argument("--seed", type=int, help="override the paper random seed")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="checkpoint used for standalone evaluation")
    parser.add_argument(
        "--action", choices=("all", "train", "eval", "profile", "prepare"), default="eval"
    )
    parser.add_argument("--propagation-steps", type=int, help="number K of synchronous updates")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true", help="write resolved configs and print commands only")
    return parser.parse_args()


def run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    if "--checkpoint" in command:
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        if not checkpoint.is_file() and not dry_run:
            raise FileNotFoundError(f"The checkpoint is missing: {checkpoint}")
    print(json.dumps(command, ensure_ascii=False))
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def require_path(value: Path | None, option: str) -> Path:
    if value is None:
        raise ValueError(f"this action requires {option}")
    return value


def propagation_steps(args: argparse.Namespace, dataset: str) -> int:
    if args.propagation_steps is not None:
        if args.propagation_steps < 1:
            raise ValueError("--propagation-steps must be at least 1")
        return args.propagation_steps
    return 5


def released_checkpoint(
    dataset: str, mixer: str, seed: int, steps: int
) -> Path | None:
    if dataset == "cityscapes" and mixer == "ours":
        setting = f"k{steps}_seed{seed}_repaired/best.pt"
    else:
        role = "last.pth" if dataset in {"ohaze", "gopro"} else "anonymous.pt" if dataset == "imagenet100" else "best.pt"
        setting = f"seed{seed}/{role}"
    path = ROOT / "weights" / dataset / mixer / setting
    return path if path.is_file() else None


def generated_path(dataset: str, mixer: str, seed: int | None, suffix: str) -> Path:
    folder = ROOT / "generated_configs"
    folder.mkdir(exist_ok=True)
    return folder / f"{dataset}_{mixer}_seed{seed if seed is not None else 'paper'}.{suffix}"


def task_dataset(args: argparse.Namespace) -> str:
    if args.task != "restoration" and args.mixer == "cspn_style":
        raise ValueError("The paper uses CSPN-style only for restoration propagation controls.")
    defaults = {"segmentation": "cityscapes", "classification": "imagenet100"}
    dataset = args.dataset or defaults.get(args.task)
    allowed = {
        "restoration": {"ohaze", "gopro"},
        "segmentation": {"cityscapes"},
        "classification": {"imagenet100"},
    }
    if dataset not in allowed[args.task]:
        raise ValueError(f"task={args.task} is incompatible with dataset={dataset}")
    return dataset


def environment(run_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    paths = [str(ROOT), str(RUNTIME)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if run_root is not None:
        env["NSI_RUN_ROOT"] = str(run_root.resolve())
    return env


def prepare_restoration(
    args: argparse.Namespace, dataset: str, run_dir: Path
) -> tuple[Path, Config]:
    names = OHAZE_CONFIGS if dataset == "ohaze" else GOPRO_CONFIGS
    if args.mixer not in names:
        raise ValueError(f"the {dataset} paper experiment does not contain mixer={args.mixer}")
    source = ROOT / "configs" / "restoration" / dataset / names[args.mixer]
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["manual_seed"])
    config["manual_seed"] = seed
    config["name"] = f"{dataset}-{args.mixer}-seed{seed}"
    data_root = require_path(args.data_root, "--data-root")
    train = config["datasets"]["train"]
    if dataset == "ohaze":
        train["dataroot_gt"] = str((data_root / "train" / "gt").resolve())
        train["dataroot_lq"] = str((data_root / "train" / "hazy").resolve())
    else:
        train["dataroot_gt"] = str((data_root / "train" / "sharp").resolve())
        train["dataroot_lq"] = str((data_root / "train" / "blur").resolve())
    if args.mixer in {"ours", "cspn_style"}:
        config["network_g"]["K"] = propagation_steps(args, dataset)
    if args.ablation:
        if dataset != "ohaze" or args.mixer != "ours":
            raise ValueError("--ablation is only valid for O-HAZE with mixer=ours")
        model_type, variant, steps, rho_max = OHAZE_ABLATIONS[args.ablation]
        if args.propagation_steps is not None and args.propagation_steps != steps:
            raise ValueError(f"{args.ablation} requires K={steps}")
        config["network_g"]["type"] = model_type
        config["network_g"]["K"] = steps
        config["network_g"]["rho_max"] = rho_max
        if variant is not None:
            config["network_g"]["ablation"] = variant
        config["name"] = f"ohaze-ours-{args.ablation}-seed{seed}"
    label = f"{args.mixer}_{args.ablation}" if args.ablation else args.mixer
    output = generated_path(dataset, label, seed, "yml")
    output.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output, config


def restoration(args: argparse.Namespace, dataset: str, run_dir: Path) -> None:
    config_path, config = prepare_restoration(args, dataset, run_dir)
    env = environment(run_dir)
    if args.action in {"all", "train"}:
        run([sys.executable, str(RUNTIME / "basicsr" / "train.py"), "-opt", str(config_path)], env, args.dry_run)
    if args.action not in {"all", "eval"}:
        return
    checkpoint = args.checkpoint
    if checkpoint is None:
        models = run_dir / "experiments" / config["name"] / "models"
        endpoint = int(config["train"]["total_iter"])
        exact, latest = models / f"net_g_{endpoint}.pth", models / "net_g_latest.pth"
        checkpoint = exact if dataset == "gopro" else latest
        if args.action == "eval":
            released = (
                ROOT / "weights" / "ohaze" / "ablations" / args.ablation / f"seed{config['manual_seed']}" / "last.pth"
                if args.ablation
                else released_checkpoint(
                    dataset, args.mixer, int(config["manual_seed"]), propagation_steps(args, dataset)
                )
            )
            if released is not None:
                checkpoint = released
    output = run_dir / "results"
    if dataset == "ohaze":
        command = [
            sys.executable, str(ROOT / "tasks" / "restoration" / "evaluate_ohaze.py"),
            "--runtime", str(RUNTIME), "--config", str(config_path),
            "--checkpoint", str(checkpoint), "--data-root", str(require_path(args.data_root, "--data-root")),
            "--output-dir", str(output),
            "--device", args.device,
        ]
    else:
        command = [
            sys.executable, str(ROOT / "tasks" / "restoration" / "evaluate_gopro.py"),
            "--workspace", str(RUNTIME), "--config", str(config_path),
            "--checkpoint", str(checkpoint),
            "--data-root", str(args.eval_data_root or args.data_root),
            "--output-dir", str(output),
            "--device", args.device,
        ]
    run(command, env, args.dry_run)


def segmentation(args: argparse.Namespace, run_dir: Path) -> None:
    seed = args.seed if args.seed is not None else 304
    steps = propagation_steps(args, "cityscapes")
    repo = ROOT / "tasks" / "segmentation" / "carrier" / "BPNet"
    data_root = require_path(args.data_root, "--data-root")
    if args.action in {"all", "train"}:
        command = [
            sys.executable, str(ROOT / "tasks" / "segmentation" / "train.py"),
            "--arm", args.mixer, "--bpnet-repo", str(repo),
            "--data", str(data_root), "--run-dir", str(run_dir),
            "--device", args.device, "--seed", str(seed),
            "--propagation-steps", str(steps),
        ]
        run(command, environment(), args.dry_run)
    if args.action in {"all", "eval"}:
        released = released_checkpoint("cityscapes", args.mixer, seed, steps) if args.action == "eval" else None
        checkpoint = args.checkpoint or released or run_dir / "best.pt"
        run([
            sys.executable, str(ROOT / "tasks" / "segmentation" / "evaluate.py"),
            "--arm", args.mixer, "--bpnet-repo", str(repo), "--data", str(data_root),
            "--checkpoint", str(checkpoint), "--output", str(run_dir / "EVAL.json"),
            "--device", args.device, "--propagation-steps", str(steps),
        ], environment(), args.dry_run)


def classification(args: argparse.Namespace, run_dir: Path) -> None:
    data_root = require_path(args.data_root, "--data-root")
    source = ROOT / "configs" / "classification" / "imagenet100.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    config["dataset"]["root"] = str(data_root.resolve())
    config["dataset"]["classes_csv"] = str((data_root / "evidence" / "classes.csv").resolve())
    config["arms"]["radial_profiled"]["K"] = propagation_steps(args, "imagenet100")
    if args.seed is not None:
        config["randomness"] = {"initialization_seed": args.seed, "data_seed": args.seed}
    output = generated_path("imagenet100", args.mixer, args.seed, "json")
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    arm = CLASSIFICATION_ARMS[args.mixer]
    if args.action in {"all", "train"}:
        run([
            sys.executable, str(ROOT / "tasks" / "classification" / "train.py"),
            "--config", str(output), "--arm", arm,
            "--mode", "formal", "--run-dir", str(run_dir),
        ], environment(), args.dry_run)
    if args.action in {"all", "eval"}:
        seed = args.seed if args.seed is not None else 20270821
        released = (
            released_checkpoint("imagenet100", args.mixer, seed, propagation_steps(args, "imagenet100"))
            if args.action == "eval" else None
        )
        checkpoint = args.checkpoint or released or run_dir / "best.pt"
        run([
            sys.executable, str(ROOT / "tasks" / "classification" / "evaluate.py"),
            "--config", str(output), "--arm", arm, "--checkpoint", str(checkpoint),
            "--output", str(run_dir / "EVAL.json"), "--device", args.device,
        ], environment(), args.dry_run)


def profile(args: argparse.Namespace, dataset: str) -> None:
    steps = OHAZE_ABLATIONS[args.ablation][2] if args.ablation else propagation_steps(args, dataset)
    label = f"{args.mixer}_{args.ablation}" if args.ablation else args.mixer
    suffix = "" if args.ablation else (f"_k{steps}" if args.mixer in {"ours", "cspn_style"} else "")
    command = [
        sys.executable, str(ROOT / "tools" / "profile_model.py"),
        "--task", args.task, "--mixer", args.mixer, "--propagation-steps", str(steps),
        "--device", args.device if args.mixer in {"ss2d", "mair"} else "cpu",
        "--output", str(ROOT / "profiles" / f"{dataset}_{label}{suffix}.json"),
    ]
    if args.task == "restoration":
        names = OHAZE_CONFIGS if dataset == "ohaze" else GOPRO_CONFIGS
        if args.mixer not in names:
            raise ValueError(f"the {dataset} paper experiment does not contain mixer={args.mixer}")
        config_path = ROOT / "configs" / "restoration" / dataset / names[args.mixer]
        if args.ablation:
            if dataset != "ohaze" or args.mixer != "ours":
                raise ValueError("--ablation is only valid for O-HAZE with mixer=ours")
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            model_type, variant, steps, rho_max = OHAZE_ABLATIONS[args.ablation]
            config["network_g"].update(type=model_type, K=steps, rho_max=rho_max)
            if variant is not None:
                config["network_g"]["ablation"] = variant
            config_path = generated_path(dataset, f"{label}_profile", config["manual_seed"], "yml")
            config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        command += ["--config", str(config_path), "--height", "256", "--width", "256"]
    elif args.task == "segmentation":
        command += ["--height", "896", "--width", "896"]
    else:
        command += ["--config", str(ROOT / "configs" / "classification" / "imagenet100.json"), "--height", "224", "--width", "224"]
    run(command, environment(), args.dry_run)


def prepare_dataset(args: argparse.Namespace, dataset: str) -> None:
    source = require_path(args.source_root, "--source-root")
    command = [sys.executable, str(ROOT / "prepare_data.py"), "--dataset", dataset, "--source", str(source)]
    if dataset != "cityscapes":
        command += ["--output", str(require_path(args.data_root, "--data-root"))]
    if args.copy_data:
        command.append("--copy")
    run(command, environment(), args.dry_run)


def main() -> None:
    args = parse_args()
    dataset = task_dataset(args)
    if dataset == "cityscapes" and args.action != "prepare":
        if args.mixer != "ours" or propagation_steps(args, dataset) != 5:
            raise ValueError("The current paper reports only the repaired Cityscapes model with ours and K=5.")
    if args.mixer == "mair" and dataset != "ohaze":
        raise ValueError("The paper uses MaIR-VMM only with O-HAZE.")
    if args.ablation and (dataset != "ohaze" or args.mixer != "ours"):
        raise ValueError("Use --ablation only with O-HAZE and --mixer ours.")
    if (args.action == "eval" and args.propagation_steps not in (None, 5)
            and not args.ablation and not args.checkpoint):
        raise ValueError("Select a depth ablation or provide its matching checkpoint.")
    if args.action in {"train", "all"} and args.device != "cuda:0":
        raise ValueError("For training, select one GPU with CUDA_VISIBLE_DEVICES and use --device cuda:0.")
    if args.action == "all" and args.checkpoint:
        raise ValueError("Use --checkpoint with --action eval. Action all uses the new training result.")
    default_seeds = {"ohaze": 42, "gopro": 42, "cityscapes": 304}
    seed_label = args.seed if args.seed is not None else default_seeds.get(dataset, "paper")
    experiment = args.ablation or args.mixer
    run_dir = (args.run_dir or ROOT / "runs" / dataset / experiment / f"seed{seed_label}").resolve()
    if args.action == "prepare":
        prepare_dataset(args, dataset)
        return
    if args.action == "profile":
        profile(args, dataset)
        return
    if not args.dry_run and args.action in {"train", "all"} and run_dir.exists():
        raise FileExistsError(f"output directory already exists; choose a new --run-dir: {run_dir}")
    if args.task == "restoration":
        restoration(args, dataset, run_dir)
    elif args.task == "segmentation":
        segmentation(args, run_dir)
    else:
        classification(args, run_dir)


if __name__ == "__main__":
    main()

