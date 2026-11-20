"""Audit every checkpoint listed in ``experiments.json``.

The default pass checks files, sizes, and expected SHA256 digests.
The ``--load`` pass loads every checkpoint in an isolated CPU process.
It also runs small CPU forwards when the model supports them.
SS2D and MaIR require compiled extensions, so the pass checks loading only.
Each row uses a fresh process to prevent conflicting task imports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping


SCHEMA = "\x65qop2027.release_check.v1"
LOAD_SENTINEL = "__\x45QOP2027_LOAD_RESULT__"


def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_experiments(root: Path) -> list[dict[str, Any]]:
    path = root / "experiments.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("experiments")
    if not isinstance(rows, list):
        raise ValueError("experiments.json must contain an experiments list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("every experiment entry must be a JSON object")
    return rows


def registry_issues(rows: list[dict[str, Any]]) -> list[str]:
    """Return structural registry errors without a fixed experiment count."""

    issues = []
    ids = [row.get("id") for row in rows]
    if any(not isinstance(identity, str) or not identity for identity in ids):
        issues.append("every experiment must have a nonempty string id")
    duplicates = sorted({identity for identity in ids if ids.count(identity) > 1})
    if duplicates:
        issues.append(f"duplicate experiment ids: {', '.join(duplicates)}")
    required = ("dataset", "task", "checkpoint", "suites")
    for index, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            issues.append(f"row {index} is missing: {', '.join(missing)}")
    return issues


def read_expected_checksums(root: Path) -> dict[str, dict[str, Any]]:
    """Read and validate the release checksum manifest."""

    path = root / "weights" / "checksums.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checksums.json must contain an object")
    if str(payload.get("algorithm", "")).lower() != "sha256":
        raise ValueError("checksums.json must use the sha256 algorithm")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("checksums.json must contain a files object")
    checksums: dict[str, dict[str, Any]] = {}
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise ValueError("checksums.json entries must map paths to objects")
        digest = metadata.get("sha256")
        size = metadata.get("bytes")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid sha256 digest for {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid byte count for {name}")
        checksums[name.replace("\\", "/")] = {
            "bytes": size,
            "sha256": digest.lower(),
        }
    return checksums


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def inventory_row(
    root: Path,
    row: Mapping[str, Any],
    *,
    with_sha256: bool,
    expected_checksums: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    checkpoint_text = str(row.get("checkpoint", ""))
    checkpoint = (root / checkpoint_text).resolve()
    result: dict[str, Any] = {
        "id": row.get("id"),
        "label": row.get("label"),
        "dataset": row.get("dataset"),
        "task": row.get("task"),
        "mixer": row.get("mixer"),
        "seed": row.get("seed"),
        "ablation": row.get("ablation"),
        "checkpoint": relative_path(root, checkpoint),
        "exists": checkpoint.is_file(),
    }
    key = relative_path(root, checkpoint)
    expected = expected_checksums.get(key) if expected_checksums is not None else None
    if expected is None:
        row_digest = row.get("sha256") or row.get("checksum")
        expected = {"sha256": str(row_digest)} if row_digest else None
    if expected is None:
        result["expected_sha256_missing"] = True
    else:
        result["expected_sha256"] = str(expected["sha256"]).lower()
        if "bytes" in expected:
            result["expected_bytes"] = int(expected["bytes"])
    if checkpoint.is_file():
        result["bytes"] = checkpoint.stat().st_size
        if expected is not None and "bytes" in expected:
            result["bytes_matches"] = result["bytes"] == result["expected_bytes"]
        if with_sha256:
            result["sha256"] = sha256_file(checkpoint)
        if expected is not None:
            actual = result.get("sha256")
            if actual is not None:
                result["sha256_matches"] = actual.lower() == result["expected_sha256"]
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return value


def _reproduction_module(root: Path):
    """Load launcher constants without running its command-line entry point."""

    sys.path.insert(0, str(root))
    import reproduce

    return reproduce


def _restoration_config(root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    reproduce = _reproduction_module(root)
    dataset = str(row["dataset"])
    mixer = str(row["mixer"])
    if dataset == "ohaze":
        name = reproduce.OHAZE_CONFIGS[mixer]
        config_path = root / "configs" / "restoration" / "ohaze" / name
    elif dataset == "gopro":
        name = reproduce.GOPRO_CONFIGS[mixer]
        config_path = root / "configs" / "restoration" / "gopro" / name
    else:
        raise ValueError(f"unsupported restoration dataset: {dataset}")
    config = _load_yaml(config_path)

    ablation = row.get("ablation")
    if ablation:
        if dataset != "ohaze" or mixer != "ours":
            raise ValueError("ablations are only supported for O-HAZE mixer=ours")
        model_type, variant, steps, rho_max = reproduce.OHAZE_ABLATIONS[str(ablation)]
        network = config["network_g"]
        network["type"] = model_type
        network["K"] = steps
        network["rho_max"] = rho_max
        if variant is not None:
            network["ablation"] = variant
    return config


def _state_dict(torch: Any, checkpoint: Path, dataset: str) -> Mapping[str, Any]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if dataset in {"ohaze", "gopro", "haze4k"}:
        state = saved.get("params_ema", saved.get("params", saved))
    else:
        state = saved.get("model", saved)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint state is not a mapping: {checkpoint}")
    return state


def _build_restoration(root: Path, row: Mapping[str, Any], torch: Any):
    runtime = root / "tasks" / "restoration" / "runtime"
    sys.path.insert(0, str(runtime))
    from basicsr.models.archs import define_network

    config = _restoration_config(root, row)
    model = define_network(dict(config["network_g"]))
    return model, None


def _build_haze4k(root: Path, row: Mapping[str, Any]):
    haze_dir = root / "tasks" / "restoration" / "haze4k"
    sys.path.insert(0, str(haze_dir))
    from models import make_model

    return make_model(str(row.get("arm", row["mixer"]))), None


def _build_segmentation(root: Path, row: Mapping[str, Any], torch: Any):
    # Load this module in a fresh process because BPNet imports generic module names.
    from tasks.segmentation import train as segmentation_train

    repo = root / "tasks" / "segmentation" / "carrier" / "BPNet"
    cfg, model, _backend, _contract = segmentation_train.build_model(
        repo, 896, str(row["mixer"]), 5
    )
    return model, (segmentation_train, cfg)


def _classification_arm(mixer: str) -> str:
    return {
        "native": "poolformerv2",
        "cspn_style": "cspn_style_k5",
        "dwconv": "dwconv",
        "mdta": "mdta",
        "swin": "swin",
        "ss2d": "ss2d",
        "ours": "radial_profiled",
    }[mixer]


def _build_classification(root: Path, row: Mapping[str, Any], torch: Any):
    config = json.loads(
        (root / "configs" / "classification" / "imagenet100.json").read_text(
            encoding="utf-8"
        )
    )
    classification_dir = root / "tasks" / "classification"
    sys.path.insert(0, str(classification_dir))
    from training_utils import make_model

    arm = _classification_arm(str(row["mixer"]))
    if arm not in config["arms"]:
        raise KeyError(f"classification arm is absent from config: {arm}")
    model = make_model(config["arms"][arm], ss2d_factory_spec=None)
    return model, None


def _extension_status(mixer: str) -> dict[str, Any]:
    if mixer == "mair":
        names = ["selective_scan_cuda_oflex"]
    elif mixer == "ss2d":
        names = ["selective_scan_cuda", "selective_scan_cuda_oflex"]
    else:
        return {"checked": False, "missing": []}
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    return {"checked": True, "required": names, "missing": missing}


def _small_forward(
    model: Any,
    row: Mapping[str, Any],
    task_context: Any,
    torch: Any,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    mixer = str(row["mixer"])
    extension = _extension_status(mixer)
    if mixer in {"ss2d", "mair"}:
        missing = extension.get("missing", [])
        detail = (
            f"strict-load-only by policy; forward requires compiled extension(s): "
            f"{', '.join(missing) if missing else 'available extensions'}"
        )
        return {
            "status": "skipped",
            "reason": detail,
            "extensions": extension,
        }

    model.eval()
    # BPNet pools by eight after its backbone, so Cityscapes needs 128 pixels.
    size = 128 if dataset == "cityscapes" else 64
    sample = torch.zeros((1, 3, size, size), dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        if dataset == "cityscapes":
            segmentation_train, cfg = task_context
            output = segmentation_train.logits(model, cfg, sample)
        else:
            output = model(sample)
    if isinstance(output, (list, tuple)):
        output = output[-1]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"forward output is not a tensor: {type(output).__name__}")
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise FloatingPointError("small CPU forward produced a non-finite tensor")
    return {
        "status": "passed",
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "finite": finite,
    }


def load_one(root: Path, index: int) -> dict[str, Any]:
    """Build and strictly load one row in an isolated process."""

    import torch

    # The child starts in ``tools/``; add the release root before task imports.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    torch.set_num_threads(1)
    rows = read_experiments(root)
    if index < 0 or index >= len(rows):
        raise IndexError(f"experiment index out of range: {index}")
    row = rows[index]
    checkpoint = (root / str(row["checkpoint"])).resolve()
    result: dict[str, Any] = {
        "id": row.get("id"),
        "index": index,
        "strict_load": {"status": "failed"},
        "forward": {"status": "not_run"},
    }
    if not checkpoint.is_file():
        result["strict_load"] = {
            "status": "failed",
            "error": f"checkpoint does not exist: {relative_path(root, checkpoint)}",
        }
        return result

    dataset = str(row["dataset"])
    task_context = None
    if dataset == "haze4k":
        try:
            model, task_context = _build_haze4k(root, row)
        except Exception as exc:
            result['strict_load'] = {'status': 'failed', 'error': str(exc),
                                     'error_type': type(exc).__name__}
            return result
    elif row.get("entrypoint"):
        import tempfile
        from supplement.runtime import network
        from supplement.run import check
        with tempfile.TemporaryDirectory(prefix="supplement-check-") as directory:
            try:
                model, _ = network(row, Path(directory))
                value = check(row, model, checkpoint)
                value['index'] = index
                return value
            except Exception as exc:
                result['strict_load'] = {'status': 'failed', 'error': str(exc),
                                         'error_type': type(exc).__name__}
                return result
    try:
        if dataset == "haze4k":
            pass
        elif dataset in {"ohaze", "gopro"}:
            model, task_context = _build_restoration(root, row, torch)
        elif dataset == "cityscapes":
            model, task_context = _build_segmentation(root, row, torch)
        elif dataset == "imagenet100":
            model, task_context = _build_classification(root, row, torch)
        else:
            raise ValueError(f"unsupported dataset: {dataset}")

        state = _state_dict(torch, checkpoint, dataset)
        model.load_state_dict(state, strict=True)
        model = model.to(device="cpu", dtype=torch.float32)
        model.eval()
        result["strict_load"] = {
            "status": "passed",
            "state_keys": len(state),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    except Exception as exc:  # Return a structured failure to the parent.
        result["strict_load"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }
        return result

    try:
        result["forward"] = _small_forward(model, row, task_context, torch)
    except Exception as exc:
        result["forward"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }
    return result


def run_load_one(root: Path, index: int) -> int:
    try:
        result = load_one(root, index)
    except Exception as exc:
        result = {
            "index": index,
            "strict_load": {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            },
            "forward": {"status": "not_run"},
        }
    print(LOAD_SENTINEL + json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def invoke_loader(root: Path, index: int) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_load-one",
        str(index),
        "--_release-root",
        str(root),
    ]
    completed = subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    payload_line = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith(LOAD_SENTINEL)),
        None,
    )
    if payload_line is None:
        return {
            "index": index,
            "strict_load": {
                "status": "failed",
                "error_type": "LoaderProcessError",
                "error": "loader subprocess returned no structured result",
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout.splitlines()[-8:],
                "stderr_tail": completed.stderr.splitlines()[-8:],
            },
            "forward": {"status": "not_run"},
        }
    result = json.loads(payload_line[len(LOAD_SENTINEL) :])
    noise = [line for line in completed.stdout.splitlines() if not line.startswith(LOAD_SENTINEL)]
    if noise or completed.stderr.strip():
        result["loader_messages"] = {
            "stdout_tail": noise[-8:],
            "stderr_tail": completed.stderr.splitlines()[-8:],
        }
    return result


def build_report(
    root: Path,
    *,
    load: bool,
    with_sha256: bool,
    limit: int | None,
) -> dict[str, Any]:
    rows = read_experiments(root)
    registry_failures = registry_issues(rows)
    checksum_manifest_error = None
    try:
        expected_checksums = read_expected_checksums(root)
    except Exception as exc:
        expected_checksums = {}
        checksum_manifest_error = f"{type(exc).__name__}: {exc}"
    selected = rows if limit is None else rows[:limit]
    records = [
        inventory_row(
            root,
            row,
            with_sha256=with_sha256,
            expected_checksums=expected_checksums,
        )
        for row in selected
    ]
    for index, record in enumerate(records):
        if load:
            record["load"] = invoke_loader(root, index)

    missing = [record for record in records if not record["exists"]]
    checksum_failures = [
        record
        for record in records
        if record.get("sha256_matches") is False
        or record.get("bytes_matches") is False
    ]
    missing_expected = [record for record in records if record.get("expected_sha256_missing")]
    load_failures = [
        record
        for record in records
        if load and record.get("load", {}).get("strict_load", {}).get("status") != "passed"
    ]
    forward_failures = [
        record
        for record in records
        if load and record.get("load", {}).get("forward", {}).get("status") == "failed"
    ]
    status = (
        "PASS"
        if not (
            missing
            or missing_expected
            or checksum_manifest_error
            or checksum_failures
            or load_failures
            or forward_failures
            or registry_failures
        )
        else "FAIL"
    )
    report = {
        "schema": SCHEMA,
        "status": status,
        "release_root": str(root.resolve()),
        "experiments_file": "experiments.json",
        "expected_experiment_count": len(rows),
        "experiment_count_source": "experiments.json",
        "registry_experiment_count": len(rows),
        "experiment_count": len(rows),
        "selected_count": len(selected),
        "load_requested": load,
        "sha256_requested": with_sha256,
        "summary": {
            "missing_checkpoints": len(missing),
            "missing_expected_hashes": len(missing_expected),
            "checksum_failures": len(checksum_failures),
            "strict_load_failures": len(load_failures),
            "forward_failures": len(forward_failures),
            "registry_failures": len(registry_failures),
        },
        "experiments": records,
    }
    if checksum_manifest_error:
        report["checksum_manifest_error"] = checksum_manifest_error
    if registry_failures:
        report["registry_failures"] = registry_failures
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load",
        action="store_true",
        help="build each listed model and strictly load its checkpoint on CPU",
    )
    parser.add_argument(
        "--no-sha256",
        action="store_true",
        help="skip file hashing in the standard-library inventory pass",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    parser.add_argument("--limit", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_load-one", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_release-root", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Emit UTF-8 JSON because Windows shells may use legacy code pages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = (args._release_root or release_root()).resolve()
    if args._load_one is not None:
        return run_load_one(root, args._load_one)
    experiment_count = len(read_experiments(root))
    if args.limit is not None and (args.limit < 0 or args.limit > experiment_count):
        raise SystemExit(f"--limit must be between 0 and {experiment_count}")
    report = build_report(
        root,
        load=args.load,
        with_sha256=not args.no_sha256,
        limit=args.limit,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
