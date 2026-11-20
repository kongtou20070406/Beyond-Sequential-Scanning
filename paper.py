"""Run the experiments in the paper experiment list."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from tasks.restoration.metric_contract import (
    DIRECT_RGB_SSIM_ID,
    historical_pooled_only,
    recorded_result,
    require_direct_ssim,
    result_metric,
)

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "verify", "eval", "train", "collect"))
    parser.add_argument("--suite", default="all")
    parser.add_argument("--id", help="Select one experiment ID.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "paper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--trained", action="store_true", help="Evaluate the new weights in --output.")
    parser.add_argument("--recorded", action="store_true", help="Collect bundled supplementary results without inference.")
    args = parser.parse_args()
    if args.trained and args.action != "eval":
        parser.error("Use --trained only with eval.")
    if args.recorded and args.action != "collect":
        parser.error("Use --recorded only with collect.")
    rows = read_json(ROOT / "experiments.json")["experiments"]
    rows = [r for r in rows if (args.suite == "all" or args.suite in r["suites"])
            and (args.id is None or r["id"] == args.id)]
    if not rows:
        parser.error("No experiment matches the selection.")
    if args.action == "list":
        for row in rows:
            status = " [historical pooled SSIM only]" if historical_pooled_only(row) else ""
            print(f'{row["id"]:48} {row["kind"]:20} {row["label"]}{status}')
        return
    if args.action == "verify":
        files = read_json(ROOT / "weights" / "checksums.json")["files"]
        failed = []
        for name in sorted({r["checkpoint"] for r in rows}):
            path = ROOT / name
            item = files.get(name)
            valid = (item is not None and path.is_file()
                     and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"])
            print(f'{"PASS" if valid else "FAIL"} {name}')
            if not valid:
                failed.append(name)
        if failed:
            raise SystemExit(1)
        return
    if args.action == "collect":
        records = []
        missing = []
        excluded = []
        for row in rows:
            if args.recorded and historical_pooled_only(row):
                excluded.append({
                    "id": row["id"],
                    "reason": "historical_pooled_ssim_only; no direct SSIM was measured",
                })
                continue
            folder = args.output / row["id"]
            candidates = [folder / "EVAL.json", folder / "results" / "FINAL.json"]
            if args.recorded:
                source = recorded_result(row)
                candidates = [ROOT / source] if source else []
            result = next((p for p in candidates if p.is_file()), None)
            if result is None:
                missing.append(row["id"])
                continue
            value = read_json(result)
            record = {"id": row["id"], "dataset": row["dataset"],
                      "label": row["label"], "seed": row.get("seed"),
                      "source": str(result)}
            expected_metrics = row.get("expected", {})
            metric_names = expected_metrics or ("psnr", "ssim")
            for metric in metric_names:
                if metric == "ssim":
                    require_direct_ssim(row, value, result)
                    record["metric_protocol"] = DIRECT_RGB_SSIM_ID
                try:
                    actual = result_metric(value, metric)
                except ValueError as error:
                    raise ValueError(f"{error}: {result}") from error
                record[metric] = actual
                if metric in expected_metrics:
                    expected = expected_metrics[metric]
                    record[metric + "_paper"] = expected
                    record[metric + "_difference"] = float(actual) - expected
            records.append(record)
        args.output.mkdir(parents=True, exist_ok=True)
        summary = {"results": records, "missing": missing, "excluded": excluded,
                   "gopro_sample_statistics": {}}
        for label in {r["label"] for r in rows if r["dataset"] == "gopro"}:
            selected = [r for r in records
                        if r["dataset"] == "gopro" and r["label"] == label]
            if len(selected) == 3:
                summary["gopro_sample_statistics"][label] = {
                    metric: {"mean": statistics.mean(r[metric] for r in selected),
                             "sample_std": statistics.stdev(r[metric] for r in selected)}
                    for metric in ("psnr", "ssim")}
        (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if records:
            keys = list(dict.fromkeys(k for r in records for k in r))
            with (args.output / "SUMMARY.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
        print(json.dumps(summary, indent=2))
        if missing:
            raise SystemExit(1)
        return
    for row in rows:
        if args.action == "train" and not row.get("trainable", True):
            print(f'SKIP fixed-weight row: {row["id"]}', flush=True)
            continue
        if row.get("entrypoint"):
            command = [sys.executable, str(ROOT / row["entrypoint"]), "--id", row["id"],
                       "--action", args.action, "--data-root", str(args.data_root / row["dataset"]),
                       "--output", str(args.output / row["id"])]
            if args.action == "eval":
                checkpoint = ROOT / row["checkpoint"]
                if args.trained:
                    if not row.get("trainable", True):
                        raise ValueError("A fixed-weight row has no --trained checkpoint.")
                    checkpoint = args.output / row["id"] / "models" / f'net_g_{row["updates"]}.pth'
                command += ["--checkpoint", str(checkpoint)]
            if args.dry_run:
                command.append("--dry-run")
            print(json.dumps(command), flush=True)
            subprocess.run(command, check=True, cwd=ROOT)
            continue
        command = [sys.executable, str(ROOT / "reproduce.py"), "--task", row["task"],
                   "--dataset", row["dataset"], "--mixer", row["mixer"],
                   "--action", args.action, "--data-root", str(args.data_root / row["dataset"]),
                   "--run-dir", str(args.output / row["id"])]
        if row.get("seed_override", True) and row.get("seed") is not None:
            command += ["--seed", str(row["seed"])]
        if row.get("ablation"):
            command += ["--ablation", row["ablation"]]
        if args.action == "eval":
            checkpoint = ROOT / row["checkpoint"]
            if args.trained:
                folder = args.output / row["id"]
                if row["task"] == "restoration":
                    name = f'{row["dataset"]}-{row["mixer"]}'
                    if row.get("ablation"):
                        name += f'-{row["ablation"]}'
                    name += f'-seed{row["seed"]}'
                    endpoint = "net_g_49350.pth" if row["dataset"] == "gopro" else "net_g_latest.pth"
                    checkpoint = folder / "experiments" / name / "models" / endpoint
                else:
                    checkpoint = folder / "best.pt"
            command += ["--checkpoint", str(checkpoint.resolve())]
        if args.dry_run:
            command.append("--dry-run")
        print(json.dumps(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
