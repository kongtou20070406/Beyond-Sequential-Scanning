from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
RUNTIME = Path(__file__).resolve().parent / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from clpo2027.data import read_tensor
from clpo2027.gopro_data import load_gopro_pairs
from clpo2027.gopro_eval import halo_tiled_inference, rgb_psnr, rgb_ssim
from tasks.restoration.metric_contract import direct_ssim_protocol


EXPECTED_ITER = 49_350


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-mode", choices=("native", "tiled"), default="native")
    return parser.parse_args()


class TensorOutput(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        value = self.model(image)
        return value[-1] if isinstance(value, list) else value


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("endpoint evaluation requires CUDA")
    if args.checkpoint.name not in {f"net_g_{EXPECTED_ITER}.pth", "last.pth"}:
        raise RuntimeError(
            f"official test is restricted to net_g_{EXPECTED_ITER}.pth, "
            f"not {args.checkpoint.name}"
        )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["train"]["total_iter"] != EXPECTED_ITER:
        raise RuntimeError("config is not the frozen 49,350-update protocol")

    package_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_root))
    workspace = args.workspace.resolve()
    sys.path.insert(0, str(workspace))
    from basicsr.models.archs import define_network

    model = define_network(dict(config["network_g"]))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = TensorOutput(model).to(device=device, dtype=torch.float32).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from clpo2027.fp32_inference import prepare_fp32_inference
    inference = prepare_fp32_inference(model)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = output_dir / "per_image.jsonl"
    final_path = output_dir / "FINAL.json"
    if per_image_path.exists() or final_path.exists():
        raise FileExistsError("endpoint output already exists")

    from supplement.benchmark.run_fusion import paired_images
    pairs = paired_images(args.data_root, "test")
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with per_image_path.open("x", encoding="utf-8") as stream:
        for index, pair in enumerate(pairs, 1):
            blur = read_tensor(pair.hazy_path)[None].to(device)
            sharp = read_tensor(pair.gt_path)[None].to(device)
            with torch.inference_mode():
                if args.inference_mode == "native":
                    prediction = model(blur).clamp(0.0, 1.0)
                else:
                    prediction = halo_tiled_inference(
                        model, blur, core_tile=384, halo=64, overlap=64,
                    ).clamp(0.0, 1.0)
            psnr = rgb_psnr(prediction, sharp)
            ssim = rgb_ssim(prediction, sharp)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "image_id": pair.image_id,
                        "psnr_db": psnr,
                        "ssim": ssim,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()

    atomic_json(
        final_path,
        {
            "schema": "\x65qop2027.vivnet_gopro_endpoint.v1",
            "status": "COMPLETE",
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "pair_count": len(pairs),
            "psnr_db": float(np.mean(psnr_values)),
            "ssim": float(np.mean(ssim_values)),
            "metric_protocol": direct_ssim_protocol(
                backend="torch.conv2d.valid",
                data_range=1,
                numeric_precision="FP32",
            ),
            "precision": "FP32_TF32_DISABLED",
            "inference": inference,
            "inference_mode": args.inference_mode,
            "tiling": {"core": 384, "halo": 64, "overlap": 64} if args.inference_mode == "tiled" else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "wall_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
