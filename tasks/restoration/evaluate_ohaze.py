"""O-HAZE endpoint evaluation: long-side 512, FP32, full-image RGB metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pytorch_msssim import ssim
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from tasks.restoration.metric_contract import direct_ssim_protocol


def image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB")).float().div_(255).unsqueeze(0)


def resize_long_side(image: torch.Tensor, side: int = 512) -> torch.Tensor:
    height, width = image.shape[-2:]
    scale = side / max(height, width)
    size = (round(height * scale), round(width * scale))
    return F.interpolate(image, size=size, mode="bicubic", align_corners=False, antialias=True)


def pad32(image: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    height, width = image.shape[-2:]
    ph, pw = (-height) % 32, (-width) % 32
    return F.pad(image, (0, pw, 0, ph), mode="reflect"), height, width


def metric(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    prediction = prediction.clamp(0, 1).float()
    target = target.float()
    psnr = 10 * torch.log10(1 / F.mse_loss(prediction, target))
    score = ssim(
        prediction,
        target,
        data_range=1,
        size_average=False,
        win_size=11,
        win_sigma=1.5,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    )
    return float(psnr), float(score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("formal O-HAZE evaluation requires CUDA")
    sys.path.insert(0, str(args.runtime.resolve()))
    from basicsr.models.archs import define_network

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = define_network(dict(config["network_g"]))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("params_ema", saved.get("params", saved)), strict=True)
    model = model.to(args.device).float().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from clpo2027.fp32_inference import prepare_fp32_inference
    inference = prepare_fp32_inference(model)

    hazy_dir, gt_dir = args.data_root / "test" / "hazy", args.data_root / "test" / "gt"
    names = sorted(path.name for path in hazy_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(names) != 5 or {int(Path(n).stem.split('_')[0]) for n in names} != set(range(41, 46)):
        raise RuntimeError("Use the five O-HAZE test images with IDs 41 to 45.")
    if any(not (gt_dir / name).is_file() for name in names):
        raise RuntimeError("the O-HAZE test set is empty or contains unmatched pairs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode():
        for name in names:
            source = resize_long_side(image_tensor(hazy_dir / name)).to(args.device)
            target = resize_long_side(image_tensor(gt_dir / name)).to(args.device)
            padded, height, width = pad32(source)
            value = model(padded)
            prediction = (value[-1] if isinstance(value, list) else value)[:, :, :height, :width]
            psnr, ssim_value = metric(prediction, target)
            to_pil_image((prediction[0].cpu() + 0.5 / 255).clamp(0, 1)).save(args.output_dir / name)
            records.append({"image": name, "psnr": psnr, "ssim": ssim_value})
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "protocol": "O-HAZE IDs 41-45; FP32; long side 512; RGB PSNR; direct RGB SSIM; no image pre-pooling; Gaussian window 11, sigma 1.5, K (0.01, 0.03)",
        "metric_protocol": direct_ssim_protocol(
            backend="pytorch_msssim.ssim",
            data_range=1,
            numeric_precision="FP32",
        ),
        "inference": inference,
        "count": len(records),
        "psnr": sum(row["psnr"] for row in records) / len(records),
        "ssim": sum(row["ssim"] for row in records) / len(records),
        "images": records,
    }
    (args.output_dir / "FINAL.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
