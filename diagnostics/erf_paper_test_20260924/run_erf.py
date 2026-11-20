"""Measure one paper-checkpoint full-model effective receptive field."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import types
import tempfile
from pathlib import Path

import numpy as np

# Triton's Windows getenv expects UTF-8. Keep compiler caches on an ASCII path.
COMPILER_CACHE = Path(os.environ.get("NSI_DIAGNOSTIC_CACHE", str(Path(tempfile.gettempdir()) / "nsi_diagnostics")))
COMPILER_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str(COMPILER_CACHE / "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(COMPILER_CACHE / "inductor"))
os.environ.setdefault("NSI_HAZE_OUTPUT", str(COMPILER_CACHE / "haze_runtime"))

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
import yaml


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
DATA = Path(os.environ.get("NSI_DATA_ROOT", str(CODE / "data")))
RUNTIME = CODE / "tasks" / "restoration" / "runtime"
HAZE = CODE / "tasks" / "restoration" / "haze4k"
for path in (CODE, RUNTIME, HAZE):
    sys.path.insert(0, str(path))

from reproduce import GOPRO_CONFIGS, OHAZE_CONFIGS
from tasks.restoration.evaluate_ohaze import image_tensor, pad32, resize_long_side


MAIN = {
    "gopro": ("native", "cspn_style", "ss2d", "ours"),
    "ohaze": ("native", "dwconv", "swin", "cspn_style", "ss2d", "ours"),
    "haze4k": ("native", "dwconv", "swin", "cspn_style", "ss2d", "full"),
}
APPENDIX = ("mair", "spatial_mamba", "mair_native_blocks", "spatial_native_blocks")
APPENDIX_IDS = {"spatial_mamba": "ohaze-spatial-mamba-seed42",
                "mair_native_blocks": "ohaze-mair-full-block-seed42",
                "spatial_native_blocks": "ohaze-spatial-full-block-seed42"}
LABELS = {"native": "VIVNet", "cspn_style": "CSPN-style", "ss2d": "SS2D",
          "ours": "NSI", "full": "NSI", "dwconv": "DWConv", "swin": "Swin",
          "mair": "MaIR core", "spatial_mamba": "Spatial-Mamba core",
          "mair_native_blocks": "MaIR full block",
          "spatial_native_blocks": "Spatial-Mamba full block"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_row(dataset: str, arm: str) -> dict:
    rows = json.loads((CODE / "experiments.json").read_text(encoding="utf-8"))["experiments"]
    identifier = APPENDIX_IDS.get(arm, f"{dataset}-{arm}-seed42")
    return next(row for row in rows if row["id"] == identifier)


def load_model(dataset: str, arm: str, row: dict, checkpoint: Path) -> torch.nn.Module:
    if dataset == "haze4k":
        from models import make_model

        model = make_model(arm)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["params"]
        model.load_state_dict(state, strict=True)
        return model
    if row.get("entrypoint") == "supplement/run.py":
        from supplement.runtime import network

        if arm == "spatial_native_blocks":
            if os.environ.get("SUPPLEMENT_SPATIAL_EXTENSIONS"):
                sys.path.insert(0, os.environ["SUPPLEMENT_SPATIAL_EXTENSIONS"])
            import selective_scan_cuda_oflex_rh

            assert hasattr(selective_scan_cuda_oflex_rh, "fwd")
        model, _ = network(row, COMPILER_CACHE / "runtime_copies" / row["runtime"])
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved.get("params_ema", saved.get("params", saved)), strict=True)
        return model
    from basicsr.models.archs import define_network

    name = (GOPRO_CONFIGS if dataset == "gopro" else OHAZE_CONFIGS)[arm]
    config = yaml.safe_load((CODE / "configs" / "restoration" / dataset / name).read_text(encoding="utf-8"))
    model = define_network(dict(config["network_g"]))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("params_ema", saved.get("params", saved)), strict=True)
    return model


def input_image(dataset: str) -> tuple[torch.Tensor, dict]:
    if dataset == "gopro":
        from clpo2027.data import read_tensor

        folder = DATA / "gopro/test/blur"
        names = sorted(folder.glob("*.png"))
        assert len(names) == 1111, "GoPro official flat test set is incomplete"
        path = names[0]
        return read_tensor(path).unsqueeze(0), {"path": str(path), "sha256": sha256(path)}
    if dataset == "ohaze":
        folder = DATA / "ohaze/test/hazy"
        names = sorted(folder.glob("*.jpg"))
        assert len(names) == 5 and [int(p.stem) for p in names] == list(range(41, 46))
        path = names[0]
        return resize_long_side(image_tensor(path)), {"path": str(path), "sha256": sha256(path)}
    manifest = json.loads((HAZE / "DATA_MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["splits"]["test"]) == 1000
    row = manifest["splits"]["test"][0]
    cache = Path(os.environ.get("NSI_HAZE_CACHE", str(DATA / "haze4k/cache"))) / "test.bin"
    spec = row["arrays"]["input"]
    data = np.memmap(cache, mode="r", dtype=np.uint8, offset=spec["offset"], shape=tuple(spec["shape"]))
    pixels = np.asarray(data)
    assert hashlib.sha256(pixels.tobytes()).hexdigest() == row["source"]["input"]["pixel_sha256"]
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float().div_(255).unsqueeze(0)
    return tensor, {"path": str(cache), "image_id": row["id"],
                    "pixel_sha256": row["source"]["input"]["pixel_sha256"]}


def enclosed_radius(gradient: np.ndarray, center: tuple[int, int], fraction: float) -> float:
    yy, xx = np.indices(gradient.shape)
    radius = np.hypot(yy - center[0], xx - center[1]).ravel()
    values = gradient.ravel()
    order = np.argsort(radius)
    index = np.searchsorted(np.cumsum(values[order]), values.sum() * fraction)
    return float(radius[order[min(index, len(order) - 1)]])


def checkpoint_blocks(model: torch.nn.Module) -> None:
    def forward(self, value):
        for block in self:
            value = activation_checkpoint(block, value, use_reentrant=False, preserve_rng_state=False)
        return value

    for name in ("encoder_level1", "encoder_level2", "encoder_level3", "decoder_level3",
                 "decoder_level2", "decoder_level1", "refinement"):
        stage = getattr(model, name)
        assert isinstance(stage, torch.nn.Sequential), name
        stage.forward = types.MethodType(forward, stage)


def differentiable_spatial_fusion(model: torch.nn.Module) -> float:
    """Use the same merged depthwise filter with autograd for input gradients."""
    def forward(self, image):
        return F.conv2d(F.pad(image, (5, 5, 5, 5), mode="replicate"),
                        self._merge_weight, groups=self.dim)

    errors = []
    for module in model.modules():
        if type(module).__name__ != "StateFusion":
            continue
        sample = torch.randn(1, module.dim, 13, 17, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            official = module(sample)
            reference = forward(module, sample)
        error = float((official - reference).abs().max() / reference.abs().max().clamp_min(1))
        errors.append(error)
        module.forward = types.MethodType(forward, module)
    assert len(errors) == 22, f"expected 22 official state-fusion modules, found {len(errors)}"
    assert max(errors) < 1e-4, f"fused depthwise forward differs: {max(errors)}"
    return max(errors)


def run(dataset: str, arm: str, use_checkpoint_blocks: bool) -> None:
    row = select_row(dataset, arm)
    checkpoint = CODE / row["checkpoint"]
    inventory = json.loads((CODE / "weights" / "checksums.json").read_text(encoding="utf-8"))["files"]
    expected = inventory[row["checkpoint"]]
    assert checkpoint.stat().st_size == expected["bytes"]
    checkpoint_sha = sha256(checkpoint)
    assert checkpoint_sha == expected["sha256"], "checkpoint differs from release inventory"
    if dataset == "gopro" and arm in {"ours", "cspn_style"}:
        lock = json.loads((HERE.parent / "GOPRO_SEED42_LOCK.json").read_text(encoding="utf-8"))
        assert checkpoint_sha == lock[arm]["sha256"], "GoPro mechanism reference lock differs"

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    source, source_id = input_image(dataset)
    model = load_model(dataset, arm, row, checkpoint).to(device=device, dtype=torch.float32).eval()
    model.requires_grad_(False)
    source = source.to(device).requires_grad_(True)
    height, width = source.shape[-2:]
    if dataset == "ohaze":
        padded, _, _ = pad32(source)
    elif dataset == "haze4k":
        padded = F.pad(source, (0, (-width) % 4, 0, (-height) % 4), mode="reflect")
    else:
        padded = source
    if arm in {"spatial_mamba", "spatial_native_blocks"}:
        with torch.no_grad():
            official = model(padded.detach())
            official = official[-1] if isinstance(official, list) else official
        fusion_error = differentiable_spatial_fusion(model)
        with torch.no_grad():
            equivalent = model(padded.detach())
            equivalent = equivalent[-1] if isinstance(equivalent, list) else equivalent
        spatial_full_error = float((official - equivalent).abs().max())
        assert spatial_full_error < 1e-5, spatial_full_error
    else:
        fusion_error = spatial_full_error = None
    if use_checkpoint_blocks:
        checkpoint_blocks(model)
    center = (height // 2, width // 2)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    value = model(padded)
    prediction = value[-1] if isinstance(value, list) else value
    assert prediction.shape == padded.shape
    # All paper backbones add the input image at output. Remove its delta-gradient.
    residual_pixel = (prediction - padded)[0, :, center[0], center[1]].mean()
    grad = torch.autograd.grad(residual_pixel, source)[0]
    gradient = grad[0].abs().sum(dim=0).detach().cpu().numpy().astype(np.float32)
    assert gradient.shape == (height, width) and np.isfinite(gradient).all()
    assert float(gradient.sum()) > 0
    elapsed = time.perf_counter() - started
    with torch.inference_mode():
        reference = model(padded.detach())
        reference = reference[-1] if isinstance(reference, list) else reference
        forward_max_abs = float((prediction.detach() - reference).abs().max())
        forward_rel_max = forward_max_abs / max(1.0, float(reference.abs().max()))
    out = Path(os.environ.get("NSI_ERF_OUTPUT", str(CODE / "runs/erf"))) / dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{arm}.npz", gradient=gradient)
    record = {
        "id": row["id"], "label": LABELS[arm], "dataset": dataset, "arm": arm,
        "checkpoint": row["checkpoint"], "checkpoint_sha256": checkpoint_sha,
        "input": source_id, "input_shape": [height, width], "center_yx": list(center),
        "precision": "FP32; TF32 disabled; eval mode; grad-enabled model forward",
        "activation_checkpoint_blocks": use_checkpoint_blocks,
        "spatial_fusion_autograd": "replicate-pad torch.conv2d with official merged kernel" if fusion_error is not None else None,
        "spatial_fusion_forward_max_relative_error": fusion_error,
        "spatial_full_forward_max_abs": spatial_full_error,
        "target": "mean RGB of one central output residual pixel (prediction - input)",
        "gradient_map": "sum of absolute input-channel derivatives",
        "padding": "right/bottom reflect to multiple of 32" if dataset == "ohaze"
                   else "right/bottom reflect to multiple of 4" if dataset == "haze4k" else "none",
        "r50_px": enclosed_radius(gradient, center, 0.5),
        "r90_px": enclosed_radius(gradient, center, 0.9),
        "r95_px": enclosed_radius(gradient, center, 0.95),
        "gradient_sum": float(gradient.sum()),
        "gradient_peak": float(gradient.max()),
        "eval_forward_max_abs": forward_max_abs,
        "eval_forward_relative_max": forward_rel_max,
        "seconds": elapsed,
        "peak_cuda_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    (out / f"{arm}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=MAIN)
    parser.add_argument("arm")
    parser.add_argument("--checkpoint-blocks", action="store_true")
    args = parser.parse_args()
    if args.arm not in MAIN[args.dataset] and not (args.dataset == "ohaze" and args.arm in APPENDIX):
        parser.error(f"{args.arm} is not a paper comparison arm for {args.dataset}")
    run(args.dataset, args.arm, args.checkpoint_blocks)
