"""Shared identity and collection rules for reported RGB SSIM."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


DIRECT_RGB_SSIM_ID = "direct_rgb_ssim_v1"
DIRECT_RGB_SSIM = {
    "id": DIRECT_RGB_SSIM_ID,
    "color_space": "RGB",
    "image_pre_pooling": False,
    "window_size": 11,
    "window_sigma": 1.5,
    "k1": 0.01,
    "k2": 0.03,
}


def direct_ssim_protocol(
    *, backend: str, data_range: float, numeric_precision: str
) -> dict[str, object]:
    """Return protocol metadata without changing a metric implementation."""

    return {
        **DIRECT_RGB_SSIM,
        "backend": backend,
        "data_range": data_range,
        "numeric_precision": numeric_precision,
    }


def recorded_result(row: Mapping[str, Any]) -> str | None:
    """Return the authoritative result path, including direct reassessments."""

    if row.get("recorded_result"):
        return str(row["recorded_result"])
    reassessment = row.get("ssim_reassessment")
    if isinstance(reassessment, Mapping) and reassessment.get("result"):
        return str(reassessment["result"])
    return None


def historical_pooled_only(row: Mapping[str, Any]) -> bool:
    reassessment = row.get("ssim_reassessment")
    return bool(
        row.get("metric_status") == "historical_pooled_only"
        or (
            isinstance(reassessment, Mapping)
            and reassessment.get("status") == "NOT_REASSESSED"
        )
    )


def require_direct_ssim(
    row: Mapping[str, Any], result: Mapping[str, Any], source: Path | str
) -> None:
    """Reject a result unless its metadata identifies direct RGB SSIM."""

    protocol = result.get("metric_protocol")
    if isinstance(protocol, Mapping):
        if any(protocol.get(key) != value for key, value in DIRECT_RGB_SSIM.items()):
            raise ValueError(f"Non-direct SSIM metric protocol: {source}")
        return

    text = " ".join(
        str(result.get(key, ""))
        for key in ("schema", "protocol", "ssim_protocol")
    ).lower()
    if "pooled ssim" in text or "release pooled" in text:
        raise ValueError(f"Pooled SSIM result is historical evidence only: {source}")
    if result.get("schema") == "ohaze-direct-ssim-reassessment-v1":
        return
    if row.get("dataset") == "gopro" and result.get("per_image_sha256"):
        # Bundled GoPro records predate metric_protocol. Their per-image hash,
        # checkpoint hash, and native protocol identify the direct metric source.
        if result.get("protocol") in (None, "native", "native_full0"):
            return
    raise ValueError(f"Direct SSIM metric protocol is missing: {source}")


def result_metric(result: Mapping[str, Any], name: str) -> float:
    """Read one reported metric without ambiguous SSIM aliases."""

    if name == "ssim":
        if "ssim" not in result:
            raise ValueError("The canonical direct SSIM field 'ssim' is missing.")
        return float(result["ssim"])
    if name == "psnr" and result.get("schema") == "ohaze-direct-ssim-reassessment-v1":
        published = result.get("legacy_published")
        if isinstance(published, Mapping) and "psnr" in published:
            return float(published["psnr"])
    aliases = {
        "psnr": ("psnr", "psnr_db", "average_psnr", "mean_psnr"),
        "miou": ("miou", "mIoU"),
        "top1": ("top1", "val_top1"),
        "top5": ("top5", "val_top5"),
    }
    for key in aliases.get(name, (name,)):
        if key in result:
            return float(result[key])
    raise ValueError(f"Missing metric {name}")
