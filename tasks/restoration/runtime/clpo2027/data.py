"""Deterministic O-HAZE pairing, crop manifest, and FP32 image loading."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class ImagePair:
    image_id: str
    hazy_path: Path
    gt_path: Path


@dataclass(frozen=True)
class CropRecord:
    sample_index: int
    pair_index: int
    image_id: str
    crop_y: int
    crop_x: int
    flip_h: bool
    flip_v: bool
    rotation: int


def _key(path: Path) -> str:
    return path.name.split("_", 1)[0]


def _pair_maps(hazy_paths: Sequence[Path], gt_paths: Sequence[Path]) -> list[ImagePair]:
    hazy = {_key(path): path for path in hazy_paths}
    gt = {_key(path): path for path in gt_paths}
    if set(hazy) != set(gt):
        raise RuntimeError(
            f"unpaired images: hazy_only={sorted(set(hazy) - set(gt))}, "
            f"gt_only={sorted(set(gt) - set(hazy))}"
        )
    return [ImagePair(key, hazy[key], gt[key]) for key in sorted(hazy)]


def load_train_pairs(root: str | Path) -> list[ImagePair]:
    root = Path(root)
    pairs = _pair_maps(
        sorted((root / "train" / "hazy").glob("*")),
        sorted((root / "train" / "GT").glob("*")),
    )
    if len(pairs) != 40:
        raise RuntimeError(f"canonical O-HAZE train split must have 40 pairs, found {len(pairs)}")
    return pairs


def load_eval512_pairs(root: str | Path) -> list[ImagePair]:
    directory = Path(root) / "test" / "512"
    pairs = _pair_maps(
        sorted(directory.glob("*outdoor_hazy*")),
        sorted(directory.glob("*outdoor_GT*")),
    )
    if [pair.image_id for pair in pairs] != ["41", "42", "43", "44", "45"]:
        raise RuntimeError(f"canonical O-HAZE 512 outdoor IDs must be 41-45, found {[p.image_id for p in pairs]}")
    return pairs


def select_pairs_by_ids(
    pairs: Sequence[ImagePair],
    image_ids: Sequence[str],
) -> list[ImagePair]:
    """Select whole image pairs in an explicit, validated order."""

    requested = tuple(str(value).zfill(2) for value in image_ids)
    if not requested:
        raise ValueError("at least one image ID must be selected")
    if len(set(requested)) != len(requested):
        raise ValueError("selected image IDs must be unique")
    pair_by_id = {str(pair.image_id).zfill(2): pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("source image pairs contain duplicate IDs")
    missing = [image_id for image_id in requested if image_id not in pair_by_id]
    if missing:
        raise ValueError(f"selected image IDs are missing from pairs: {missing}")
    return [pair_by_id[image_id] for image_id in requested]


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def dataset_identity(pairs: Sequence[ImagePair]) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        for value in (pair.image_id, str(pair.hazy_path.resolve()), str(pair.gt_path.resolve())):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for path in (pair.hazy_path, pair.gt_path):
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


VALID_CROP_SAMPLING_MODES = frozenset({"iid_with_replacement", "balanced_permutation"})


def build_crop_records(
    pairs: Sequence[ImagePair], *, total_samples: int, patch: int, seed: int,
    start_sample: int = 0,
    sampling_mode: str = "iid_with_replacement",
) -> list[CropRecord]:
    if total_samples <= 0 or patch <= 0 or start_sample < 0:
        raise ValueError("total_samples/patch must be positive and start_sample non-negative")
    if sampling_mode not in VALID_CROP_SAMPLING_MODES:
        raise ValueError(f"unknown crop sampling mode: {sampling_mode}")
    sizes: list[tuple[int, int]] = []
    for pair in pairs:
        hazy_size, gt_size = _image_size(pair.hazy_path), _image_size(pair.gt_path)
        if hazy_size != gt_size:
            raise RuntimeError(f"shape mismatch for {pair.image_id}: hazy={hazy_size}, gt={gt_size}")
        if min(hazy_size) < patch:
            raise RuntimeError(f"image {pair.image_id} is smaller than patch={patch}: {hazy_size}")
        sizes.append(hazy_size)

    rng = random.Random(seed)
    records: list[CropRecord] = []
    balanced_block: list[int] = []
    for sample_index in range(start_sample + total_samples):
        if sampling_mode == "balanced_permutation":
            if sample_index % len(pairs) == 0:
                balanced_block = list(range(len(pairs)))
                rng.shuffle(balanced_block)
            pair_index = balanced_block[sample_index % len(pairs)]
        else:
            pair_index = rng.randrange(len(pairs))
        height, width = sizes[pair_index]
        record = CropRecord(
            sample_index=sample_index,
            pair_index=pair_index,
            image_id=pairs[pair_index].image_id,
            crop_y=rng.randint(0, height - patch),
            crop_x=rng.randint(0, width - patch),
            flip_h=rng.random() < 0.5,
            flip_v=rng.random() < 0.5,
            rotation=rng.randrange(4),
        )
        if sample_index >= start_sample:
            records.append(record)
    return records


def write_or_validate_manifest(
    path: str | Path,
    pairs: Sequence[ImagePair],
    records: Sequence[CropRecord],
    *,
    patch: int,
    seed: int,
    sampling_mode: str = "iid_with_replacement",
) -> str:
    path = Path(path)
    start_sample = records[0].sample_index if records else 0
    if sampling_mode not in VALID_CROP_SAMPLING_MODES:
        raise ValueError(f"unknown crop sampling mode: {sampling_mode}")
    balanced = sampling_mode == "balanced_permutation"
    metadata = {
        "schema": (
            "clpo2027.ohaze_balanced_crop_manifest.v1"
            if balanced and start_sample == 0
            else "clpo2027.ohaze_balanced_crop_suffix_manifest.v1"
            if balanced
            else "clpo2027.ohaze_crop_manifest.v1"
            if start_sample == 0
            else "clpo2027.ohaze_crop_suffix_manifest.v1"
        ),
        "dataset_identity": dataset_identity(pairs),
        "patch": patch,
        "seed": seed,
        "total_samples": len(records),
        "pair_ids": [pair.image_id for pair in pairs],
    }
    if balanced:
        metadata["sampling_mode"] = sampling_mode
    if start_sample != 0:
        metadata["start_sample"] = start_sample
    payload = {"metadata": metadata, "records": [asdict(record) for record in records]}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if hashlib.sha256(existing.encode("utf-8")).hexdigest() != digest:
            raise RuntimeError(f"existing crop manifest differs from the frozen manifest: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(canonical, encoding="utf-8")
        temporary.replace(path)
    return digest


def _read_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_tensor(path: str | Path) -> Tensor:
    array = _read_uint8(Path(path))
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).float().div_(255.0)


def read_center_crop_pair(
    pair: ImagePair,
    *,
    size: int = 512,
) -> tuple[Tensor, Tensor]:
    """Read one deterministic square center crop for train-only proxy audits."""

    if size <= 0:
        raise ValueError("center-crop size must be positive")
    hazy = _read_uint8(pair.hazy_path)
    target = _read_uint8(pair.gt_path)
    if hazy.shape != target.shape:
        raise RuntimeError(f"decoded shape mismatch for {pair.image_id}")
    height, width = hazy.shape[:2]
    if min(height, width) < size:
        raise RuntimeError(
            f"image {pair.image_id} is smaller than center crop {size}: "
            f"{height}x{width}"
        )
    y = (height - size) // 2
    x = (width - size) // 2

    def tensor(array: np.ndarray) -> Tensor:
        crop = np.ascontiguousarray(
            array[y:y + size, x:x + size].transpose(2, 0, 1)
        )
        return torch.from_numpy(crop).float().div_(255.0)

    return tensor(hazy), tensor(target)


VALID_AUGMENTATION_MODES = frozenset({"full_dihedral", "outdoor_safe"})


class ManifestCropDataset:
    """Index-stable crop stream; decoded source images are retained as uint8."""

    def __init__(
        self,
        pairs: Sequence[ImagePair],
        records: Sequence[CropRecord],
        patch: int,
        *,
        augmentation_mode: str = "full_dihedral",
        cache_size_pairs: int | None = None,
    ) -> None:
        if augmentation_mode not in VALID_AUGMENTATION_MODES:
            raise ValueError(
                f"unknown augmentation_mode={augmentation_mode!r}; "
                f"expected one of {sorted(VALID_AUGMENTATION_MODES)}"
            )
        if cache_size_pairs is not None and cache_size_pairs <= 0:
            raise ValueError("cache_size_pairs must be positive or None")
        self.pairs = tuple(pairs)
        self.records = tuple(records)
        self.patch = patch
        self.augmentation_mode = augmentation_mode
        self.cache_size_pairs = cache_size_pairs
        self._cache: OrderedDict[
            int, tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()
        self._scene_thumbnail_cache: dict[
            tuple[int, int], Tensor
        ] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _images(self, pair_index: int) -> tuple[np.ndarray, np.ndarray]:
        if pair_index in self._cache:
            self._cache.move_to_end(pair_index)
            return self._cache[pair_index]
        pair = self.pairs[pair_index]
        hazy, gt = _read_uint8(pair.hazy_path), _read_uint8(pair.gt_path)
        if hazy.shape != gt.shape:
            raise RuntimeError(f"decoded shape mismatch for {pair.image_id}")
        if self.cache_size_pairs is not None:
            while len(self._cache) >= self.cache_size_pairs:
                self._cache.popitem(last=False)
        self._cache[pair_index] = hazy, gt
        return hazy, gt

    def pair_residual_severities(self) -> tuple[float, ...]:
        """Return train-only mean absolute hazy-clear residual per source."""

        severities = []
        for pair_index in range(len(self.pairs)):
            hazy, gt = self._images(pair_index)
            severities.append(
                float(
                    np.abs(hazy.astype(np.float32) - gt.astype(np.float32)).mean()
                    / 255.0
                )
            )
        return tuple(severities)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        hazy, gt = self._images(record.pair_index)
        p, y, x = self.patch, record.crop_y, record.crop_x
        hazy_crop = hazy[y:y + p, x:x + p]
        gt_crop = gt[y:y + p, x:x + p]
        if record.flip_h:
            hazy_crop, gt_crop = hazy_crop[:, ::-1], gt_crop[:, ::-1]
        if self.augmentation_mode == "full_dihedral":
            if record.flip_v:
                hazy_crop, gt_crop = hazy_crop[::-1], gt_crop[::-1]
            if record.rotation:
                hazy_crop = np.rot90(hazy_crop, record.rotation)
                gt_crop = np.rot90(gt_crop, record.rotation)

        def tensor(array: np.ndarray) -> Tensor:
            contiguous = np.ascontiguousarray(array.transpose(2, 0, 1))
            return torch.from_numpy(contiguous).float().div_(255.0)

        return tensor(hazy_crop), tensor(gt_crop)

    def scene_context(
        self,
        record: CropRecord,
        *,
        size: int = 64,
    ) -> Tensor:
        """Return a PyTorch-bilinear scene thumbnail under crop augmentation."""

        if size <= 0:
            raise ValueError("scene context size must be positive")
        cache_key = (record.pair_index, int(size))
        if cache_key not in self._scene_thumbnail_cache:
            hazy, _gt = self._images(record.pair_index)
            source = torch.from_numpy(
                np.ascontiguousarray(hazy.transpose(2, 0, 1))
            ).float().div_(255.0)
            self._scene_thumbnail_cache[cache_key] = F.interpolate(
                source.unsqueeze(0),
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        context = self._scene_thumbnail_cache[cache_key]
        if record.flip_h:
            context = torch.flip(context, dims=(-1,))
        if self.augmentation_mode == "full_dihedral":
            if record.flip_v:
                context = torch.flip(context, dims=(-2,))
            if record.rotation:
                context = torch.rot90(
                    context,
                    record.rotation,
                    dims=(-2, -1),
                )
        return context.contiguous().clone()

    def batch(self, iteration: int, batch_size: int) -> tuple[Tensor, Tensor]:
        hazy, target, _records = self.batch_with_records(iteration, batch_size)
        return hazy, target

    def batch_with_records(
        self, iteration: int, batch_size: int,
    ) -> tuple[Tensor, Tensor, tuple[CropRecord, ...]]:
        start = iteration * batch_size
        indices = tuple(range(start, start + batch_size))
        items = [self[index] for index in indices]
        return (
            torch.stack([item[0] for item in items]),
            torch.stack([item[1] for item in items]),
            tuple(self.records[index] for index in indices),
        )

    def batch_with_scene_context(
        self,
        iteration: int,
        batch_size: int,
        *,
        context_size: int = 64,
    ) -> tuple[Tensor, Tensor, Tensor, tuple[CropRecord, ...]]:
        """Return the matched crop batch plus its whole-source thumbnails."""

        hazy, target, records = self.batch_with_records(iteration, batch_size)
        context = torch.stack([
            self.scene_context(record, size=context_size)
            for record in records
        ])
        return hazy, target, context, records
