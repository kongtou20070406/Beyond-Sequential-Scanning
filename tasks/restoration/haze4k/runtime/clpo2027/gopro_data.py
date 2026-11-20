"""Deterministic GoPro Large pairing, sequence holdout, and crop streaming."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
from PIL import Image
from torch import Tensor
import torch

from .data import ImagePair


GOPRO_SPLIT_COUNTS = {"train": 2103, "test": 1111}


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def load_gopro_pairs(root: str | Path, split: str) -> list[ImagePair]:
    """Load sequence-disjoint GoPro blur/sharp pairs in stable lexical order."""

    if split not in GOPRO_SPLIT_COUNTS:
        raise ValueError(f"split must be one of {sorted(GOPRO_SPLIT_COUNTS)}")
    split_root = Path(root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"missing GoPro split directory: {split_root}")
    pairs: list[ImagePair] = []
    for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        blur_dir = sequence_dir / "blur"
        sharp_dir = sequence_dir / "sharp"
        if not blur_dir.is_dir() or not sharp_dir.is_dir():
            continue
        blur_by_name = {path.name: path for path in _image_files(blur_dir)}
        sharp_by_name = {path.name: path for path in _image_files(sharp_dir)}
        if set(blur_by_name) != set(sharp_by_name):
            raise RuntimeError(
                f"unpaired GoPro frames in {sequence_dir.name}: "
                f"blur_only={sorted(set(blur_by_name) - set(sharp_by_name))[:8]}, "
                f"sharp_only={sorted(set(sharp_by_name) - set(blur_by_name))[:8]}"
            )
        pairs.extend(
            ImagePair(
                image_id=f"{sequence_dir.name}/{name}",
                hazy_path=blur_by_name[name],
                gt_path=sharp_by_name[name],
            )
            for name in sorted(blur_by_name)
        )
    expected = GOPRO_SPLIT_COUNTS[split]
    if len(pairs) != expected:
        raise RuntimeError(
            f"canonical GoPro {split} split must have {expected} pairs, "
            f"found {len(pairs)}"
        )
    return pairs


def gopro_sequences(pairs: Sequence[ImagePair]) -> tuple[str, ...]:
    return tuple(sorted({pair.image_id.split("/", 1)[0] for pair in pairs}))


def deterministic_dev_val_sequences(
    pairs: Sequence[ImagePair],
    *,
    count: int = 2,
    seed: int = 20270727,
) -> tuple[str, ...]:
    """Choose whole training sequences without consulting image quality."""

    sequences = gopro_sequences(pairs)
    if count <= 0 or count >= len(sequences):
        raise ValueError("dev-val count must leave at least one training sequence")

    def key(sequence: str) -> bytes:
        return hashlib.sha256(f"{seed}\0{sequence}".encode("utf-8")).digest()

    return tuple(sorted(sorted(sequences, key=key)[:count]))


def split_by_sequences(
    pairs: Sequence[ImagePair],
    val_sequences: Sequence[str],
) -> tuple[list[ImagePair], list[ImagePair]]:
    held_out = frozenset(val_sequences)
    available = frozenset(gopro_sequences(pairs))
    if not held_out or not held_out <= available:
        raise ValueError(
            f"invalid held-out sequences: {sorted(held_out - available)}"
        )
    train = [
        pair for pair in pairs if pair.image_id.split("/", 1)[0] not in held_out
    ]
    val = [
        pair for pair in pairs if pair.image_id.split("/", 1)[0] in held_out
    ]
    if not train or not val:
        raise RuntimeError("sequence split produced an empty train or validation set")
    return train, val


def gopro_dataset_identity(
    root: str | Path,
    split: str,
    pairs: Sequence[ImagePair],
) -> str:
    """Path-independent identity over relative names and byte sizes."""

    split_root = (Path(root) / split).resolve()
    digest = hashlib.sha256()
    for pair in pairs:
        digest.update(pair.image_id.encode("utf-8"))
        digest.update(b"\0")
        for path in (pair.hazy_path, pair.gt_path):
            relative = path.resolve().relative_to(split_root)
            digest.update(relative.as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class GoProCropPlan:
    pair_index: np.ndarray
    crop_y: np.ndarray
    crop_x: np.ndarray
    transform: np.ndarray
    patch: int
    seed: int

    def __len__(self) -> int:
        return int(self.pair_index.shape[0])

    def digest(self) -> str:
        value = hashlib.sha256()
        value.update(b"clpo2027.gopro_crop_plan.v2\0")
        value.update(str(self.patch).encode("ascii"))
        value.update(b"\0")
        value.update(str(self.seed).encode("ascii"))
        value.update(b"\0")
        for array in (
            self.pair_index,
            self.crop_y,
            self.crop_x,
            self.transform,
        ):
            value.update(str(array.dtype).encode("ascii"))
            value.update(b"\0")
            value.update(array.tobytes(order="C"))
        return value.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.height, image.width


def build_gopro_crop_plan(
    pairs: Sequence[ImagePair],
    *,
    total_samples: int,
    patch: int,
    seed: int,
) -> GoProCropPlan:
    if not pairs or total_samples <= 0 or patch <= 0:
        raise ValueError("pairs, total_samples, and patch must be positive")
    sizes: list[tuple[int, int]] = []
    for pair in pairs:
        blur_size = _image_size(pair.hazy_path)
        sharp_size = _image_size(pair.gt_path)
        if blur_size != sharp_size:
            raise RuntimeError(
                f"GoPro shape mismatch for {pair.image_id}: "
                f"blur={blur_size}, sharp={sharp_size}"
            )
        if min(blur_size) < patch:
            raise RuntimeError(
                f"GoPro frame {pair.image_id} is smaller than patch={patch}"
            )
        sizes.append(blur_size)

    pair_index = np.empty(total_samples, dtype=np.int32)
    crop_y = np.empty(total_samples, dtype=np.int32)
    crop_x = np.empty(total_samples, dtype=np.int32)
    transform = np.empty(total_samples, dtype=np.uint8)
    rng = random.Random(seed)
    indices: list[int] = []
    while len(indices) < total_samples:
        block = list(range(len(pairs)))
        rng.shuffle(block)
        indices.extend(block)
    for sample in range(total_samples):
        index = indices[sample]
        height, width = sizes[index]
        pair_index[sample] = index
        crop_y[sample] = rng.randint(0, height - patch)
        crop_x[sample] = rng.randint(0, width - patch)
        transform[sample] = rng.randrange(8)
    return GoProCropPlan(pair_index, crop_y, crop_x, transform, patch, seed)


def write_or_validate_gopro_manifest(
    path: str | Path,
    *,
    root: str | Path,
    split: str,
    pairs: Sequence[ImagePair],
    plan: GoProCropPlan,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "clpo2027.gopro_crop_manifest.v2",
        "dataset": "GoPro Large",
        "split": split,
        "dataset_identity": gopro_dataset_identity(root, split, pairs),
        "pair_count": len(pairs),
        "pair_ids": [pair.image_id for pair in pairs],
        "sequence_ids": list(gopro_sequences(pairs)),
        "crop_plan": {
            "schema": "clpo2027.gopro_crop_plan.v2",
            "seed": plan.seed,
            "patch": plan.patch,
            "total_samples": len(plan),
            "digest": plan.digest(),
            "transform": "8-way dihedral",
            "sampling": "shuffled full-dataset blocks",
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    output = Path(path)
    if output.exists():
        if output.read_text(encoding="utf-8") != canonical:
            raise RuntimeError(f"existing GoPro manifest differs: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(canonical, encoding="utf-8")
        temporary.replace(output)
    return payload


def _read_uint8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def preload_gopro_pairs(
    pairs: Sequence[ImagePair],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    values = tuple(
        (_read_uint8(pair.hazy_path), _read_uint8(pair.gt_path))
        for pair in pairs
    )
    for pair, (blur, sharp) in zip(pairs, values, strict=True):
        if blur.shape != sharp.shape:
            raise RuntimeError(f"decoded shape mismatch for {pair.image_id}")
    return values


class GoProCropDataset:
    """Fixed crop plan backed by one process-local decoded uint8 cache."""

    def __init__(
        self,
        pairs: Sequence[ImagePair],
        plan: GoProCropPlan,
        *,
        preload: bool = True,
    ) -> None:
        self.pairs = tuple(pairs)
        self.plan = plan
        self._preloaded = (
            preload_gopro_pairs(self.pairs)
            if preload
            else None
        )
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.plan)

    def _images(self, pair_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._preloaded is not None:
            return self._preloaded[pair_index]
        if pair_index in self._cache:
            self._cache.move_to_end(pair_index)
            return self._cache[pair_index]
        pair = self.pairs[pair_index]
        blur = _read_uint8(pair.hazy_path)
        sharp = _read_uint8(pair.gt_path)
        if blur.shape != sharp.shape:
            raise RuntimeError(f"decoded shape mismatch for {pair.image_id}")
        while len(self._cache) >= 8:
            self._cache.popitem(last=False)
        self._cache[pair_index] = (blur, sharp)
        return blur, sharp

    def __getitem__(self, sample: int) -> tuple[Tensor, Tensor]:
        pair_index = int(self.plan.pair_index[sample])
        y = int(self.plan.crop_y[sample])
        x = int(self.plan.crop_x[sample])
        patch = self.plan.patch
        blur, sharp = self._images(pair_index)
        blur_crop = blur[y : y + patch, x : x + patch]
        sharp_crop = sharp[y : y + patch, x : x + patch]
        transform = int(self.plan.transform[sample])
        if transform & 1:
            blur_crop, sharp_crop = blur_crop[:, ::-1], sharp_crop[:, ::-1]
        if transform & 2:
            blur_crop, sharp_crop = blur_crop[::-1], sharp_crop[::-1]
        rotations = (transform >> 2) & 1
        if rotations:
            blur_crop = np.rot90(blur_crop, rotations)
            sharp_crop = np.rot90(sharp_crop, rotations)

        def tensor(value: np.ndarray) -> Tensor:
            chw = np.ascontiguousarray(value.transpose(2, 0, 1))
            return torch.from_numpy(chw).float().div_(255.0)

        return tensor(blur_crop), tensor(sharp_crop)
