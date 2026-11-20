"""Build the paper directory layouts and splits from official datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".JPEG"}


def place(source: Path, target: Path, copy: bool) -> None:
    """Use hard links by default; use --copy across storage volumes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.samefile(source):
            return
        with source.open("rb") as left, target.open("rb") as right:
            if hashlib.file_digest(left, "sha256").digest() == hashlib.file_digest(right, "sha256").digest():
                return
        raise FileExistsError(f"The target contains different data: {target}")
    shutil.copy2(source, target) if copy else os.link(source, target)


def indexed_files(folder: Path, marker: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            try:
                index = int(path.name.split("_", 1)[0])
            except ValueError:
                continue
            if marker.lower() in path.stem.lower():
                result[index] = path
    return result


def prepare_ohaze(source: Path, output: Path, copy: bool) -> None:
    """Split the 45 official outdoor pairs into IDs 1--40 and 41--45."""
    gt_candidates = [source / "GT", source / "gt", source / "train" / "GT"]
    hazy_candidates = [source / "hazy", source / "Hazy", source / "train" / "hazy"]
    # Also accept official archives already arranged into train/test folders.
    gt: dict[int, Path] = {}
    hazy: dict[int, Path] = {}
    for split in ("train", "test"):
        for folder, marker, target in (
            (source / split / "GT", "GT", gt),
            (source / split / "gt", "GT", gt),
            (source / split / "hazy", "hazy", hazy),
        ):
            if folder.is_dir():
                target.update(indexed_files(folder, marker))
    for folder in gt_candidates:
        if folder.is_dir():
            gt.update(indexed_files(folder, "GT"))
    for folder in hazy_candidates:
        if folder.is_dir():
            hazy.update(indexed_files(folder, "hazy"))
    expected = set(range(1, 46))
    if set(gt) != expected or set(hazy) != expected:
        raise RuntimeError(f"O-HAZE requires 45 pairs; found GT={len(gt)}, hazy={len(hazy)}")
    for index in sorted(expected):
        split = "train" if index <= 40 else "test"
        suffix = gt[index].suffix.lower()
        if hazy[index].suffix.lower() != suffix:
            raise RuntimeError(f"O-HAZE pair {index} has mismatched file extensions")
        name = f"{index:02d}{suffix}"
        place(gt[index], output / split / "gt" / name, copy)
        place(hazy[index], output / split / "hazy" / name, copy)
    print(f"O-HAZE ready: train=40, test=5, root={output}")


def prepare_gopro(source: Path, output: Path, copy: bool) -> None:
    """Preserve official test sequences and flatten only the VIVNet training set."""
    train_sequences = sorted(path for path in (source / "train").iterdir() if path.is_dir())
    test_sequences = sorted(path for path in (source / "test").iterdir() if path.is_dir())
    train_count = 0
    for sequence in train_sequences:
        blur_files = sorted((sequence / "blur").glob("*.png"))
        for blur in blur_files:
            sharp = sequence / "sharp" / blur.name
            if not sharp.is_file():
                raise FileNotFoundError(sharp)
            name = f"{sequence.name}__{blur.name}"
            place(blur, output / "train" / "blur" / name, copy)
            place(sharp, output / "train" / "sharp" / name, copy)
            train_count += 1
    test_count = 0
    for sequence in test_sequences:
        for blur in sorted((sequence / "blur").glob("*.png")):
            sharp = sequence / "sharp" / blur.name
            if not sharp.is_file():
                raise FileNotFoundError(sharp)
            place(blur, output / "test" / sequence.name / "blur" / blur.name, copy)
            place(sharp, output / "test" / sequence.name / "sharp" / blur.name, copy)
            test_count += 1
    if (train_count, test_count) != (2103, 1111):
        raise RuntimeError(f"GoPro count mismatch: train={train_count}, test={test_count}")
    print(f"GoPro ready: train={train_count}, test={test_count}, root={output}")


def prepare_imagenet100(source: Path, output: Path, copy: bool) -> None:
    """Select the fixed 100 classes from WNID-organized ILSVRC2012 train/val."""
    split_root = ROOT / "splits" / "imagenet100"
    with (split_root / "classes.csv").open(encoding="utf-8-sig", newline="") as stream:
        wnids = [row["wnid"] for row in csv.DictReader(stream)]
    if len(wnids) != 100 or len(set(wnids)) != 100:
        raise RuntimeError("classes.csv must contain 100 unique WNIDs")
    with (split_root / "official_selection.csv").open(encoding="utf-8-sig", newline="") as stream:
        expected_val = {row["file"] for row in csv.DictReader(stream)}
    with (split_root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        expected_files = {(row["split"], row["wnid"], row["file"]): int(row["bytes"])
                          for row in csv.DictReader(stream)}
    actual_files = set()
    counts: dict[str, int] = {}
    actual_val: set[str] = set()
    for split in ("train", "val"):
        count = 0
        for wnid in wnids:
            folder = source / split / wnid
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            for image in sorted(path for path in folder.iterdir() if path.suffix in IMAGE_SUFFIXES):
                key = (split, wnid, image.name)
                if key not in expected_files or image.stat().st_size != expected_files[key]:
                    raise RuntimeError(f"The image does not match the paper manifest: {image}")
                actual_files.add(key)
                place(image, output / split / wnid / image.name, copy)
                count += 1
                if split == "val":
                    actual_val.add(image.name)
        counts[split] = count
    if counts != {"train": 126689, "val": 5000}:
        raise RuntimeError(f"ImageNet-100 count mismatch: {counts}")
    if actual_val != expected_val:
        raise RuntimeError("ImageNet-100 validation files differ from the frozen manifest")
    if actual_files != set(expected_files):
        raise RuntimeError("The ImageNet-100 files differ from the paper manifest.")
    evidence = output / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    shutil.copy2(split_root / "classes.csv", evidence / "classes.csv")
    print(f"ImageNet-100 ready: {counts}, root={output}")


def check_cityscapes(source: Path) -> None:
    counts: dict[str, int] = {}
    for split, expected in (("train", 2975), ("val", 500)):
        images = list((source / "leftImg8bit" / split).glob("*/*_leftImg8bit.png"))
        labels = list((source / "gtFine" / split).glob("*/*_gtFine_labelIds.png"))
        if len(images) != expected or len(labels) != expected:
            raise RuntimeError(f"Cityscapes {split}: images={len(images)}, labels={len(labels)}, expected={expected}")
        for image in images:
            label = source / "gtFine" / split / image.parent.name / image.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
            if not label.is_file():
                raise FileNotFoundError(label)
        counts[split] = expected
    print(f"Cityscapes official split verified: {counts}, root={source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the paper dataset splits")
    parser.add_argument("--dataset", choices=("ohaze", "gopro", "cityscapes", "imagenet100"), required=True)
    parser.add_argument("--source", type=Path, required=True, help="official dataset root")
    parser.add_argument("--output", type=Path, help="output root; omitted for Cityscapes validation")
    parser.add_argument("--copy", action="store_true", help="copy files instead of creating hard links")
    args = parser.parse_args()
    if args.dataset == "cityscapes":
        check_cityscapes(args.source)
        return
    if args.output is None:
        raise ValueError(f"{args.dataset} requires --output")
    {"ohaze": prepare_ohaze, "gopro": prepare_gopro, "imagenet100": prepare_imagenet100}[args.dataset](
        args.source, args.output, args.copy
    )


if __name__ == "__main__":
    main()
