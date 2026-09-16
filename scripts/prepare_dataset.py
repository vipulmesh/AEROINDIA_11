#!/usr/bin/env python3
"""Build a one-class YOLO detection dataset from C2A and VisDrone."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VISDRONE_PERSON_CLASSES = {1, 2}  # pedestrian, people


@dataclass
class Stats:
    images_seen: int = 0
    images_processed: int = 0
    person_instances: int = 0
    images_skipped: int = 0
    invalid_annotations: int = 0
    duplicate_images: int = 0
    empty_person_images: int = 0
    corrupt_images: int = 0
    missing_annotations: int = 0


@dataclass
class Candidate:
    source: str
    source_image: Path
    source_label: Path | None
    image_name: str
    labels: list[str]
    digest: str
    stats: Stats


def image_files(root: Path) -> Iterable[Path]:
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def normalized_box(x1: float, y1: float, width: float, height: float, image_width: int, image_height: int) -> str | None:
    x2 = x1 + width
    y2 = y1 + height
    x1 = max(0.0, min(float(image_width), x1))
    y1 = max(0.0, min(float(image_height), y1))
    x2 = max(0.0, min(float(image_width), x2))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    center_x = ((x1 + x2) / 2) / image_width
    center_y = ((y1 + y2) / 2) / image_height
    box_width = (x2 - x1) / image_width
    box_height = (y2 - y1) / image_height
    return f"0 {center_x:.9f} {center_y:.9f} {box_width:.9f} {box_height:.9f}"


def c2a_candidates(root: Path, stats: Stats) -> list[Candidate]:
    candidates = []
    source_root = root / "C2A_Dataset" / "new_dataset3"
    for split in ("train", "val", "test"):
        image_root = source_root / split / "images"
        label_root = source_root / split / "labels"
        for image in image_files(image_root):
            stats.images_seen += 1
            stats.images_processed += 1
            size = image_size(image)
            label = label_root / f"{image.stem}.txt"
            if size is None:
                stats.images_processed -= 1
                stats.images_skipped += 1
                stats.corrupt_images += 1
                continue
            if not label.is_file():
                stats.images_processed -= 1
                stats.images_skipped += 1
                stats.missing_annotations += 1
                continue
            labels = []
            for line in label.read_text(errors="replace").splitlines():
                fields = line.split()
                if len(fields) < 5:
                    stats.invalid_annotations += 1
                    continue
                try:
                    class_id, center_x, center_y, width, height = map(float, fields[:5])
                except ValueError:
                    stats.invalid_annotations += 1
                    continue
                if class_id != 0:
                    stats.invalid_annotations += 1
                    continue
                box = normalized_box(
                    (center_x - width / 2) * size[0],
                    (center_y - height / 2) * size[1],
                    width * size[0],
                    height * size[1],
                    *size,
                )
                if box is None:
                    stats.invalid_annotations += 1
                else:
                    labels.append(box)
            if not labels:
                stats.images_processed -= 1
                stats.images_skipped += 1
                stats.empty_person_images += 1
                continue
            stats.person_instances += len(labels)
            candidates.append(Candidate("C2A", image, label, f"C2A_{len(candidates):06d}{image.suffix.lower()}", labels, digest(image), stats))
    return candidates


def visdrone_candidates(root: Path, stats: Stats, start_index: int = 0) -> list[Candidate]:
    candidates = []
    for dataset_root in sorted(root.glob("VisDrone2019-DET-*/*")):
        image_root = dataset_root / "images"
        annotation_root = dataset_root / "annotations"
        if not image_root.is_dir() or not annotation_root.is_dir():
            continue
        for image in image_files(image_root):
            stats.images_seen += 1
            stats.images_processed += 1
            size = image_size(image)
            label = annotation_root / f"{image.stem}.txt"
            if size is None:
                stats.images_processed -= 1
                stats.images_skipped += 1
                stats.corrupt_images += 1
                continue
            if not label.is_file():
                stats.images_processed -= 1
                stats.images_skipped += 1
                stats.missing_annotations += 1
                continue
            labels = []
            for line in label.read_text(errors="replace").splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 8:
                    stats.invalid_annotations += 1
                    continue
                try:
                    x, y, width, height = map(float, fields[:4])
                    class_id = int(fields[5])
                except ValueError:
                    stats.invalid_annotations += 1
                    continue
                if class_id not in VISDRONE_PERSON_CLASSES:
                    continue
                box = normalized_box(x, y, width, height, *size)
                if box is None:
                    stats.invalid_annotations += 1
                else:
                    labels.append(box)
            if not labels:
                stats.images_processed -= 1
                stats.images_skipped += 1
                stats.empty_person_images += 1
                continue
            stats.person_instances += len(labels)
            candidates.append(Candidate("VisDrone", image, label, f"VisDrone_{start_index + len(candidates):06d}{image.suffix.lower()}", labels, digest(image), stats))
    return candidates


def deduplicate(candidates: list[Candidate], stats: Stats) -> list[Candidate]:
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate.digest in seen:
            stats.duplicate_images += 1
            stats.images_processed -= 1
            stats.images_skipped += 1
            stats.person_instances -= len(candidate.labels)
            continue
        seen.add(candidate.digest)
        unique.append(candidate)
    return unique


def write_dataset(candidates: list[Candidate], output: Path, seed: int) -> None:
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = round(total * 0.70)
    val_end = train_end + round(total * 0.20)
    assignments = [("train", candidate) for candidate in shuffled[:train_end]]
    assignments += [("val", candidate) for candidate in shuffled[train_end:val_end]]
    assignments += [("test", candidate) for candidate in shuffled[val_end:]]
    for split, candidate in assignments:
        shutil.copy2(candidate.source_image, output / "images" / split / candidate.image_name)
        (output / "labels" / split / f"{Path(candidate.image_name).stem}.txt").write_text("\n".join(candidate.labels) + "\n")
    (output / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: person\n")

    counts = {split: sum(1 for name, _ in assignments if name == split) for split in ("train", "val", "test")}
    (output / "summary.json").write_text(json.dumps({"splits": counts}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2a-root", type=Path, default=Path("C2A_DataSet"))
    parser.add_argument("--visdrone-root", type=Path, default=Path("visDroneDATASET"))
    parser.add_argument("--output", type=Path, default=Path("VAYU-NETRA-Dataset"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Output is not empty: {args.output}. Use --overwrite to rebuild it.")
        shutil.rmtree(args.output)

    c2a_stats = Stats()
    vis_stats = Stats()
    candidates = c2a_candidates(args.c2a_root, c2a_stats)
    vis_candidates = visdrone_candidates(args.visdrone_root, vis_stats, len(candidates))
    candidates.extend(vis_candidates)
    combined_stats = Stats()
    for stats in (c2a_stats, vis_stats):
        for name in stats.__dataclass_fields__:
            setattr(combined_stats, name, getattr(combined_stats, name) + getattr(stats, name))
    candidates = deduplicate(candidates, combined_stats)
    write_dataset(candidates, args.output, args.seed)
    print(json.dumps({
        "c2a_images_processed": c2a_stats.images_processed,
        "c2a_person_instances": c2a_stats.person_instances,
        "visdrone_images_processed": vis_stats.images_processed,
        "visdrone_person_instances": vis_stats.person_instances,
        "images_skipped": combined_stats.images_skipped,
        "invalid_annotations": combined_stats.invalid_annotations,
        "duplicate_images_removed": combined_stats.duplicate_images,
        "final_candidate_images": len(candidates),
        "total_person_instances": sum(len(candidate.labels) for candidate in candidates),
    }, indent=2))


if __name__ == "__main__":
    main()