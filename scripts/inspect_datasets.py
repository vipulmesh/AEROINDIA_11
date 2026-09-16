#!/usr/bin/env python3
"""Inspect the C2A and VisDrone sources without modifying them."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c2a-root", type=Path, default=Path("C2A_DataSet"))
    parser.add_argument("--visdrone-root", type=Path, default=Path("visDroneDATASET"))
    args = parser.parse_args()

    print("C2A")
    for split in ("train", "val", "test"):
        root = args.c2a_root / "C2A_Dataset" / "new_dataset3" / split
        images = [p for p in (root / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        labels = list((root / "labels").glob("*.txt"))
        print(f"  {split}: {len(images)} images, {len(labels)} labels")

    print("VisDrone")
    classes = Counter()
    for root in sorted(args.visdrone_root.glob("VisDrone2019-DET-*/*")):
        image_root = root / "images"
        annotation_root = root / "annotations"
        if not image_root.is_dir():
            continue
        images = [p for p in image_root.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        annotations = list(annotation_root.glob("*.txt")) if annotation_root.is_dir() else []
        rows = 0
        for annotation in annotations:
            for line in annotation.read_text(errors="replace").splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 8:
                    rows += 1
                    classes[fields[5]] += 1
        print(f"  {root.parent.name}: {len(images)} images, {len(annotations)} annotations, {rows} rows")
    print("VisDrone class ids:", json.dumps(dict(sorted(classes.items())), sort_keys=True))


if __name__ == "__main__":
    main()