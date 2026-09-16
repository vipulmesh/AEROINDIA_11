#!/usr/bin/env python3
"""Validate image/label pairs and one-class YOLO boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("VAYU-NETRA-Dataset"))
    args = parser.parse_args()
    missing = empty = invalid = total_instances = 0
    split_counts = {}
    for split in ("train", "val", "test"):
        image_root = args.dataset / "images" / split
        label_root = args.dataset / "labels" / split
        images = sorted(p for p in image_root.iterdir() if p.is_file())
        split_counts[split] = len(images)
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in label_root.glob("*.txt")}
        missing += len(image_stems - label_stems)
        for image in images:
            label = label_root / f"{image.stem}.txt"
            if not label.is_file():
                continue
            try:
                width, height = Image.open(image).size
            except Exception:
                invalid += 1
                continue
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
            if not rows:
                empty += 1
            for fields in rows:
                total_instances += 1
                if len(fields) != 5:
                    invalid += 1
                    continue
                try:
                    class_id, center_x, center_y, box_width, box_height = map(float, fields)
                except ValueError:
                    invalid += 1
                    continue
                if class_id != 0 or not (0 < box_width <= 1 and 0 < box_height <= 1):
                    invalid += 1
                    continue
                x1 = center_x - box_width / 2
                y1 = center_y - box_height / 2
                x2 = center_x + box_width / 2
                y2 = center_y + box_height / 2
                epsilon = 1e-8
                if not (-epsilon <= x1 < x2 <= 1 + epsilon and -epsilon <= y1 < y2 <= 1 + epsilon):
                    invalid += 1
        missing += len(label_stems - image_stems)
    result = {"train_images": split_counts["train"], "val_images": split_counts["val"], "test_images": split_counts["test"], "total_person_instances": total_instances, "missing_labels": missing, "empty_labels": empty, "invalid_bounding_boxes": invalid}
    print(json.dumps(result, indent=2))
    if missing or empty or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()