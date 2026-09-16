#!/usr/bin/env python3
"""Thermal (hot-region) video processing pipeline.

Reads a video frame-by-frame, converts to grayscale, normalizes intensity,
segments the brightest ("hot") regions via per-frame Otsu thresholding,
cleans the mask with morphological operations, finds contours, filters
them by area, and draws labeled bounding boxes on a copy of each frame.
The result is written to a new video file; the source video is never
read in write mode and is never modified.

NOTE ON DATA TYPE (read this before trusting the output):
This script treats pixel intensity as a *visual* hot/bright-region signal,
not calibrated temperature. It works on any video that can be converted to
8-bit grayscale -- a standard visualized thermal export, or (for now) a
placeholder RGB video used to validate the pipeline mechanics. It does NOT
interpret pixel values as real-world Celsius/Kelvin, and it must not be
described that way in demos or reports unless the input is genuinely
radiometric (per-pixel temperature) data in a raw format such as a 16-bit
FLIR radiometric export. Swapping in real thermal footage later requires
no code changes -- just point --input at the new file and re-tune
--min-area / --threshold if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=Path("assets/test_cid.mp4"),
        help="Path to the source video (never modified). Default: assets/test_cid.mp4",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/thermal/test_cid_hotregions.mp4"),
        help="Path to write the processed video. Default: results/thermal/test_cid_hotregions.mp4",
    )
    parser.add_argument(
        "--min-area", type=int, default=150,
        help="Minimum contour area in pixels to keep as a hot region (noise filter). Default: 150",
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Fixed 0-255 brightness threshold. If omitted, Otsu's method picks "
             "a threshold automatically for each frame (recommended default).",
    )
    parser.add_argument(
        "--blur-kernel", type=int, default=5,
        help="Gaussian blur kernel size (odd number) applied before thresholding, "
             "to reduce sensor/compression noise. Use 0 to disable. Default: 5",
    )
    parser.add_argument(
        "--morph-kernel", type=int, default=5,
        help="Kernel size for morphological open/close cleanup of the hot-region "
             "mask (removes speckle, fills small gaps). Default: 5",
    )
    return parser.parse_args()


def detect_hot_regions(
    gray_frame: np.ndarray,
    threshold: int | None,
    blur_kernel: int,
    morph_kernel: int,
) -> np.ndarray:
    """Return a binary mask (0/255) of hot/bright regions for one grayscale frame."""
    working = gray_frame

    if blur_kernel and blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        working = cv2.GaussianBlur(working, (k, k), 0)

    if threshold is None:
        _, mask = cv2.threshold(working, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(working, threshold, 255, cv2.THRESH_BINARY)

    if morph_kernel and morph_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input video not found: {args.input}")

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"OpenCV could not open VideoWriter for output: {args.output}")

    frame_index = 0
    total_regions = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            mask = detect_hot_regions(
                normalized, args.threshold, args.blur_kernel, args.morph_kernel
            )
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            annotated = frame.copy()
            frame_regions = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < args.min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    annotated, "Hot Region", (x, max(y - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
                frame_regions += 1

            total_regions += frame_regions
            writer.write(annotated)
            frame_index += 1

    finally:
        cap.release()
        writer.release()

    print(f"Processed {frame_index}/{total_frames} frames -> {args.output}")
    print(f"Total hot-region detections (summed across all frames): {total_regions}")


if __name__ == "__main__":
    main()
