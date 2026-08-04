#!/usr/bin/env python3
"""Prepare drawings for YOLO training: resize, convert, sanitize names."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def sanitize_stem(name: str) -> str:
    stem = Path(name).stem
    stem = stem.strip().replace(" ", "_")
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem or "image"


def prepare_image(
    src_path: Path,
    dst_path: Path,
    target_size: int = 1500,
    quality: int = 85,
) -> dict:
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {src_path}")

    h, w = img.shape[:2]
    orig_size = (w, h)
    long_side = max(h, w)
    if long_side > target_size:
        scale = target_size / long_side
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(dst_path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(str(dst_path), img)

    return {
        "original": str(src_path),
        "output": str(dst_path),
        "orig_size": orig_size,
        "new_size": (img.shape[1], img.shape[0]),
    }


def batch_prepare(
    input_dir: Path,
    output_dir: Path,
    target_size: int = 1500,
    quality: int = 85,
    prefix: str = "custom_",
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    images: list[Path] = []
    for ext in IMAGE_EXTS:
        images.extend(input_dir.glob(f"*{ext}"))
        images.extend(input_dir.glob(f"*{ext.upper()}"))
    images = sorted({p.resolve() for p in images})

    used: set[str] = set()
    print(f"Found {len(images)} images in {input_dir}")
    for i, src in enumerate(images):
        try:
            stem = sanitize_stem(src.name)
            if not stem.startswith(prefix):
                stem = f"{prefix}{stem}"
            base = stem
            n = 1
            while stem in used or (output_dir / f"{stem}.jpg").exists():
                stem = f"{base}_{n}"
                n += 1
            used.add(stem)
            dst = output_dir / f"{stem}.jpg"
            result = prepare_image(src, dst, target_size, quality)
            results.append(result)
            if (i + 1) % 20 == 0:
                print(f"  processed {i + 1}/{len(images)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {src.name}: {exc}")
    print(f"Prepared {len(results)} images -> {output_dir}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=1500)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--prefix", default="custom_")
    args = parser.parse_args()
    batch_prepare(args.input, args.output, args.target_size, args.quality, args.prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
