#!/usr/bin/env python3
"""Merge custom labeled images into datasets/combined and refresh data.yaml."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMBINED = ROOT / "datasets" / "combined"
NAMES = ["wall", "door", "window", "stair", "partition"]


def write_data_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    # Absolute path works both on host and when repo is mounted at /app
    content = (
        f"# Auto-generated YOLO-seg floorplan dataset\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(NAMES)}\n"
        f"names: {NAMES}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def merge_custom(
    custom_images: Path,
    custom_labels: Path,
    combined: Path = COMBINED,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    train_img = combined / "images" / "train"
    train_lbl = combined / "labels" / "train"
    val_img = combined / "images" / "val"
    val_lbl = combined / "labels" / "val"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Path]] = []
    for img in sorted(custom_images.iterdir()):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl = custom_labels / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        # skip empty pseudo-labels
        if not lbl.read_text(encoding="utf-8").strip():
            continue
        pairs.append((img, lbl))

    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if pairs else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def copy_pair(img: Path, lbl: Path, img_dir: Path, lbl_dir: Path) -> None:
        shutil.copy2(img, img_dir / img.name)
        shutil.copy2(lbl, lbl_dir / lbl.name)

    for img, lbl in train_pairs:
        copy_pair(img, lbl, train_img, train_lbl)
    for img, lbl in val_pairs:
        copy_pair(img, lbl, val_img, val_lbl)

    yaml_path = write_data_yaml(combined)
    return {
        "custom_merged": len(pairs),
        "custom_train": len(train_pairs),
        "custom_val": len(val_pairs),
        "train_images": len(list(train_img.glob("*"))),
        "val_images": len(list(val_img.glob("*"))),
        "data_yaml": str(yaml_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=ROOT / "datasets" / "custom" / "images")
    parser.add_argument("--labels", type=Path, default=ROOT / "datasets" / "custom" / "labels")
    parser.add_argument("--combined", type=Path, default=COMBINED)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    stats = merge_custom(args.images, args.labels, args.combined, args.val_ratio)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
