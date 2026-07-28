#!/usr/bin/env python3
"""Convert CVC-FP (+ optional synthetic R2V) into a YOLO-seg training set.

Classes (Model 1):
  0 wall | 1 door | 2 window | 3 stair | 4 partition

Usage:
  python scripts/prepare_yolo_dataset.py
  python scripts/prepare_yolo_dataset.py --cvc-only
  python scripts/prepare_yolo_dataset.py --max-r2v 400
"""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CVC_DIR = ROOT / "datasets" / "cvc_fp" / "ImagesGT"
R2V_DIR = (
    ROOT
    / "datasets"
    / "r2v"
    / "FloorplanTransformation"
    / "data"
    / "floorplan_representation"
)
OUT = ROOT / "datasets" / "combined"

CLASS_IDS = {
    "wall": 0,
    "door": 1,
    "window": 2,
    "stair": 3,
    "partition": 4,
}
NAMES = ["wall", "door", "window", "stair", "partition"]

_CLASS_RE = re.compile(r"class\s*=\s*[\"']([^\"']+)[\"']", re.I)
_POINTS_RE = re.compile(r"points\s*=\s*[\"']([^\"']+)[\"']", re.I)


def _map_cvc_class(raw: str) -> str | None:
    key = (raw or "").strip().lower()
    if key.startswith("wall"):
        return "wall"
    if key.startswith("door") or key.startswith("parking"):
        return "door"
    if key.startswith("window"):
        return "window"
    if key.startswith("stair"):
        return "stair"
    if key.startswith("separation"):
        return "partition"
    return None


def _parse_points(blob: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for tok in blob.replace(",", " ").split():
        try:
            pts.append(float(tok))
        except ValueError:
            continue
    out: list[tuple[float, float]] = []
    for i in range(0, len(pts) - 1, 2):
        out.append((pts[i], pts[i + 1]))
    return out


def _poly_to_yolo(pts: list[tuple[float, float]], w: int, h: int) -> str | None:
    if len(pts) < 3 or w < 1 or h < 1:
        return None
    coords: list[str] = []
    for x, y in pts:
        nx = min(max(x / w, 0.0), 1.0)
        ny = min(max(y / h, 0.0), 1.0)
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    # degenerate after clamp
    uniq = {(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)}
    if len(uniq) < 3:
        return None
    return " ".join(coords)


def convert_cvc(out_images: Path, out_labels: Path, prefix: str = "cvc") -> int:
    if not CVC_DIR.is_dir():
        print(f"CVC-FP missing at {CVC_DIR}")
        return 0
    n = 0
    for svg in sorted(CVC_DIR.glob("*_gt_*.svg")):
        base = svg.name.rsplit("_gt_", 1)[0]
        img_path = CVC_DIR / f"{base}.png"
        if not img_path.is_file():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        text = svg.read_text(encoding="utf-8", errors="ignore")
        # Prefer regex over ET — CVC SVGs use nonstandard <width>/<class> tags
        lines: list[str] = []
        for m in re.finditer(r"<polygon\b[^>]*>", text, flags=re.I):
            tag = m.group(0)
            cm = _CLASS_RE.search(tag)
            pm = _POINTS_RE.search(tag)
            if not cm or not pm:
                continue
            mapped = _map_cvc_class(cm.group(1))
            if mapped is None:
                continue
            pts = _parse_points(pm.group(1))
            body = _poly_to_yolo(pts, w, h)
            if body is None:
                continue
            lines.append(f"{CLASS_IDS[mapped]} {body}")
        if not lines:
            continue
        stem = f"{prefix}_{base}"
        dest_img = out_images / f"{stem}.png"
        dest_lbl = out_labels / f"{stem}.txt"
        shutil.copy2(img_path, dest_img)
        dest_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1
    print(f"CVC-FP → {n} samples")
    return n


def _thick_segment_poly(
    x1: float, y1: float, x2: float, y2: float, thickness: float
) -> list[tuple[float, float]]:
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy * thickness / 2, ux * thickness / 2
    return [
        (x1 + px, y1 + py),
        (x2 + px, y2 + py),
        (x2 - px, y2 - py),
        (x1 - px, y1 - py),
    ]


def _parse_r2v_line(line: str) -> tuple[str, list[float]] | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        nums = [float(parts[i]) for i in range(4)]
    except ValueError:
        return None
    label = parts[4].strip().lower()
    return label, nums


def render_r2v_sample(txt_path: Path, wall_t: int = 8, door_t: int = 12) -> tuple[np.ndarray, list[str]] | None:
    rows: list[tuple[str, list[float]]] = []
    try:
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    for line in raw.splitlines():
        parsed = _parse_r2v_line(line)
        if parsed:
            rows.append(parsed)
    if not rows:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for _, nums in rows:
        xs.extend(nums[0:4:2])
        ys.extend(nums[1:4:2])
    if not xs or not ys:
        return None
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 40.0
    w = int(max(max_x - min_x + 2 * pad, 64))
    h = int(max(max_y - min_y + 2 * pad, 64))
    # Cap huge plans
    scale = 1.0
    max_side = 1280
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        w = int(w * scale)
        h = int(h * scale)

    def tx(x: float, y: float) -> tuple[float, float]:
        return ((x - min_x + pad) * scale, (y - min_y + pad) * scale)

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    labels: list[str] = []

    for label, nums in rows:
        x1, y1, x2, y2 = nums
        p1 = tx(x1, y1)
        p2 = tx(x2, y2)
        if label == "wall":
            poly = _thick_segment_poly(*p1, *p2, float(wall_t))
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(canvas, [pts], (20, 20, 20))
            body = _poly_to_yolo(poly, w, h)
            if body:
                labels.append(f"{CLASS_IDS['wall']} {body}")
        elif label == "door":
            poly = _thick_segment_poly(*p1, *p2, float(door_t))
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(canvas, [pts], (60, 60, 200))
            # door swing hint
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            cv2.circle(canvas, (int(mid[0]), int(mid[1])), max(door_t, 6), (100, 100, 220), 1)
            body = _poly_to_yolo(poly, w, h)
            if body:
                labels.append(f"{CLASS_IDS['door']} {body}")
        elif label == "window":
            poly = _thick_segment_poly(*p1, *p2, float(wall_t))
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(canvas, [pts], (200, 140, 40))
            body = _poly_to_yolo(poly, w, h)
            if body:
                labels.append(f"{CLASS_IDS['window']} {body}")

    if not labels:
        return None
    return canvas, labels


def convert_r2v(
    out_images: Path,
    out_labels: Path,
    max_samples: int | None,
    seed: int = 42,
) -> int:
    if not R2V_DIR.is_dir():
        print(f"R2V annotations missing at {R2V_DIR}")
        return 0
    files = sorted(R2V_DIR.rglob("*.txt"))
    rng = random.Random(seed)
    rng.shuffle(files)
    if max_samples is not None:
        files = files[:max_samples]
    n = 0
    for i, path in enumerate(files):
        rendered = render_r2v_sample(path)
        if rendered is None:
            continue
        img, lines = rendered
        stem = f"r2v_{i:05d}"
        cv2.imwrite(str(out_images / f"{stem}.png"), img)
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n += 1
    print(f"R2V synthetic → {n} samples")
    return n


def split_train_val(staging_images: Path, staging_labels: Path, val_ratio: float, seed: int) -> None:
    stems = sorted({p.stem for p in staging_images.glob("*.png") if (staging_labels / f"{p.stem}.txt").is_file()})
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(len(stems) * val_ratio)) if len(stems) > 5 else max(1, len(stems) // 5)
    val_set = set(stems[:n_val])
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
    for stem in stems:
        split = "val" if stem in val_set else "train"
        shutil.copy2(staging_images / f"{stem}.png", OUT / "images" / split / f"{stem}.png")
        shutil.copy2(staging_labels / f"{stem}.txt", OUT / "labels" / split / f"{stem}.txt")
    print(f"Split train={len(stems) - len(val_set)} val={len(val_set)}")


def write_yaml() -> Path:
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        f"""# Auto-generated YOLO-seg floorplan dataset (CVC-FP + R2V)
path: {OUT.resolve()}
train: images/train
val: images/val
nc: {len(NAMES)}
names: {NAMES}
""",
        encoding="utf-8",
    )
    print(f"Wrote {yaml_path}")
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvc-only", action="store_true", help="Skip synthetic R2V")
    parser.add_argument("--max-r2v", type=int, default=800, help="Cap R2V synthetic samples")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Wipe datasets/combined first")
    args = parser.parse_args()

    if args.clean and OUT.exists():
        shutil.rmtree(OUT)
    staging = OUT / "_staging"
    if staging.exists():
        shutil.rmtree(staging)
    img_dir = staging / "images"
    lbl_dir = staging / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    n = convert_cvc(img_dir, lbl_dir)
    if not args.cvc_only:
        n += convert_r2v(img_dir, lbl_dir, max_samples=args.max_r2v, seed=args.seed)
    if n == 0:
        print("No samples prepared. Run: python scripts/download_datasets.py --cvc-fp --r2v")
        return 1

    # Clear previous split dirs
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        p = OUT / sub
        if p.exists():
            shutil.rmtree(p)
    split_train_val(img_dir, lbl_dir, args.val_ratio, args.seed)
    shutil.rmtree(staging)
    write_yaml()
    print(f"Ready: {OUT} ({n} images). Train with: python scripts/train_floorplan_seg.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
