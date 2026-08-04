#!/usr/bin/env python3
"""Synthesize 'dirty scan' variants from clean floorplan drawings.

From CVAT_training.txt: train on synthetic garbage, validate on cleaner /
real photos. Mild geometric warp keeps YOLO labels approximately valid
(± a few percent), which Ultralytics tolerates for seg bootstrap.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def create_garbage_scan(
    img: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if img.ndim == 2:
        gray = img.copy()
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Contrast / brightness (bad scanner)
    alpha = rng.uniform(0.65, 1.35)
    beta = rng.randint(-35, 35)
    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Mild paper ripple (keep small so labels stay usable)
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    amp = rng.uniform(0.5, 2.5)
    period = rng.uniform(28, 70)
    map_x = map_x + amp * np.sin(map_y / period)
    map_y = map_y + amp * np.sin(map_x / period)
    gray = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Sensor noise
    noise = np_rng.normal(0, rng.uniform(4, 14), gray.shape)
    dirty = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Soft stains / shadows
    for _ in range(rng.randint(0, 4)):
        x = rng.randint(0, max(w - 1, 1))
        y = rng.randint(0, max(h - 1, 1))
        r = rng.randint(15, max(16, min(w, h) // 8))
        color = rng.randint(90, 190)
        overlay = dirty.copy()
        cv2.circle(overlay, (x, y), r, int(color), -1)
        dirty = cv2.addWeighted(dirty, 0.82, overlay, 0.18, 0)

    # Occasional blur (out of focus scan)
    if rng.random() < 0.35:
        k = rng.choice([3, 5])
        dirty = cv2.GaussianBlur(dirty, (k, k), 0)

    # Mild perspective (photo of paper)
    if rng.random() < 0.7:
        sx = rng.randint(-12, 12)
        sy = rng.randint(-12, 12)
        pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        pts2 = np.float32(
            [
                [sx, sy],
                [w - 1 + rng.randint(-8, 8), sy + rng.randint(-6, 6)],
                [sx + rng.randint(-6, 6), h - 1 + rng.randint(-8, 8)],
                [w - 1 + rng.randint(-8, 8), h - 1 + rng.randint(-8, 8)],
            ]
        )
        m = cv2.getPerspectiveTransform(pts1, pts2)
        dirty = cv2.warpPerspective(dirty, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return dirty


def refine_label_text(text: str, max_stairs: int = 4) -> str:
    """Drop excessive stair pseudo-labels; keep walls/doors/windows/partitions."""
    walls, doors, windows, stairs, parts, other = [], [], [], [], [], []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        cls = int(ln.split()[0])
        coords = [float(x) for x in ln.split()[1:]]
        if len(coords) < 6:
            continue
        xs, ys = coords[0::2], coords[1::2]
        area = max(xs) - min(xs)
        area *= max(ys) - min(ys)
        item = (area, ln)
        if cls == 0:
            walls.append(item)
        elif cls == 1:
            doors.append(item)
        elif cls == 2:
            windows.append(item)
        elif cls == 3:
            stairs.append(item)
        elif cls == 4:
            parts.append(item)
        else:
            other.append(item)

    stairs = sorted(stairs, key=lambda t: t[0], reverse=True)[:max_stairs]
    # Prefer larger stair candidates only
    stairs = [t for t in stairs if t[0] >= 0.002]

    out = [t[1] for t in walls + doors + windows + stairs + parts + other]
    return "\n".join(out) + ("\n" if out else "")


def synthesize_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_root: Path,
    stems: list[str],
    variations: int = 20,
    val_stems: list[str] | None = None,
    max_stairs: int = 4,
) -> dict:
    train_img = out_root / "images" / "train"
    train_lbl = out_root / "labels" / "train"
    val_img = out_root / "images" / "val"
    val_lbl = out_root / "labels" / "val"
    for d in (train_img, train_lbl, val_img, val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    val_set = set(val_stems or [])
    train_set = [s for s in stems if s not in val_set]
    stats = {"train_images": 0, "val_images": 0, "skipped": 0}

    # Val = clean originals held out (proxy until real phone photos exist)
    for stem in sorted(val_set):
        src_img = None
        for ext in IMAGE_EXTS:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                src_img = cand
                break
        src_lbl = labels_dir / f"{stem}.txt"
        if not src_img or not src_lbl.exists():
            stats["skipped"] += 1
            continue
        shutil.copy2(src_img, val_img / src_img.name)
        (val_lbl / f"{stem}.txt").write_text(
            refine_label_text(src_lbl.read_text(encoding="utf-8"), max_stairs=max_stairs),
            encoding="utf-8",
        )
        stats["val_images"] += 1

    # Train = dirty variants of remaining curated plans
    for stem in train_set:
        src_img = None
        for ext in IMAGE_EXTS:
            cand = images_dir / f"{stem}{ext}"
            if cand.exists():
                src_img = cand
                break
        src_lbl = labels_dir / f"{stem}.txt"
        if not src_img or not src_lbl.exists():
            stats["skipped"] += 1
            continue
        img = cv2.imread(str(src_img), cv2.IMREAD_COLOR)
        if img is None:
            stats["skipped"] += 1
            continue
        label = refine_label_text(src_lbl.read_text(encoding="utf-8"), max_stairs=max_stairs)
        if not label.strip():
            stats["skipped"] += 1
            continue
        for i in range(variations):
            dirty = create_garbage_scan(img, seed=hash((stem, i)) % (2**32))
            out_name = f"{stem}_garb_{i:03d}.jpg"
            cv2.imwrite(str(train_img / out_name), dirty, [cv2.IMWRITE_JPEG_QUALITY, 85])
            (train_lbl / f"{stem}_garb_{i:03d}.txt").write_text(label, encoding="utf-8")
            stats["train_images"] += 1

    yaml_path = out_root / "data.yaml"
    # Project-relative path (cwd = repo root on host and under Docker /work).
    try:
        path_line = str(out_root.resolve().relative_to(Path(__file__).resolve().parents[1])).replace(
            "\\", "/"
        )
    except ValueError:
        path_line = str(out_root.resolve())
    yaml_path.write_text(
        f"path: {path_line}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 5\n"
        "names: ['wall', 'door', 'window', 'stair', 'partition']\n",
        encoding="utf-8",
    )
    stats["data_yaml"] = str(yaml_path)
    stats["train_plans"] = len(train_set)
    stats["val_plans"] = len(val_set)
    return stats


def pick_curated_stems(labels_dir: Path, n: int = 25) -> list[str]:
    rows = []
    for p in labels_dir.glob("*.txt"):
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        from collections import Counter

        c = Counter(int(ln.split()[0]) for ln in lines)
        # Prefer floorplan-like: several walls + some openings; not facade-only
        score = c[0] * 2 + c[1] * 3 + c[2] * 3 - max(0, c[3] - 8)
        rows.append((score, c[0], c[1] + c[2], p.stem))
    rows.sort(reverse=True)
    # Always include plan_007 if present
    stems = []
    if (labels_dir / "plan_007.txt").exists():
        stems.append("plan_007")
    for _, walls, opens, stem in rows:
        if stem in stems:
            continue
        if walls < 5:
            continue
        stems.append(stem)
        if len(stems) >= n:
            break
    return stems[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("datasets/user_clean/images"))
    parser.add_argument("--labels", type=Path, default=Path("datasets/user_clean/labels"))
    parser.add_argument("--out", type=Path, default=Path("datasets/floorplans_dataset"))
    parser.add_argument("--plans", type=int, default=25)
    parser.add_argument("--variations", type=int, default=20)
    parser.add_argument("--val-plans", type=int, default=5)
    parser.add_argument("--max-stairs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    stems = pick_curated_stems(args.labels, n=args.plans)
    if not stems:
        print("No curated plans found")
        return 1
    # Hold out last N for clean val (except keep plan_007 in train if possible)
    val_count = min(args.val_plans, max(1, len(stems) // 5))
    candidates = [s for s in stems if s != "plan_007"]
    random.shuffle(candidates)
    val_stems = candidates[:val_count]
    print("curated", len(stems), "val", val_stems)

    stats = synthesize_dataset(
        args.images,
        args.labels,
        args.out,
        stems=stems,
        variations=args.variations,
        val_stems=val_stems,
        max_stairs=args.max_stairs,
    )
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
