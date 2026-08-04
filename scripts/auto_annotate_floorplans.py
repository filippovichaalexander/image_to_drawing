#!/usr/bin/env python3
"""Auto-annotate clean architectural floorplans for YOLO-seg.

Produces labels:
  0 wall | 1 door | 2 window | 3 stair | 4 partition
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

CLASS = {"wall": 0, "door": 1, "window": 2, "stair": 3, "partition": 4}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _poly_to_yolo(pts: np.ndarray, w: int, h: int) -> str | None:
    if pts is None or len(pts) < 3 or w < 1 or h < 1:
        return None
    flat = pts.reshape(-1, 2).astype(np.float64)
    coords: list[str] = []
    for x, y in flat:
        nx = min(1.0, max(0.0, float(x) / w))
        ny = min(1.0, max(0.0, float(y) / h))
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    if len(coords) < 6:
        return None
    return " ".join(coords)


def _simplify(contour: np.ndarray, eps_frac: float = 0.002) -> np.ndarray:
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, max(eps_frac * peri, 1.0), True)


def extract_wall_polygons(gray: np.ndarray, min_area_frac: float = 0.00015, max_area_frac: float = 0.35) -> list[np.ndarray]:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(2, int(round(min(h, w) * 0.002)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    thick = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thick, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = float(h * w)
    walls: list[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_frac * area_img or area > max_area_frac * area_img:
            continue
        approx = _simplify(cnt, 0.0015)
        if len(approx) >= 3:
            walls.append(approx)
    return walls


def extract_stair_candidates(gray: np.ndarray) -> list[np.ndarray]:
    h, w = gray.shape[:2]
    edges = cv2.Canny(gray, 60, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=max(12, int(min(h, w) * 0.02)), maxLineGap=4)
    if lines is None:
        return []
    segs = []
    raw = np.array(lines).reshape(-1, 4)
    for x1, y1, x2, y2 in raw:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        length = math.hypot(x2 - x1, y2 - y1)
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
        ortho = min(ang, abs(90 - ang), abs(180 - ang))
        if ortho > 12:
            continue
        segs.append((x1, y1, x2, y2, length))
    if len(segs) < 6:
        return []
    cell = max(20, int(min(h, w) * 0.04))
    buckets: dict[tuple[int, int], list] = {}
    for x1, y1, x2, y2, length in segs:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        key = (cx // cell, cy // cell)
        buckets.setdefault(key, []).append((x1, y1, x2, y2, length))
    stairs: list[np.ndarray] = []
    for items in buckets.values():
        if len(items) < 6:
            continue
        xs = [v for it in items for v in (it[0], it[2])]
        ys = [v for it in items for v in (it[1], it[3])]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        bw, bh = x1 - x0, y1 - y0
        if bw < 10 or bh < 10:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 4.5 or bw * bh < 0.0004 * h * w:
            continue
        poly = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)
        stairs.append(poly)
    return stairs


def yolo_assist_polygons(image_path: Path, model_path: Path | None, conf: float = 0.25, imgsz: int = 640) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {"door": [], "window": [], "stair": [], "wall": [], "partition": []}
    if model_path is None or not model_path.exists():
        return out
    try:
        from ultralytics import YOLO
    except ImportError:
        return out
    model = YOLO(str(model_path))
    results = model.predict(str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    names = model.names or {}
    for result in results:
        if result.masks is None or result.boxes is None:
            continue
        classes = result.boxes.cls.tolist()
        for i, xy in enumerate(result.masks.xy):
            if xy is None or len(xy) < 3:
                continue
            cls_id = int(classes[i]) if i < len(classes) else -1
            name = str(names.get(cls_id, "")).lower()
            if name not in out:
                continue
            pts = np.array(xy, dtype=np.float32).reshape(-1, 1, 2)
            out[name].append(pts.astype(np.int32))
    return out


def annotate_image(image_path: Path, model_path: Path | None = None, conf: float = 0.25) -> list[str]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines: list[str] = []
    walls = extract_wall_polygons(gray)
    for poly in walls:
        body = _poly_to_yolo(poly, w, h)
        if body:
            lines.append(f"{CLASS['wall']} {body}")
    assist = yolo_assist_polygons(image_path, model_path, conf=conf)
    if len(walls) < 2:
        for poly in assist.get("wall") or []:
            body = _poly_to_yolo(poly, w, h)
            if body:
                lines.append(f"{CLASS['wall']} {body}")
    for key in ("door", "window", "partition"):
        for poly in assist.get(key) or []:
            body = _poly_to_yolo(poly, w, h)
            if body:
                lines.append(f"{CLASS[key]} {body}")
    stairs = assist.get("stair") or extract_stair_candidates(gray)
    for poly in stairs:
        body = _poly_to_yolo(poly, w, h)
        if body:
            lines.append(f"{CLASS['stair']} {body}")
    return lines


def batch_annotate(images_dir: Path, labels_dir: Path, model_path: Path | None, conf: float = 0.25, overwrite: bool = False) -> dict:
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    stats = {"images": len(images), "written": 0, "skipped": 0, "failed": 0, "empty": 0}
    for img_path in images:
        out = labels_dir / f"{img_path.stem}.txt"
        if out.exists() and not overwrite:
            head = out.read_bytes()[:4]
            if head[:2] != b"\xff\xd8" and head != b"\x89PNG":
                stats["skipped"] += 1
                continue
        try:
            lines = annotate_image(img_path, model_path=model_path, conf=conf)
            out.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
            stats["written"] += 1
            if not lines:
                stats["empty"] += 1
        except Exception as exc:
            print(f"FAIL {img_path.name}: {exc}")
            stats["failed"] += 1
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/floorplan-seg.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single", type=Path, default=None)
    args = parser.parse_args()
    model = args.model if args.model.exists() else None
    if args.single:
        args.labels.mkdir(parents=True, exist_ok=True)
        lines = annotate_image(args.single, model_path=model, conf=args.conf)
        out = args.labels / f"{args.single.stem}.txt"
        out.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
        print({"image": str(args.single), "objects": len(lines), "label": str(out)})
        return 0 if lines else 1
    if args.images is None:
        parser.error("--images is required unless --single is set")
    stats = batch_annotate(args.images, args.labels, model, args.conf, args.overwrite)
    print(stats)
    return 0 if stats["written"] or stats["skipped"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
