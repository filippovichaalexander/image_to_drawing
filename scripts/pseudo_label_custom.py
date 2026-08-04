#!/usr/bin/env python3
"""Pseudo-label floorplan images with an existing YOLO-seg checkpoint.

Use when custom drawings have no CVAT labels yet. Quality is weaker than
manual labels, but enables a first fine-tune pass on your drawing style.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def mask_to_yolo_line(cls_id: int, xy: list, w: int, h: int) -> str | None:
    if xy is None or len(xy) < 3 or w < 1 or h < 1:
        return None
    coords: list[str] = []
    for x, y in xy:
        nx = min(1.0, max(0.0, float(x) / w))
        ny = min(1.0, max(0.0, float(y) / h))
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")
    if len(coords) < 6:
        return None
    return f"{cls_id} " + " ".join(coords)


def pseudo_label(
    images_dir: Path,
    labels_dir: Path,
    model_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict:
    from ultralytics import YOLO
    import cv2

    model = YOLO(str(model_path))
    labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    labeled = 0
    empty = 0
    instances = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        results = model.predict(
            str(img_path),
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )
        lines: list[str] = []
        for result in results:
            if result.masks is None or result.boxes is None:
                continue
            classes = result.boxes.cls.tolist()
            for i, poly in enumerate(result.masks.xy):
                cls_id = int(classes[i]) if i < len(classes) else 0
                line = mask_to_yolo_line(cls_id, poly, w, h)
                if line:
                    lines.append(line)
                    instances += 1
        out = labels_dir / f"{img_path.stem}.txt"
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            labeled += 1
        else:
            empty += 1
    return {
        "images": len(images),
        "labeled": labeled,
        "empty": empty,
        "instances": instances,
        "labels_dir": str(labels_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/floorplan-seg.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    stats = pseudo_label(args.images, args.labels, args.model, args.conf, args.imgsz)
    print(stats)
    return 0 if stats["labeled"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
