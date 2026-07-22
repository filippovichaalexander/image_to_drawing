"""Stage 2: geometry + text recognition (stub contours / optional YOLO+OCR)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def extract_geometry(cleaned_images: list[Path], ai_mode: str = "stub") -> dict[str, Any]:
    if ai_mode == "full":
        try:
            return _extract_full(cleaned_images)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI_MODE=full failed (%s); falling back to stub", exc)
    return _extract_stub(cleaned_images, ai_mode=ai_mode)


def _extract_full(cleaned_images: list[Path]) -> dict[str, Any]:
    """
    YOLOv8-seg + PaddleOCR when packages/weights are present.
    Raises ImportError/RuntimeError to trigger stub fallback.
    """
    from ultralytics import YOLO  # type: ignore
    from paddleocr import PaddleOCR  # type: ignore

    model_path = Path(__file__).resolve().parents[2] / "models" / "walls_seg.pt"
    if not model_path.exists():
        # Official nano-seg as bootstrap; fine-tune later on architectural dataset
        model_path = Path("yolov8n-seg.pt")

    model = YOLO(str(model_path))
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    walls: list[list[list[float]]] = []
    dimensions: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []

    for path in cleaned_images:
        results = model.predict(str(path), verbose=False)
        h = w = 1
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            h, w = img.shape[:2]
        scale_x, scale_y = 12000 / max(w, 1), 8000 / max(h, 1)

        for r in results:
            if r.masks is None:
                continue
            for poly in r.masks.xy:
                pts = [[float(x * scale_x), float((h - y) * scale_y)] for x, y in poly]
                if len(pts) >= 3:
                    pts.append(pts[0])
                    walls.append(pts)

        ocr_result = ocr.ocr(str(path), cls=True)
        for block in ocr_result or []:
            for line in block or []:
                box, (text, conf) = line
                if conf < 0.5:
                    continue
                text = (text or "").strip()
                digits = "".join(ch for ch in text if ch.isdigit())
                if digits and len(digits) >= 3:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    cx = float(sum(xs) / len(xs) * scale_x)
                    cy = float((h - sum(ys) / len(ys)) * scale_y)
                    dimensions.append(
                        {
                            "p1": [cx - float(digits) / 2, cy],
                            "p2": [cx + float(digits) / 2, cy],
                            "value_mm": int(digits),
                            "ocr": text,
                        }
                    )
                if text.upper().startswith("ОСЬ") or text.upper().startswith("AXIS"):
                    axes.append({"name": text.split()[-1], "x": 0, "y0": 0, "y1": 8000, "ocr": text})

    if not walls:
        raise RuntimeError("YOLO produced no wall masks")

    return {
        "mode": "full",
        "walls": walls,
        "openings": [],
        "axes": axes,
        "dimensions": dimensions,
        "units": "mm",
        "source_count": len(cleaned_images),
    }


def _extract_stub(cleaned_images: list[Path], ai_mode: str = "stub") -> dict[str, Any]:
    walls: list[list[list[float]]] = []
    axes: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []

    if not cleaned_images:
        walls.append([[0, 0], [12000, 0], [12000, 8000], [0, 8000], [0, 0]])
        dimensions.append({"p1": [0, 0], "p2": [12000, 0], "value_mm": 12000})
        axes.extend(
            [
                {"name": "1", "x": 0, "y0": -1500, "y1": 9500},
                {"name": "2", "x": 12000, "y0": -1500, "y1": 9500},
                {"name": "А", "y": 0, "x0": -1500, "x1": 13500},
                {"name": "Б", "y": 8000, "x0": -1500, "x1": 13500},
            ]
        )
        return {
            "mode": ai_mode,
            "walls": walls,
            "openings": [],
            "axes": axes,
            "dimensions": dimensions,
            "units": "mm",
        }

    for path in cleaned_images:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        work = cv2.bitwise_not(img) if np.mean(img) > 127 else img
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        scale_x, scale_y = 12000 / max(w, 1), 8000 / max(h, 1)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
        for cnt in contours:
            if cv2.contourArea(cnt) < (w * h) * 0.002:
                continue
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            poly = [[float(p[0][0] * scale_x), float((h - p[0][1]) * scale_y)] for p in approx]
            if len(poly) >= 3:
                poly.append(poly[0])
                walls.append(poly)

    if not walls:
        walls.append([[0, 0], [12000, 0], [12000, 8000], [0, 8000], [0, 0]])

    xs = [p[0] for wall in walls for p in wall]
    ys = [p[1] for wall in walls for p in wall]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dimensions.append({"p1": [min_x, min_y], "p2": [max_x, min_y], "value_mm": round(max_x - min_x)})
    dimensions.append({"p1": [min_x, min_y], "p2": [min_x, max_y], "value_mm": round(max_y - min_y)})
    axes.extend(
        [
            {"name": "1", "x": min_x, "y0": min_y - 1500, "y1": max_y + 1500},
            {"name": "2", "x": max_x, "y0": min_y - 1500, "y1": max_y + 1500},
            {"name": "А", "y": min_y, "x0": min_x - 1500, "x1": max_x + 1500},
            {"name": "Б", "y": max_y, "x0": min_x - 1500, "x1": max_x + 1500},
        ]
    )
    return {
        "mode": ai_mode,
        "walls": walls,
        "openings": [],
        "axes": axes,
        "dimensions": dimensions,
        "units": "mm",
        "source_count": len(cleaned_images),
    }
