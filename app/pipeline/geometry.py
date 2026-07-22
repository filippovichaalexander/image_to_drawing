"""Stage 2: geometry extraction (stub / future YOLO+OCR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def extract_geometry(cleaned_images: list[Path], ai_mode: str = "stub") -> dict[str, Any]:
    """
    Return wall polylines, openings, axes and dimension candidates.

    stub: contour heuristics from binary image (no heavy models).
    full: reserved for YOLOv8-seg + PaddleOCR (local weights, not SaaS).
    """
    walls: list[list[list[float]]] = []
    axes: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []

    if not cleaned_images:
        # Synthetic rectangle so CAD export still demonstrable
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
        # Invert if background is white (typical binary scan)
        if np.mean(img) > 127:
            work = cv2.bitwise_not(img)
        else:
            work = img
        contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = img.shape[:2]
        scale_x = 12000 / max(w, 1)
        scale_y = 8000 / max(h, 1)
        # Keep a few largest contours as wall candidates
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

    # Bounding box dimensions for stamp/scale demo
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
