"""Lightweight dimension OCR — read size labels drawn on scans (mm)."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DIM_RE = re.compile(r"(?<!\d)(\d{3,5})(?!\d)")
_AREA_RE = re.compile(r"^\d+[.,]\d{1,2}$")  # room areas like 16,71 m² — not mm
_MIN_MM = 400
_MAX_MM = 120_000


@lru_cache(maxsize=1)
def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        return RapidOCR()
    except Exception as exc:  # noqa: BLE001
        logger.warning("RapidOCR unavailable (%s); size labels will not be read", exc)
        return None


def _read_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def extract_dimension_labels(image_path: Path, max_side: int = 2000) -> list[dict[str, Any]]:
    """Return dimension candidates [{value_mm, cx, cy, conf, text, is_area}, ...]."""
    eng = _engine()
    if eng is None:
        return []

    img = _read_bgr(image_path)
    if img is None:
        return []

    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    try:
        result, _ = eng(img)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OCR failed on %s: %s", image_path.name, exc)
        return []

    out: list[dict[str, Any]] = []
    for item in result or []:
        if not item or len(item) < 3:
            continue
        box, text, score = item[0], str(item[1]), float(item[2])
        if score < 0.45:
            continue
        raw = (text or "").strip()
        is_area = bool(_AREA_RE.match(raw.replace(" ", "")))
        cleaned = raw.replace(" ", "").replace(",", "").replace(".", "")
        for match in _DIM_RE.finditer(cleaned):
            value = int(match.group(1))
            if value < _MIN_MM or value > _MAX_MM:
                continue
            # Area labels "16,71" become 1671 — drop those
            if is_area:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append(
                {
                    "value_mm": value,
                    "cx": float(sum(xs) / len(xs)),
                    "cy": float(sum(ys) / len(ys)),
                    "conf": score,
                    "text": raw,
                    "is_area": False,
                    "ocr_scale": scale,
                    "img_h": float(img.shape[0]),
                    "img_w": float(img.shape[1]),
                }
            )
    out.sort(key=lambda d: (-d["value_mm"], -d["conf"]))
    return out


def calibrate_span_mm(
    labels: list[dict[str, Any]],
    bbox_w_px: float,
    bbox_h_px: float,
) -> dict[str, Any] | None:
    """
    Map building pixel bbox → real mm.

    Prefer the largest overall chain (e.g. 16000) for the long side;
    next distinct large value or aspect ratio for the short side.
    """
    if not labels or bbox_w_px < 8 or bbox_h_px < 8:
        return None

    values: list[int] = []
    seen: set[int] = set()
    for lab in labels:
        if lab.get("is_area"):
            continue
        v = int(lab["value_mm"])
        key = int(round(v / 50.0) * 50)
        if key in seen:
            continue
        seen.add(key)
        values.append(v)
        if len(values) >= 8:
            break

    if not values:
        return None

    long_px = max(bbox_w_px, bbox_h_px)
    short_px = min(bbox_w_px, bbox_h_px)
    aspect = short_px / long_px

    # Overall dims are typically the largest 1–2 values and >> room spans
    long_mm = float(values[0])
    short_mm = None
    for v in values[1:]:
        # Accept second overall if it roughly matches bbox aspect (±35%)
        expected = long_mm * aspect
        if 0.55 * expected <= v <= 1.45 * expected:
            short_mm = float(v)
            break
        # Or clearly a large chain (>= 40% of long)
        if v >= 0.4 * long_mm:
            short_mm = float(v)
            break
    if short_mm is None:
        short_mm = long_mm * aspect

    if short_mm > long_mm:
        short_mm, long_mm = long_mm, short_mm

    if bbox_w_px >= bbox_h_px:
        span_x, span_y = long_mm, short_mm
    else:
        span_x, span_y = short_mm, long_mm

    return {
        "span_x_mm": span_x,
        "span_y_mm": span_y,
        "from_ocr": True,
        "ocr_values": values[:8],
        "confidence": min(0.6 + 0.05 * len(values), 0.95),
    }
