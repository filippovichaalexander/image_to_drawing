"""Lightweight dimension OCR — read size labels and calibrate scale from chains."""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_DIM_RE = re.compile(r"(?<!\d)(\d{3,5})(?!\d)")
_AREA_RE = re.compile(r"^\d+[.,]\d{1,2}$")
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
                    "box": box,
                }
            )
    out.sort(key=lambda d: (-d["value_mm"], -d["conf"]))
    return out


def _seg_len(seg: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = seg
    return float(math.hypot(x2 - x1, y2 - y1))


def _orient(seg: tuple[float, float, float, float]) -> str:
    x1, y1, x2, y2 = seg
    return "h" if abs(x2 - x1) >= abs(y2 - y1) else "v"


def _point_to_seg_dist(px: float, py: float, seg: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    qx, qy = x1 + t * dx, y1 + t * dy
    return math.hypot(px - qx, py - qy)


def _project_span_on_seg(px: float, py: float, seg: tuple[float, float, float, float]) -> float | None:
    """If point projects onto segment interior, return full segment length (chain proxy)."""
    x1, y1, x2, y2 = seg
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 8:
        return None
    t = ((px - x1) * dx + (py - y1) * dy) / (L * L)
    if t < -0.05 or t > 1.05:
        return None
    return L


def match_label_to_chain(
    label: dict[str, Any],
    segments: list[tuple[float, float, float, float]],
    max_dist_px: float,
) -> dict[str, Any] | None:
    """
    Bind an OCR size to the nearest parallel wall/dim chain segment.
    Scale = value_mm / chain_length_px (full chain, not building bbox).
    """
    # Label coords are in OCR image space; caller must pass segments in same space
    cx, cy = float(label["cx"]), float(label["cy"])
    value = float(label["value_mm"])
    best = None
    best_score = 1e18
    for seg in segments:
        dist = _point_to_seg_dist(cx, cy, seg)
        if dist > max_dist_px:
            continue
        span = _project_span_on_seg(cx, cy, seg)
        if span is None:
            # Still allow nearby parallel segment (label beside chain)
            span = _seg_len(seg)
        if span < 12:
            continue
        # Prefer closer + longer chains (overall dims)
        score = dist + (80.0 / max(span, 1.0))
        if score < best_score:
            best_score = score
            mm_per_px = value / span
            best = {
                "value_mm": value,
                "chain_px": span,
                "mm_per_px": mm_per_px,
                "orient": _orient(seg),
                "dist_px": dist,
                "conf": float(label.get("conf") or 0.5),
                "text": label.get("text"),
                "seg": seg,
            }
    return best


def calibrate_from_chains(
    labels: list[dict[str, Any]],
    segments_px: list[tuple[float, float, float, float]],
    bbox_w_px: float,
    bbox_h_px: float,
    ocr_to_proc: float = 1.0,
) -> dict[str, Any] | None:
    """
    Compute mm/px from OCR labels matched to full dimension/wall chains.

    `segments_px` — in the same pixel space as geometry processing.
    `ocr_to_proc` — multiply OCR coords to reach process space
                    (proc = ocr_coord / ocr_scale * (1/up) … caller supplies factor).
    """
    usable = [l for l in labels if not l.get("is_area") and int(l["value_mm"]) >= 800]
    if not usable or not segments_px:
        return None

    # Remap label centers into process pixel space
    mapped: list[dict[str, Any]] = []
    for lab in usable:
        m = dict(lab)
        m["cx"] = float(lab["cx"]) * ocr_to_proc
        m["cy"] = float(lab["cy"]) * ocr_to_proc
        mapped.append(m)

    max_dist = max(28.0, 0.06 * max(bbox_w_px, bbox_h_px))
    matches: list[dict[str, Any]] = []
    for lab in mapped:
        hit = match_label_to_chain(lab, segments_px, max_dist_px=max_dist)
        if hit:
            matches.append(hit)

    if not matches:
        # Fallback: largest label ↔ longest segment of dominant orientation
        lab = max(mapped, key=lambda l: int(l["value_mm"]))
        longest = max(segments_px, key=_seg_len)
        span = _seg_len(longest)
        if span < 12:
            return None
        matches = [
            {
                "value_mm": float(lab["value_mm"]),
                "chain_px": span,
                "mm_per_px": float(lab["value_mm"]) / span,
                "orient": _orient(longest),
                "dist_px": 999.0,
                "conf": float(lab.get("conf") or 0.4) * 0.7,
                "text": lab.get("text"),
                "seg": longest,
                "fallback": True,
            }
        ]

    # Separate H/V scales; prefer median of high-value matches
    def pick(orient: str) -> float | None:
        cand = [m for m in matches if m["orient"] == orient]
        if not cand:
            return None
        cand.sort(key=lambda m: (-m["value_mm"], m["dist_px"]))
        # top few by value
        top = cand[: min(5, len(cand))]
        rates = sorted(m["mm_per_px"] for m in top)
        return float(rates[len(rates) // 2])

    sx = pick("h")
    sy = pick("v")
    if sx is None and sy is None:
        return None
    if sx is None:
        sx = sy
    if sy is None:
        sy = sx

    # Confidence: more chain matches + agreement between H/V
    ratio = min(sx, sy) / max(sx, sy)
    n = len(matches)
    mean_conf = sum(m["conf"] for m in matches) / n
    conf = 0.45 + 0.08 * min(n, 5) + 0.15 * mean_conf
    if ratio < 0.75:
        conf *= 0.75  # anisotropic disagreement
    conf = float(min(0.95, conf))

    span_x = sx * bbox_w_px
    span_y = sy * bbox_h_px
    values = sorted({int(m["value_mm"]) for m in matches}, reverse=True)

    return {
        "span_x_mm": span_x,
        "span_y_mm": span_y,
        "mm_per_px_x": sx,
        "mm_per_px_y": sy,
        "from_ocr": True,
        "from_chains": True,
        "ocr_values": values[:8],
        "matches": len(matches),
        "confidence": conf,
        "scale_source": "ocr_chain",
    }


def calibrate_span_mm(
    labels: list[dict[str, Any]],
    bbox_w_px: float,
    bbox_h_px: float,
    segments_px: list[tuple[float, float, float, float]] | None = None,
    ocr_to_proc: float = 1.0,
) -> dict[str, Any] | None:
    """
    Prefer chain-based calibration; legacy bbox mapping only as last resort
    and with reduced confidence (never silent 12000 mm).
    """
    if segments_px:
        chain = calibrate_from_chains(labels, segments_px, bbox_w_px, bbox_h_px, ocr_to_proc)
        if chain:
            return chain

    if not labels or bbox_w_px < 8 or bbox_h_px < 8:
        return None

    values: list[int] = []
    seen: set[int] = set()
    for lab in labels:
        if lab.get("is_area"):
            continue
        v = int(lab["value_mm"])
        if v < 800:
            continue
        key = int(round(v / 50.0) * 50)
        if key in seen:
            continue
        seen.add(key)
        values.append(v)
        if len(values) >= 6:
            break
    if not values:
        return None

    # Weak legacy: map largest OCR to long bbox side — mark low confidence
    long_px = max(bbox_w_px, bbox_h_px)
    short_px = min(bbox_w_px, bbox_h_px)
    long_mm = float(values[0])
    short_mm = float(values[1]) if len(values) > 1 else long_mm * (short_px / long_px)
    if short_mm > long_mm:
        short_mm, long_mm = long_mm, short_mm
    if bbox_w_px >= bbox_h_px:
        span_x, span_y = long_mm, short_mm
    else:
        span_x, span_y = short_mm, long_mm

    return {
        "span_x_mm": span_x,
        "span_y_mm": span_y,
        "mm_per_px_x": span_x / bbox_w_px,
        "mm_per_px_y": span_y / bbox_h_px,
        "from_ocr": True,
        "from_chains": False,
        "ocr_values": values[:8],
        "confidence": 0.42,  # forces needs_scale unless user confirms
        "scale_source": "ocr_bbox_weak",
        "warning": "OCR sizes not matched to dimension chains; scale is approximate",
    }
