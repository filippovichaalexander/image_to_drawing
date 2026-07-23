"""Stage 2: geometry + text recognition (fast OpenCV stub / optional YOLO+OCR)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Process at most this long side — keeps CPU work light; coords scaled to mm.
MAX_PROC_SIDE = 1600
# Assumed real-world span when no OCR / user scale is available (plan metres → mm).
DEFAULT_SPAN_MM = 12000.0


def extract_geometry(cleaned_images: list[Path], ai_mode: str = "stub") -> dict[str, Any]:
    if ai_mode == "full":
        try:
            return _extract_full(cleaned_images)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI_MODE=full failed (%s); falling back to stub", exc)
    return _extract_stub(cleaned_images, ai_mode=ai_mode)


def _read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _downscale(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (small_image, scale_to_original) where coords_orig = coords_small * scale."""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_PROC_SIDE:
        return img, 1.0
    factor = MAX_PROC_SIDE / long_side
    small = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return small, 1.0 / factor


def _extract_full(cleaned_images: list[Path]) -> dict[str, Any]:
    """
    YOLOv8-seg + PaddleOCR when packages/weights are present.
    Raises ImportError/RuntimeError to trigger stub fallback.
    """
    from ultralytics import YOLO  # type: ignore
    from paddleocr import PaddleOCR  # type: ignore

    model_path = Path(__file__).resolve().parents[2] / "models" / "walls_seg.pt"
    if not model_path.exists():
        model_path = Path("yolov8n-seg.pt")

    model = YOLO(str(model_path))
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    walls: list[list[list[float]]] = []
    openings: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    sheets: list[dict[str, Any]] = []

    for sheet_idx, path in enumerate(cleaned_images):
        results = model.predict(str(path), verbose=False)
        h = w = 1
        img = _read_gray(path)
        if img is not None:
            h, w = img.shape[:2]
        scale_x, scale_y = DEFAULT_SPAN_MM / max(w, 1), (DEFAULT_SPAN_MM * 0.67) / max(h, 1)

        sheet_walls: list[list[list[float]]] = []
        for r in results:
            if r.masks is None:
                continue
            for poly in r.masks.xy:
                pts = [[float(x * scale_x), float((h - y) * scale_y)] for x, y in poly]
                if len(pts) >= 3:
                    pts.append(pts[0])
                    sheet_walls.append(pts)
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
                            "sheet": sheet_idx,
                        }
                    )
                upper = text.upper()
                if upper.startswith("ОСЬ") or upper.startswith("AXIS"):
                    axes.append(
                        {
                            "name": text.split()[-1],
                            "x": 0,
                            "y0": 0,
                            "y1": 8000,
                            "ocr": text,
                            "sheet": sheet_idx,
                        }
                    )

        sheets.append({"index": sheet_idx, "source": path.name, "wall_count": len(sheet_walls)})

    if not walls:
        raise RuntimeError("YOLO produced no wall masks")

    return {
        "mode": "full",
        "walls": walls,
        "partitions": [],
        "openings": openings,
        "axes": axes,
        "dimensions": dimensions,
        "sheets": sheets,
        "units": "mm",
        "source_count": len(cleaned_images),
        "scale_confidence": 0.55,
        "needs_scale": True,
    }


def _ink_mask(img: np.ndarray) -> np.ndarray:
    """Binary ink=255 on black background, robust to white-on-black cleans."""
    work = cv2.bitwise_not(img) if float(np.mean(img)) > 127 else img
    _, binary = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _hough_segments(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    h, w = mask.shape[:2]
    min_len = max(30, int(0.08 * min(h, w)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(40, min_len // 2),
        minLineLength=min_len,
        maxLineGap=8,
    )
    if lines is None:
        return []
    segs: list[tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        # Prefer near-orthogonal segments (architectural drawings)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx < 4 and dy < 4:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle <= 12 or angle >= 78:
            segs.append((int(x1), int(y1), int(x2), int(y2)))
    return segs[:200]


def _merge_collinear(
    segs: list[tuple[int, int, int, int]],
    tol: int = 12,
) -> list[tuple[int, int, int, int]]:
    """Cheap merge of near-collinear axis-aligned segments (O(n²), n≤200)."""
    horiz: list[list[int]] = []  # y, x0, x1
    vert: list[list[int]] = []  # x, y0, y1
    for x1, y1, x2, y2 in segs:
        if abs(y2 - y1) <= abs(x2 - x1):
            y = (y1 + y2) // 2
            xa, xb = sorted((x1, x2))
            horiz.append([y, xa, xb])
        else:
            x = (x1 + x2) // 2
            ya, yb = sorted((y1, y2))
            vert.append([x, ya, yb])

    def merge_1d(items: list[list[int]], axis_idx: int) -> list[list[int]]:
        if not items:
            return []
        items = sorted(items, key=lambda t: (t[0], t[1]))
        out: list[list[int]] = [items[0][:]]
        for cur in items[1:]:
            prev = out[-1]
            if abs(cur[0] - prev[0]) <= tol and cur[1] <= prev[2] + tol:
                prev[2] = max(prev[2], cur[2])
                prev[0] = (prev[0] + cur[0]) // 2
            else:
                out.append(cur[:])
        return out

    mh = merge_1d(horiz, 0)
    mv = merge_1d(vert, 0)
    merged: list[tuple[int, int, int, int]] = []
    for y, x0, x1 in mh:
        if x1 - x0 >= tol:
            merged.append((x0, y, x1, y))
    for x, y0, y1 in mv:
        if y1 - y0 >= tol:
            merged.append((x, y0, x, y1))
    return merged


def _outer_contour_poly(mask: np.ndarray) -> list[list[float]] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    h, w = mask.shape[:2]
    if cv2.contourArea(cnt) < 0.01 * w * h:
        return None
    approx = cv2.approxPolyDP(cnt, 0.008 * cv2.arcLength(cnt, True), True)
    if len(approx) < 3:
        return None
    poly = [[float(p[0][0]), float(p[0][1])] for p in approx]
    poly.append(poly[0])
    return poly


def _openings_from_gaps(
    mask: np.ndarray,
    segs: list[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    """
    Heuristic openings: long wall segments with a mid-gap → window/door slot.
    Fast morphology on downscaled mask; no neural net.
    """
    h, w = mask.shape[:2]
    openings: list[dict[str, Any]] = []
    # Dilate ink slightly, then look for rectangular voids near long edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thick = cv2.dilate(mask, kernel, iterations=1)
    inv = cv2.bitwise_not(thick)
    # Ignore border frame
    margin = max(8, int(0.03 * min(h, w)))
    inv[:margin, :] = 0
    inv[-margin:, :] = 0
    inv[:, :margin] = 0
    inv[:, -margin:] = 0
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 0.0008 * w * h or area > 0.08 * w * h:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect < 1.4 or aspect > 12:
            continue
        # Prefer slots adjacent to a Hough wall
        cx, cy = x + bw / 2, y + bh / 2
        near_wall = False
        for x1, y1, x2, y2 in segs:
            dist = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / (
                max(np.hypot(x2 - x1, y2 - y1), 1.0)
            )
            if dist < 18:
                near_wall = True
                break
        if not near_wall:
            continue
        kind = "door" if min(bw, bh) > 0.04 * min(h, w) else "window"
        openings.append(
            {
                "type": kind,
                "p1": [float(x), float(y + bh)],
                "p2": [float(x + bw), float(y)],
                "width_px": float(bw),
                "height_px": float(bh),
            }
        )
        if len(openings) >= 24:
            break
    return openings


def _px_to_mm(
    x: float,
    y: float,
    h: int,
    scale_x: float,
    scale_y: float,
) -> list[float]:
    # Image origin top-left → CAD origin bottom-left
    return [float(x * scale_x), float((h - y) * scale_y)]


def _sheet_geometry(path: Path, sheet_idx: int) -> dict[str, Any]:
    img = _read_gray(path)
    if img is None:
        return {"walls": [], "partitions": [], "openings": [], "axes": [], "dimensions": [], "ok": False}

    small, up = _downscale(img)
    mask = _ink_mask(small)
    sh, sw = small.shape[:2]
    segs = _merge_collinear(_hough_segments(mask))
    outer = _outer_contour_poly(mask)
    openings_px = _openings_from_gaps(mask, segs)

    # Map processing pixels → assumed mm span of the long side
    long_px = max(sw, sh)
    mm_per_px = DEFAULT_SPAN_MM / max(long_px, 1)
    scale_x = scale_y = mm_per_px

    walls: list[list[list[float]]] = []
    partitions: list[list[list[float]]] = []

    if outer:
        walls.append([_px_to_mm(p[0], p[1], sh, scale_x, scale_y) for p in outer])

    # Build thin wall polylines from merged Hough segments (as 2-point walls)
    wall_lines: list[list[list[float]]] = []
    for x1, y1, x2, y2 in segs:
        length = np.hypot(x2 - x1, y2 - y1)
        p1 = _px_to_mm(x1, y1, sh, scale_x, scale_y)
        p2 = _px_to_mm(x2, y2, sh, scale_x, scale_y)
        if length > 0.25 * min(sw, sh):
            wall_lines.append([p1, p2])
        elif length > 0.08 * min(sw, sh):
            partitions.append([p1, p2])

    # If no outer contour, synthesize a bounding box from long lines
    if not walls and wall_lines:
        xs = [p[0] for line in wall_lines for p in line]
        ys = [p[1] for line in wall_lines for p in line]
        walls.append(
            [
                [min(xs), min(ys)],
                [max(xs), min(ys)],
                [max(xs), max(ys)],
                [min(xs), max(ys)],
                [min(xs), min(ys)],
            ]
        )
    walls.extend(wall_lines)

    openings: list[dict[str, Any]] = []
    for op in openings_px:
        p1 = _px_to_mm(op["p1"][0], op["p1"][1], sh, scale_x, scale_y)
        p2 = _px_to_mm(op["p2"][0], op["p2"][1], sh, scale_x, scale_y)
        openings.append(
            {
                "type": op["type"],
                "p1": p1,
                "p2": p2,
                "sheet": sheet_idx,
                "from": "opencv_gap",
            }
        )

    # Bounds for axes / overall dims
    pts = [p for wall in walls for p in wall] or [[0.0, 0.0], [DEFAULT_SPAN_MM, DEFAULT_SPAN_MM * 0.67]]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    dimensions = [
        {
            "p1": [min_x, min_y],
            "p2": [max_x, min_y],
            "value_mm": round(max_x - min_x),
            "sheet": sheet_idx,
            "role": "overall_x",
        },
        {
            "p1": [min_x, min_y],
            "p2": [min_x, max_y],
            "value_mm": round(max_y - min_y),
            "sheet": sheet_idx,
            "role": "overall_y",
        },
    ]
    axes = [
        {"name": "1", "x": min_x, "y0": min_y - 1500, "y1": max_y + 1500, "sheet": sheet_idx},
        {"name": "2", "x": max_x, "y0": min_y - 1500, "y1": max_y + 1500, "sheet": sheet_idx},
        {"name": "А", "y": min_y, "x0": min_x - 1500, "x1": max_x + 1500, "sheet": sheet_idx},
        {"name": "Б", "y": max_y, "x0": min_x - 1500, "x1": max_x + 1500, "sheet": sheet_idx},
    ]

    # Confidence: more orthogonal structure → higher; still ask for scale without OCR
    score = 0.25
    if outer:
        score += 0.2
    if len(segs) >= 4:
        score += 0.15
    if openings:
        score += 0.1
    score = min(score, 0.65)

    return {
        "ok": True,
        "walls": walls,
        "partitions": partitions,
        "openings": openings,
        "axes": axes,
        "dimensions": dimensions,
        "scale_confidence": score,
        "proc_size": [sw, sh],
        "upscale": up,
        "source": path.name,
        "index": sheet_idx,
    }


def _extract_stub(cleaned_images: list[Path], ai_mode: str = "stub") -> dict[str, Any]:
    if not cleaned_images:
        walls = [[[0, 0], [12000, 0], [12000, 8000], [0, 8000], [0, 0]]]
        return {
            "mode": ai_mode,
            "walls": walls,
            "partitions": [],
            "openings": [],
            "axes": [
                {"name": "1", "x": 0, "y0": -1500, "y1": 9500, "sheet": 0},
                {"name": "2", "x": 12000, "y0": -1500, "y1": 9500, "sheet": 0},
                {"name": "А", "y": 0, "x0": -1500, "x1": 13500, "sheet": 0},
                {"name": "Б", "y": 8000, "x0": -1500, "x1": 13500, "sheet": 0},
            ],
            "dimensions": [{"p1": [0, 0], "p2": [12000, 0], "value_mm": 12000, "sheet": 0}],
            "sheets": [],
            "units": "mm",
            "scale_confidence": 0.1,
            "needs_scale": True,
            "source_count": 0,
        }

    walls: list[list[list[float]]] = []
    partitions: list[list[list[float]]] = []
    openings: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    sheets: list[dict[str, Any]] = []
    confidences: list[float] = []

    for idx, path in enumerate(cleaned_images):
        sheet = _sheet_geometry(path, idx)
        sheets.append(
            {
                "index": idx,
                "source": sheet.get("source", path.name),
                "ok": sheet.get("ok", False),
                "proc_size": sheet.get("proc_size"),
            }
        )
        if not sheet.get("ok"):
            continue
        walls.extend(sheet["walls"])
        partitions.extend(sheet["partitions"])
        openings.extend(sheet["openings"])
        # Keep axes/dims from first successful sheet as primary; others tagged for correlate
        if idx == 0 or not axes:
            axes.extend(sheet["axes"])
            dimensions.extend(sheet["dimensions"])
        else:
            for a in sheet["axes"]:
                axes.append(a)
            for d in sheet["dimensions"]:
                dimensions.append(d)
        confidences.append(float(sheet.get("scale_confidence", 0.2)))

    if not walls:
        walls.append([[0, 0], [12000, 0], [12000, 8000], [0, 8000], [0, 0]])
        confidences.append(0.1)

    mean_conf = sum(confidences) / max(len(confidences), 1)
    return {
        "mode": ai_mode,
        "walls": walls,
        "partitions": partitions,
        "openings": openings,
        "axes": axes,
        "dimensions": dimensions,
        "sheets": sheets,
        "units": "mm",
        "source_count": len(cleaned_images),
        "scale_confidence": round(mean_conf, 3),
        "needs_scale": mean_conf < 0.5,
    }
