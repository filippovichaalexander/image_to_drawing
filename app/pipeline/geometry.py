"""Stage 2: geometry + text recognition (fast OpenCV stub / optional YOLO+OCR)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_PROC_SIDE = 1800
DEFAULT_SPAN_MM = 12000.0


def extract_geometry(
    cleaned_images: list[Path],
    ai_mode: str = "stub",
    source_images: list[Path] | None = None,
) -> dict[str, Any]:
    """
    `cleaned_images` — binary denoise outputs (vectorization).
    `source_images` — original rasters for OCR (preferred; cleaned kills text).

    Primary path is OpenCV stub: OCR→dimension-chain scale, contour post-process,
    closedness checks. YOLO (AI_MODE=full) is a fallback when stub finds no walls.
    """
    stub = _extract_stub(cleaned_images, ai_mode=ai_mode, source_images=source_images)
    if stub.get("walls") and any(len(w) >= 2 for w in stub["walls"]):
        # Mark mode full if requested, but keep stub geometry/scale
        if ai_mode == "full":
            stub["mode"] = "full+stub"
        return stub

    if ai_mode == "full":
        try:
            return _extract_full(cleaned_images)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI_MODE=full failed (%s); using stub geometry", exc)
    return stub


def _read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _downscale(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_PROC_SIDE:
        return img, 1.0
    factor = MAX_PROC_SIDE / long_side
    small = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return small, 1.0 / factor


def _extract_full(cleaned_images: list[Path]) -> dict[str, Any]:
    from ultralytics import YOLO  # type: ignore
    from paddleocr import PaddleOCR  # type: ignore

    model_path = Path(__file__).resolve().parents[2] / "models" / "walls_seg.pt"
    if not model_path.exists():
        model_path = Path("yolov8n-seg.pt")
        logger.warning(
            "models/walls_seg.pt missing — using %s (run: python scripts/download_models.py)",
            model_path,
        )

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

        sheets.append({"index": sheet_idx, "source": path.name, "wall_count": len(sheet_walls)})

    if not walls:
        raise RuntimeError("YOLO produced no wall masks")

    from app.pipeline.postprocess import postprocess_walls

    walls, partitions, closedness = postprocess_walls(walls, [])
    scale_warning = (
        "Полный AI-режим: масштаб по умолчанию условный (12 000 мм). "
        "Укажите контрольный отрезок в мм."
    )
    return {
        "mode": "full",
        "walls": walls,
        "partitions": partitions,
        "openings": openings,
        "axes": axes,
        "dimensions": dimensions,
        "sheets": sheets,
        "units": "mm",
        "source_count": len(cleaned_images),
        "scale_confidence": 0.35,
        "scale_source": "provisional",
        "scale_warning": scale_warning,
        "closedness": closedness,
        "needs_scale": True,
    }


def _merge_h(items: list[tuple[int, int, int]], tol: int = 5) -> list[tuple[int, int, int]]:
    items = sorted(items, key=lambda t: (t[1], t[0]))
    out: list[tuple[int, int, int]] = []
    for x0, y, x1 in items:
        if out and abs(out[-1][1] - y) <= tol and x0 <= out[-1][2] + 10:
            px0, py, px1 = out[-1]
            out[-1] = (min(px0, x0), (py + y) // 2, max(px1, x1))
        else:
            out.append((x0, y, x1))
    return out


def _merge_v(items: list[tuple[int, int, int]], tol: int = 5) -> list[tuple[int, int, int]]:
    items = sorted(items, key=lambda t: (t[0], t[1]))
    out: list[tuple[int, int, int]] = []
    for x, y0, y1 in items:
        if out and abs(out[-1][0] - x) <= tol and y0 <= out[-1][2] + 10:
            px, py0, py1 = out[-1]
            out[-1] = ((px + x) // 2, min(py0, y0), max(py1, y1))
        else:
            out.append((x, y0, y1))
    return out


def _suppress_hatch_h(segs: list[tuple[int, int, int]], band: int = 18) -> list[tuple[int, int, int]]:
    """Drop stair/furniture hatching: dense parallel H lines → keep none (or envelope)."""
    if not segs:
        return []
    segs = sorted(segs, key=lambda t: t[1])
    clusters: list[list[tuple[int, int, int]]] = [[segs[0]]]
    for s in segs[1:]:
        if abs(s[1] - clusters[-1][-1][1]) <= band:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    out: list[tuple[int, int, int]] = []
    for cl in clusters:
        if len(cl) >= 5:
            # Hatch pack — skip interior; keep nothing (stairs/furniture)
            continue
        out.extend(cl)
    return out


def _suppress_hatch_v(segs: list[tuple[int, int, int]], band: int = 18) -> list[tuple[int, int, int]]:
    if not segs:
        return []
    segs = sorted(segs, key=lambda t: t[0])
    clusters: list[list[tuple[int, int, int]]] = [[segs[0]]]
    for s in segs[1:]:
        if abs(s[0] - clusters[-1][-1][0]) <= band:
            clusters[-1].append(s)
        else:
            clusters.append([s])
    out: list[tuple[int, int, int]] = []
    for cl in clusters:
        if len(cl) >= 5:
            continue
        out.extend(cl)
    return out


def _unique_grid_h(segs: list[tuple[int, int, int]], tol: int = 10, limit: int = 14) -> list[tuple[int, int, int]]:
    """One longest horizontal wall per Y-grid slot."""
    segs = sorted(segs, key=lambda t: -(t[2] - t[0]))
    picked: list[tuple[int, int, int]] = []
    for s in segs:
        if any(abs(s[1] - p[1]) <= tol for p in picked):
            continue
        picked.append(s)
        if len(picked) >= limit:
            break
    return sorted(picked, key=lambda t: t[1])


def _unique_grid_v(segs: list[tuple[int, int, int]], tol: int = 10, limit: int = 14) -> list[tuple[int, int, int]]:
    segs = sorted(segs, key=lambda t: -(t[2] - t[1]))
    picked: list[tuple[int, int, int]] = []
    for s in segs:
        if any(abs(s[0] - p[0]) <= tol for p in picked):
            continue
        picked.append(s)
        if len(picked) >= limit:
            break
    return sorted(picked, key=lambda t: t[0])


def _extract_ortho_walls(gray: np.ndarray) -> tuple[list[tuple[int, int, int, int]], tuple[float, float, float, float]]:
    """
    Structural orthogonal wall centerlines only.

    Filters stair hatching / furniture / dimension ticks that make the CAD
    look chaotic. Returns segments + building bbox in image pixels.
    """
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 70, 170)
    # Prefer thicker ink (walls) over thin dim ticks
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    thick = ((dist >= 1.0).astype(np.uint8) * 255)
    thick_edges = cv2.Canny(thick, 40, 120)
    edges = cv2.bitwise_or(edges, thick_edges)

    min_len = max(50, int(0.14 * min(h, w)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(55, min_len // 2),
        minLineLength=min_len,
        maxLineGap=8,
    )
    hs: list[tuple[int, int, int]] = []
    vs: list[tuple[int, int, int]] = []
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            ang = abs(float(np.degrees(np.arctan2(dy, dx))))
            if ang <= 8:
                hs.append((min(x1, x2), (y1 + y2) // 2, max(x1, x2)))
            elif ang >= 82:
                vs.append(((x1 + x2) // 2, min(y1, y2), max(y1, y2)))

    mh = _unique_grid_h(_suppress_hatch_h(_merge_h(hs)))
    mv = _unique_grid_v(_suppress_hatch_v(_merge_v(vs)))

    xs: list[float] = []
    ys: list[float] = []
    for x0, y, x1 in mh:
        xs.extend((x0, x1))
        ys.append(float(y))
    for x, y0, y1 in mv:
        xs.append(float(x))
        ys.extend((y0, y1))
    if len(xs) < 2:
        return [], (0.0, 0.0, float(w), float(h))

    # Tighter bbox from structural grid (not dimension rings)
    x0, x1 = float(np.percentile(xs, 8)), float(np.percentile(xs, 92))
    y0, y1 = float(np.percentile(ys, 8)), float(np.percentile(ys, 92))
    # Prefer extent of the longest H/V as building sides when available
    if mh:
        longest_h = max(mh, key=lambda t: t[2] - t[0])
        x0, x1 = float(longest_h[0]), float(longest_h[2])
    if mv:
        longest_v = max(mv, key=lambda t: t[2] - t[1])
        y0, y1 = float(longest_v[1]), float(longest_v[2])
    # Expand to cover other long walls
    for sx0, sy, sx1 in mh:
        if (sx1 - sx0) >= 0.55 * (x1 - x0):
            x0, x1 = min(x0, float(sx0)), max(x1, float(sx1))
            y0, y1 = min(y0, float(sy)), max(y1, float(sy))
    for sx, sy0, sy1 in mv:
        if (sy1 - sy0) >= 0.55 * (y1 - y0):
            y0, y1 = min(y0, float(sy0)), max(y1, float(sy1))
            x0, x1 = min(x0, float(sx)), max(x1, float(sx))

    pad = 0.005 * max(w, h)
    x0, y0 = max(0.0, x0 - pad), max(0.0, y0 - pad)
    x1, y1 = min(float(w), x1 + pad), min(float(h), y1 + pad)

    segs: list[tuple[int, int, int, int]] = []
    # Clip internal walls to bbox
    for sx0, sy, sx1 in mh:
        if y0 - 15 <= sy <= y1 + 15:
            a, b = max(int(x0), sx0), min(int(x1), sx1)
            if b - a >= min_len * 0.5:
                segs.append((a, int(sy), b, int(sy)))
    for sx, sy0, sy1 in mv:
        if x0 - 15 <= sx <= x1 + 15:
            a, b = max(int(y0), sy0), min(int(y1), sy1)
            if b - a >= min_len * 0.5:
                segs.append((int(sx), a, int(sx), b))

    # Outer closed rectangle (true building envelope)
    segs = [
        (int(x0), int(y0), int(x1), int(y0)),
        (int(x1), int(y0), int(x1), int(y1)),
        (int(x1), int(y1), int(x0), int(y1)),
        (int(x0), int(y1), int(x0), int(y0)),
        *segs,
    ]
    return segs, (x0, y0, x1, y1)


def _px_to_mm(x: float, y: float, h: int, scale_x: float, scale_y: float) -> list[float]:
    return [float(x * scale_x), float((h - y) * scale_y)]


def _classify_wall_len(length_px: float, min_side: float) -> str:
    if length_px >= 0.22 * min_side:
        return "wall"
    if length_px >= 0.08 * min_side:
        return "partition"
    return "skip"


def _sheet_geometry(
    path: Path,
    sheet_idx: int,
    ocr_path: Path | None = None,
) -> dict[str, Any]:
    img = _read_gray(path)
    if img is None:
        return {"walls": [], "partitions": [], "openings": [], "axes": [], "dimensions": [], "ok": False}

    # Prefer a less-destroyed source for edges when available
    ocr_src = ocr_path or path
    edge_img = _read_gray(ocr_src) if ocr_path else img
    if edge_img is None:
        edge_img = img

    small, up = _downscale(edge_img)
    sh, sw = small.shape[:2]
    segs_px, bbox = _extract_ortho_walls(small)
    x0, y0, x1, y1 = bbox
    bbox_w_px = max(x1 - x0, 1.0)
    bbox_h_px = max(y1 - y0, 1.0)

    from app.pipeline.ocr_dims import calibrate_span_mm, extract_dimension_labels
    from app.pipeline.postprocess import postprocess_walls

    labels = extract_dimension_labels(ocr_src)
    # OCR coords → process (`small`) space: full = ocr/ocr_scale, proc = full/up
    ocr_scale0 = float(labels[0]["ocr_scale"]) if labels else 1.0
    ocr_to_proc = 1.0 / max(ocr_scale0 * up, 1e-9)
    calib = calibrate_span_mm(
        labels,
        bbox_w_px,
        bbox_h_px,
        segments_px=segs_px,
        ocr_to_proc=ocr_to_proc,
    )
    scale_warning: str | None = None
    if calib:
        scale_x = float(calib["mm_per_px_x"])
        scale_y = float(calib["mm_per_px_y"])
        score = float(calib["confidence"])
        scale_source = str(calib.get("scale_source") or "ocr_chain")
        scale_warning = calib.get("warning")
    else:
        # Provisional display scale only — never treat 12 000 mm as measured
        long_px = max(bbox_w_px, bbox_h_px)
        mm_per_px = DEFAULT_SPAN_MM / max(long_px, 1)
        scale_x = scale_y = mm_per_px
        scale_source = "provisional"
        score = 0.2
        scale_warning = (
            "Размерные надписи не найдены или не сопоставлены с цепочками. "
            "Применён условный масштаб 12 000 мм по длинной стороне — укажите контрольный отрезок."
        )

    # up is full/small; segments are in `small` space already
    walls: list[list[list[float]]] = []
    partitions: list[list[list[float]]] = []
    min_side = float(min(sw, sh))

    outer = [
        _px_to_mm(x0, y0, sh, scale_x, scale_y),
        _px_to_mm(x1, y0, sh, scale_x, scale_y),
        _px_to_mm(x1, y1, sh, scale_x, scale_y),
        _px_to_mm(x0, y1, sh, scale_x, scale_y),
        _px_to_mm(x0, y0, sh, scale_x, scale_y),
    ]
    walls.append(outer)

    # Internal segments only (outer rectangle already added as closed poly)
    for xa, ya, xb, yb in segs_px[4:] if len(segs_px) > 4 else []:
        length = float(np.hypot(xb - xa, yb - ya))
        kind = _classify_wall_len(length, min_side)
        if kind == "skip":
            continue
        # Skip near-duplicates of the envelope
        if length >= 0.85 * max(bbox_w_px, bbox_h_px):
            continue
        p1 = _px_to_mm(xa, ya, sh, scale_x, scale_y)
        p2 = _px_to_mm(xb, yb, sh, scale_x, scale_y)
        (walls if kind == "wall" else partitions).append([p1, p2])

    walls, partitions, closedness = postprocess_walls(walls, partitions)

    pts = [p for poly in walls + partitions for p in poly]
    xs = [p[0] for p in pts] or [0.0, DEFAULT_SPAN_MM]
    ys = [p[1] for p in pts] or [0.0, DEFAULT_SPAN_MM * 0.67]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    dimensions: list[dict[str, Any]] = [
        {
            "p1": [min_x, min_y - max(400.0, (max_y - min_y) * 0.04)],
            "p2": [max_x, min_y - max(400.0, (max_y - min_y) * 0.04)],
            "value_mm": int(round(max_x - min_x)),
            "sheet": sheet_idx,
            "role": "overall_x",
            "source": scale_source,
        },
        {
            "p1": [min_x - max(400.0, (max_x - min_x) * 0.04), min_y],
            "p2": [min_x - max(400.0, (max_x - min_x) * 0.04), max_y],
            "value_mm": int(round(max_y - min_y)),
            "sheet": sheet_idx,
            "role": "overall_y",
            "source": scale_source,
        },
    ]
    # OCR numbers as text labels at their image position (not fake dim chains)
    texts: list[dict[str, Any]] = []
    for lab in labels[:30]:
        if lab.get("is_area"):
            continue
        v = int(lab["value_mm"])
        # Skip tiny thickness-like and keep architectural spans
        if v < 800:
            continue
        ocr_scale = float(lab.get("ocr_scale") or 1.0)
        full_cx = lab["cx"] / ocr_scale
        full_cy = lab["cy"] / ocr_scale
        sx, sy = full_cx / up, full_cy / up
        # Only labels near / inside building bbox (±margin)
        margin = 0.12 * max(bbox_w_px, bbox_h_px)
        if not (x0 - margin <= sx <= x1 + margin and y0 - margin <= sy <= y1 + margin):
            continue
        cx, cy = _px_to_mm(sx, sy, sh, scale_x, scale_y)
        texts.append(
            {
                "text": str(v),
                "pos": [cx, cy],
                "height": max(120.0, min(bbox_w_px, bbox_h_px) * scale_x * 0.025),
                "sheet": sheet_idx,
                "source": "ocr",
                "conf": lab.get("conf"),
            }
        )

    axes = [
        {"name": "1", "x": min_x, "y0": min_y - 1500, "y1": max_y + 1500, "sheet": sheet_idx},
        {"name": "2", "x": max_x, "y0": min_y - 1500, "y1": max_y + 1500, "sheet": sheet_idx},
        {"name": "А", "y": min_y, "x0": min_x - 1500, "x1": max_x + 1500, "sheet": sheet_idx},
        {"name": "Б", "y": max_y, "x0": min_x - 1500, "x1": max_x + 1500, "sheet": sheet_idx},
    ]

    return {
        "ok": True,
        "walls": walls,
        "partitions": partitions,
        "openings": [],
        "axes": axes,
        "dimensions": dimensions,
        "texts": texts,
        "scale_confidence": score,
        "scale_source": scale_source,
        "scale_warning": scale_warning,
        "ocr_values": (calib or {}).get("ocr_values", []),
        "ocr_chain_matches": (calib or {}).get("matches"),
        "closedness": closedness,
        "proc_size": [sw, sh],
        "upscale": up,
        "source": path.name,
        "index": sheet_idx,
        "bbox_mm": [min_x, min_y, max_x, max_y],
    }


def _extract_stub(
    cleaned_images: list[Path],
    ai_mode: str = "stub",
    source_images: list[Path] | None = None,
) -> dict[str, Any]:
    sources = list(source_images or [])
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
            "scale_source": "provisional",
            "scale_warning": (
                "Нет входных листов; условный контур 12 000 мм. Укажите контрольный отрезок."
            ),
            "needs_scale": True,
            "source_count": 0,
        }

    walls: list[list[list[float]]] = []
    partitions: list[list[list[float]]] = []
    openings: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    sheets: list[dict[str, Any]] = []
    confidences: list[float] = []
    scale_sources: list[str] = []
    warnings: list[str] = []
    closedness_reports: list[dict[str, Any]] = []
    ocr_values: list[int] = []

    for idx, path in enumerate(cleaned_images):
        ocr_path = sources[idx] if idx < len(sources) else path
        sheet = _sheet_geometry(path, idx, ocr_path=ocr_path)
        sheets.append(
            {
                "index": idx,
                "source": sheet.get("source", path.name),
                "ok": sheet.get("ok", False),
                "proc_size": sheet.get("proc_size"),
                "scale_source": sheet.get("scale_source"),
                "ocr_values": sheet.get("ocr_values") or [],
                "closedness": sheet.get("closedness"),
            }
        )
        if not sheet.get("ok"):
            continue
        walls.extend(sheet["walls"])
        partitions.extend(sheet["partitions"])
        openings.extend(sheet["openings"])
        axes.extend(sheet["axes"])
        dimensions.extend(sheet["dimensions"])
        texts.extend(sheet.get("texts") or [])
        confidences.append(float(sheet.get("scale_confidence", 0.2)))
        src = str(sheet.get("scale_source") or "provisional")
        scale_sources.append(src)
        if sheet.get("scale_warning"):
            warnings.append(str(sheet["scale_warning"]))
        if sheet.get("closedness"):
            closedness_reports.append(sheet["closedness"])
        if src.startswith("ocr"):
            ocr_values.extend(int(v) for v in (sheet.get("ocr_values") or []))

    if not walls:
        walls.append([[0, 0], [12000, 0], [12000, 8000], [0, 8000], [0, 0]])
        confidences.append(0.1)
        scale_sources.append("provisional")
        warnings.append(
            "Геометрия не извлечена; показан условный контур. Укажите контрольный отрезок."
        )

    mean_conf = sum(confidences) / max(len(confidences), 1)
    # Prefer strongest sheet source for summary
    priority = {"ocr_chain": 3, "ocr_bbox_weak": 2, "provisional": 1, "default": 0}
    scale_source = max(scale_sources, key=lambda s: priority.get(s, 0)) if scale_sources else "provisional"
    # Ask for control length unless chain OCR is confident
    needs_scale = mean_conf < 0.55 or scale_source in {"provisional", "default", "ocr_bbox_weak"}
    if needs_scale and not warnings:
        if scale_source == "ocr_bbox_weak":
            warnings.append(
                "OCR нашёл размеры, но не привязал их к цепочкам — масштаб приблизительный. "
                "Укажите контрольный отрезок в мм."
            )
        elif mean_conf < 0.55:
            warnings.append(
                "Низкая уверенность масштаба. Укажите контрольный отрезок в мм."
            )

    return {
        "mode": ai_mode,
        "walls": walls,
        "partitions": partitions,
        "openings": openings,
        "axes": axes,
        "dimensions": dimensions,
        "texts": texts,
        "sheets": sheets,
        "units": "mm",
        "source_count": len(cleaned_images),
        "scale_confidence": round(mean_conf, 3),
        "scale_source": scale_source,
        "scale_warning": warnings[0] if warnings else None,
        "ocr_values": ocr_values[:12],
        "closedness": closedness_reports[0] if closedness_reports else None,
        "needs_scale": needs_scale,
    }
