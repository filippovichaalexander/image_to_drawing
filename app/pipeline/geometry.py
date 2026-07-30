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
    vectorization_profile: str = "default",
    vectorization_auto_profile: bool = True,
) -> dict[str, Any]:
    """
    Prefer drawing-faithful exact-trace (contours / lines / arcs / circles).

    Semantic AI (YOLO) is used only inside exact-trace for optional opening
    classification. Stub / full OpenCV+YOLO paths remain as fallback.
    """
    sources = list(source_images or cleaned_images)

    # Primary path: reproduce the scan as DXF primitives
    try:
        from app.pipeline.exact_trace import extract_exact_trace

        traced = extract_exact_trace(
            sources,
            use_yolo_openings=(ai_mode == "full"),
            vectorization_profile=vectorization_profile,
            vectorization_auto_profile=vectorization_auto_profile,
        )
        if _wall_segment_count(traced) >= 1 or traced.get("strokes") or traced.get("details"):
            return traced
        logger.warning("exact_trace returned empty geometry; falling back")
    except Exception as exc:  # noqa: BLE001
        logger.warning("exact_trace failed (%s); falling back to stub/YOLO", exc)

    stub = _extract_stub(cleaned_images, ai_mode=ai_mode, source_images=sources)

    if ai_mode != "full":
        return stub

    try:
        full = _extract_full_ai(sources, cleaned_images=cleaned_images)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AI_MODE=full failed (%s); using OpenCV stub geometry", exc)
        stub["mode"] = "stub"
        stub["yolo_note"] = f"yolo_failed:{exc}"
        return stub

    return _merge_yolo_with_stub(full, stub)


def _wall_segment_count(geometry: dict[str, Any]) -> int:
    """Count wall polylines (closed outer counts as 1; each internal segment too)."""
    n = 0
    for poly in geometry.get("walls") or []:
        if len(poly) >= 2:
            n += 1
    for poly in geometry.get("partitions") or []:
        if len(poly) >= 2:
            n += 1
    return n


def _is_sparse_yolo(geometry: dict[str, Any]) -> bool:
    """True when YOLO only returned an outer box / almost no structure."""
    walls = geometry.get("walls") or []
    openings = geometry.get("openings") or []
    segs = _wall_segment_count(geometry)
    # One closed rect (± duplicate sheets) and no doors/windows → useless for plans
    if segs <= 2 and not openings:
        return True
    if len(walls) <= 2 and all(len(w) <= 5 for w in walls) and not openings:
        return True
    return False


def _merge_yolo_with_stub(full: dict[str, Any], stub: dict[str, Any]) -> dict[str, Any]:
    """Prefer rich OpenCV walls when YOLO is sparse; keep YOLO openings + best OCR."""
    if not full.get("walls") and stub.get("walls"):
        out = dict(stub)
        out["mode"] = "full+stub"
        out["yolo_note"] = "yolo_empty"
        return out

    if _is_sparse_yolo(full) and _wall_segment_count(stub) > _wall_segment_count(full):
        logger.warning(
            "YOLO geometry sparse (%s segs); using OpenCV stub walls (%s segs)",
            _wall_segment_count(full),
            _wall_segment_count(stub),
        )
        out = dict(stub)
        out["mode"] = "full+stub"
        out["yolo_note"] = "yolo_sparse_used_opencv_walls"
        # Merge any YOLO openings / texts not already present
        stub_texts = list(out.get("texts") or [])
        seen = {t.get("text") for t in stub_texts}
        for t in full.get("texts") or []:
            if t.get("text") not in seen:
                stub_texts.append(t)
        out["texts"] = stub_texts
        if full.get("openings"):
            out["openings"] = list(out.get("openings") or []) + list(full["openings"])
        # Prefer OCR-chain scale from either side
        if (full.get("scale_confidence") or 0) > (stub.get("scale_confidence") or 0):
            for key in (
                "scale_confidence",
                "scale_source",
                "scale_warning",
                "needs_scale",
                "ocr_values",
            ):
                if key in full:
                    out[key] = full[key]
        out["sheets"] = full.get("sheets") or stub.get("sheets")
        return out

    full = dict(full)
    full.setdefault("mode", "full")
    return full


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


def _px_to_mm_yup(x: float, y: float, h: float, sx: float, sy: float) -> list[float]:
    """Image Y-down → CAD Y-up."""
    return [float(x * sx), float((h - y) * sy)]


def _extract_full_ai(
    source_images: list[Path],
    cleaned_images: list[Path] | None = None,
) -> dict[str, Any]:
    """
    Full flowchart from ML-instructions.txt:
    preprocess → YOLO-seg → PaddleOCR → scale → postprocess → geometry dict.
    """
    from app.pipeline.ocr import extract_dimensions_and_axes
    from app.pipeline.postprocess import postprocess_walls
    from app.pipeline.preprocess import prepare_for_inference
    from app.pipeline.scale import calibrate_scale, provisional_scale
    from app.pipeline.segmentation import segment_floorplan

    if not source_images:
        raise RuntimeError("No source images for full AI pipeline")

    walls: list[list[list[float]]] = []
    partitions: list[list[list[float]]] = []
    openings: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    sheets: list[dict[str, Any]] = []
    confidences: list[float] = []
    warnings: list[str] = []
    ocr_values: list[int] = []
    scale_sources: list[str] = []

    for sheet_idx, src in enumerate(source_images):
        # uploads/<job>/drawings/x.png → uploads/<job>/prep/
        if src.parent.name in {"drawings", "cleaned"}:
            prep_dir = src.parent.parent / "prep"
        else:
            prep_dir = src.parent / "prep"
        prep = prepare_for_inference(src, prep_dir)

        seg = segment_floorplan(prep["yolo_path"], prep["orig_h"], prep["orig_w"])
        dims, axis_marks = extract_dimensions_and_axes(
            prep["ocr_path"],
            ocr_to_orig=float(prep["ocr_scale"]),
        )

        wall_px = [[list(pt) for pt in poly] for poly in seg["walls"]]
        scale = calibrate_scale(dims, wall_px, prep["orig_w"], prep["orig_h"])
        if scale is None:
            scale = provisional_scale(prep["orig_w"], prep["orig_h"])
            warnings.append(str(scale["warning"]))
        elif scale.get("warning"):
            warnings.append(str(scale["warning"]))

        sx = float(scale["mm_per_px_x"])
        sy = float(scale["mm_per_px_y"])
        oh = float(prep["orig_h"])
        confidences.append(float(scale["confidence"]))
        scale_sources.append(str(scale["source"]))
        ocr_values.extend(int(v) for v in (scale.get("ocr_values") or []))

        sheet_walls: list[list[list[float]]] = []
        for poly in wall_px:
            pts = [_px_to_mm_yup(p[0], p[1], oh, sx, sy) for p in poly]
            if len(pts) >= 3:
                if abs(pts[0][0] - pts[-1][0]) > 1 or abs(pts[0][1] - pts[-1][1]) > 1:
                    pts.append(list(pts[0]))
                sheet_walls.append(pts)

        walls.extend(sheet_walls)

        for door in seg.get("doors") or []:
            openings.append(
                {
                    "kind": "door",
                    "p1": _px_to_mm_yup(door["p1"][0], door["p1"][1], oh, sx, sy),
                    "p2": _px_to_mm_yup(door["p2"][0], door["p2"][1], oh, sx, sy),
                    "sheet": sheet_idx,
                }
            )
        for win in seg.get("windows") or []:
            openings.append(
                {
                    "kind": "window",
                    "p1": _px_to_mm_yup(win["p1"][0], win["p1"][1], oh, sx, sy),
                    "p2": _px_to_mm_yup(win["p2"][0], win["p2"][1], oh, sx, sy),
                    "sheet": sheet_idx,
                }
            )

        for mark in axis_marks:
            cx, cy = _px_to_mm_yup(mark["cx"], mark["cy"], oh, sx, sy)
            name = str(mark["name"])
            if name.isdigit():
                axes.append(
                    {
                        "name": name,
                        "x": cx,
                        "y0": cy - 1500,
                        "y1": cy + 1500,
                        "sheet": sheet_idx,
                        "ocr": True,
                    }
                )
            else:
                axes.append(
                    {
                        "name": name,
                        "y": cy,
                        "x0": cx - 1500,
                        "x1": cx + 1500,
                        "sheet": sheet_idx,
                        "ocr": True,
                    }
                )

        for lab in dims[:40]:
            if int(lab["value_mm"]) < 800:
                continue
            pos = _px_to_mm_yup(lab["cx"], lab["cy"], oh, sx, sy)
            texts.append(
                {
                    "text": str(lab["value_mm"]),
                    "pos": pos,
                    "height": max(120.0, min(prep["orig_w"], prep["orig_h"]) * sx * 0.02),
                    "sheet": sheet_idx,
                    "source": "ocr",
                    "conf": lab.get("conf"),
                }
            )
            dimensions.append(
                {
                    "p1": [pos[0] - float(lab["value_mm"]) / 2, pos[1]],
                    "p2": [pos[0] + float(lab["value_mm"]) / 2, pos[1]],
                    "value_mm": int(lab["value_mm"]),
                    "sheet": sheet_idx,
                    "source": scale["source"],
                    "ocr": lab.get("text"),
                }
            )

        sheets.append(
            {
                "index": sheet_idx,
                "source": src.name,
                "wall_count": len(sheet_walls),
                "scale_source": scale["source"],
                "ocr_values": [d["value_mm"] for d in dims[:8]],
                "weights": seg.get("weights"),
            }
        )

    if not walls:
        raise RuntimeError("YOLO produced no wall masks")

    walls, partitions, closedness = postprocess_walls(walls, partitions)

    mean_conf = sum(confidences) / max(len(confidences), 1)
    priority = {"ocr_chain": 3, "ocr_bbox_weak": 2, "ocr": 2, "provisional": 1, "default": 0}
    scale_source = max(scale_sources, key=lambda s: priority.get(s, 0)) if scale_sources else "provisional"
    needs_scale = mean_conf < 0.55 or scale_source in {"provisional", "default", "ocr_bbox_weak"}
    if needs_scale and not warnings:
        warnings.append(
            "Низкая уверенность масштаба. Укажите контрольный отрезок в мм."
        )

    # Overall dims if OCR chains empty
    if not dimensions and walls:
        pts = [p for poly in walls for p in poly]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dimensions = [
            {
                "p1": [min(xs), min(ys) - 400],
                "p2": [max(xs), min(ys) - 400],
                "value_mm": int(round(max(xs) - min(xs))),
                "sheet": 0,
                "role": "overall_x",
                "source": scale_source,
            },
            {
                "p1": [min(xs) - 400, min(ys)],
                "p2": [min(xs) - 400, max(ys)],
                "value_mm": int(round(max(ys) - min(ys))),
                "sheet": 0,
                "role": "overall_y",
                "source": scale_source,
            },
        ]

    return {
        "mode": "full",
        "walls": walls,
        "partitions": partitions,
        "openings": openings,
        "axes": axes,
        "dimensions": dimensions,
        "texts": texts,
        "sheets": sheets,
        "units": "mm",
        "source_count": len(source_images),
        "scale_confidence": round(mean_conf, 3),
        "scale_source": scale_source,
        "scale_warning": warnings[0] if warnings else None,
        "ocr_values": ocr_values[:12],
        "closedness": closedness,
        "needs_scale": needs_scale,
    }


def _extract_full(cleaned_images: list[Path]) -> dict[str, Any]:
    """Backward-compatible alias — prefer originals via cleaned list."""
    return _extract_full_ai(cleaned_images, cleaned_images=cleaned_images)


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


def _extract_wall_contours(gray: np.ndarray) -> tuple[list[list[list[float]]], tuple[float, float, float, float]]:
    """
    Preserve thick wall outlines from a clean raster.

    Unlike `_extract_ortho_walls()`, this keeps closed wall polygons instead of
    collapsing everything into orthogonal centerlines. It is intentionally biased
    toward the visual drawing matching the source JPG more closely.
    """
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Favor structural ink over dimension ticks/text while keeping wall thickness.
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    thick = ((dist >= 1.0).astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], (0.0, 0.0, float(w), float(h))

    min_area = max(80.0, 0.00025 * h * w)
    min_side = max(10.0, 0.02 * min(h, w))
    polys: list[list[list[float]]] = []
    xs: list[float] = []
    ys: list[float] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if max(cw, ch) < min_side:
            continue
        peri = float(cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, max(1.5, 0.002 * peri), True)
        pts = approx.reshape(-1, 2).tolist()
        if len(pts) < 3:
            continue
        poly = [[float(px), float(py)] for px, py in pts]
        if poly[0] != poly[-1]:
            poly.append(list(poly[0]))
        polys.append(poly)
        xs.extend([float(x), float(x + cw)])
        ys.extend([float(y), float(y + ch)])

    if not polys:
        return [], (0.0, 0.0, float(w), float(h))

    return polys, (min(xs), min(ys), max(xs), max(ys))


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
    wall_polys_px, bbox = _extract_wall_contours(small)
    segs_px, ortho_bbox = _extract_ortho_walls(small)
    if not wall_polys_px:
        bbox = ortho_bbox
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

    if wall_polys_px:
        for poly in wall_polys_px:
            mm_poly = [_px_to_mm(px, py, sh, scale_x, scale_y) for px, py in poly]
            if len(mm_poly) >= 4:
                walls.append(mm_poly)
    else:
        outer = [
            _px_to_mm(x0, y0, sh, scale_x, scale_y),
            _px_to_mm(x1, y0, sh, scale_x, scale_y),
            _px_to_mm(x1, y1, sh, scale_x, scale_y),
            _px_to_mm(x0, y1, sh, scale_x, scale_y),
            _px_to_mm(x0, y0, sh, scale_x, scale_y),
        ]
        walls.append(outer)

    # Keep orthogonal segments only as supplemental partitions / thin internals.
    for xa, ya, xb, yb in segs_px[4:] if len(segs_px) > 4 else []:
        length = float(np.hypot(xb - xa, yb - ya))
        kind = _classify_wall_len(length, min_side)
        if kind == "skip":
            continue
        # Skip near-duplicates of the contour envelope
        if wall_polys_px and length >= 0.85 * max(bbox_w_px, bbox_h_px):
            continue
        p1 = _px_to_mm(xa, ya, sh, scale_x, scale_y)
        p2 = _px_to_mm(xb, yb, sh, scale_x, scale_y)
        partitions.append([p1, p2])

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
