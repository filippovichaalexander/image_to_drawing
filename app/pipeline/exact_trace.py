"""Drawing-faithful vectorization: contours, lines, arcs, circles, hatch.

Goal: reproduce the source JPG as DXF primitives (LINE / LWPOLYLINE / ARC /
CIRCLE / TEXT / HATCH), not a semantic BIM sketch.

YOLO is optional here — only for opening classification / OCR assist.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MAX_TRACE_SIDE = 2400
DEFAULT_SPAN_MM = 12000.0


def _classify_text_kind(text: str) -> str:
    """dimension = numeric size labels; label = room names / misc OCR."""
    cleaned = (
        str(text or "")
        .lower()
        .replace(" ", "")
        .replace(",", "")
        .replace(".", "")
        .replace("мм", "")
        .replace("mm", "")
        .replace("±", "")
    )
    if cleaned.isdigit() and len(cleaned) >= 2:
        return "dimension"
    return "label"


def _poly_centroid(poly: list[list[float]]) -> tuple[float, float]:
    if not poly:
        return 0.0, 0.0
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _poly_length(poly: list[list[float]]) -> float:
    if len(poly) < 2:
        return 0.0
    return float(
        sum(
            math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
            for i in range(len(poly) - 1)
        )
    )


def _reject_ocr_coincident_walls(
    wall_polys: list[list[list[float]]],
    ocr_labs: list[dict[str, Any]],
    *,
    up: float,
    max_len_px: float = 90.0,
    pad_px: float = 18.0,
) -> tuple[list[list[list[float]]], int]:
    """Drop short wall candidates whose centroid sits on OCR digit boxes."""
    if not wall_polys or not ocr_labs:
        return wall_polys, 0
    boxes: list[tuple[float, float, float, float]] = []
    for lab in ocr_labs:
        text = str(lab.get("text") or lab.get("value_mm") or "")
        if _classify_text_kind(text) != "dimension":
            continue
        cx = float(lab.get("cx") or 0.0) / max(up, 1e-9)
        cy = float(lab.get("cy") or 0.0) / max(up, 1e-9)
        # Prefer bbox when present; else a small pad around center
        if lab.get("x0") is not None and lab.get("x1") is not None:
            x0 = float(lab["x0"]) / max(up, 1e-9) - pad_px
            y0 = float(lab.get("y0", cy - pad_px)) / max(up, 1e-9) - pad_px
            x1 = float(lab["x1"]) / max(up, 1e-9) + pad_px
            y1 = float(lab.get("y1", cy + pad_px)) / max(up, 1e-9) + pad_px
        else:
            x0, y0, x1, y1 = cx - pad_px, cy - pad_px, cx + pad_px, cy + pad_px
        boxes.append((x0, y0, x1, y1))
    if not boxes:
        return wall_polys, 0

    kept: list[list[list[float]]] = []
    rejected = 0
    for poly in wall_polys:
        length = _poly_length(poly)
        if length > max_len_px:
            kept.append(poly)
            continue
        cx, cy = _poly_centroid(poly)
        hit = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)
        if hit:
            rejected += 1
            continue
        kept.append(poly)
    return kept, rejected


def _poly_bbox(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    return inter / max(area_a + area_b - inter, 1e-9)


def _strip_walls_overlapping_stairs(
    walls: list[list[list[float]]],
    stairs: list[list[list[float]]],
    iou_thr: float = 0.35,
) -> list[list[list[float]]]:
    if not stairs:
        return walls
    stair_boxes = [_poly_bbox(s) for s in stairs if len(s) >= 3]
    if not stair_boxes:
        return walls
    out: list[list[list[float]]] = []
    for w in walls:
        if len(w) < 2:
            continue
        wb = _poly_bbox(w)
        if any(_bbox_iou(wb, sb) >= iou_thr for sb in stair_boxes):
            continue
        out.append(w)
    return out


def _point_to_seg_dist(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _filter_openings_near_walls(
    openings: list[dict[str, Any]],
    walls: list[list[list[float]]],
    partitions: list[list[list[float]]] | None = None,
    max_dist_mm: float = 350.0,
) -> list[dict[str, Any]]:
    """Keep YOLO doors/windows only if they sit on/near traced wall lines.

    Industrial plans (columns, equipment) create many false openings; without a
    nearby wall stroke they are almost always wrong.
    """
    segs: list[tuple[float, float, float, float]] = []
    for poly in list(walls or []) + list(partitions or []):
        if len(poly) < 2:
            continue
        for i in range(len(poly) - 1):
            a, b = poly[i], poly[i + 1]
            segs.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    if not segs:
        # No walls yet — drop semantic openings (keep CV arcs/leaves with arc/leaf)
        return [op for op in openings if op.get("arc") or op.get("leaf")]

    kept: list[dict[str, Any]] = []
    for op in openings:
        if op.get("arc") or op.get("leaf"):
            kept.append(op)
            continue
        if "p1" not in op or "p2" not in op:
            continue
        mx = 0.5 * (float(op["p1"][0]) + float(op["p2"][0]))
        my = 0.5 * (float(op["p1"][1]) + float(op["p2"][1]))
        dist = min(_point_to_seg_dist(mx, my, *s) for s in segs)
        if dist <= max_dist_mm:
            kept.append(op)
    return kept


@dataclass(frozen=True)
class WallVectorProfile:
    name: str
    min_area_factor: float = 0.00009
    wall_area_factor: float = 0.00025
    blob_area_factor: float = 0.04
    med_t_min: float = 0.55
    med_t_strong_min: float = 0.9
    thinness_min: float = 40.0
    thinness_max: float = 320.0
    aspect_min: float = 1.2
    min_wall_peri_factor: float = 0.028
    hough_thr_factor: float = 0.05
    hough_min_len_factor: float = 0.03
    hough_max_gap: int = 5
    ortho_tol_deg: float = 12.0
    snap_tol_deg: float = 12.0
    noise_peri_max: float = 30.0
    noise_area_max: float = 50.0
    open_kernel: int = 2
    close_kernel: int = 2
    open_iterations: int = 1
    close_iterations: int = 1
    stair_aspect_min: float = 4.5
    stair_med_t_max: float = 1.8
    detail_aspect_max: float = 4.0


WALL_VECTOR_PROFILES: dict[str, WallVectorProfile] = {
    "default": WallVectorProfile(name="default"),
    "thick_lines": replace(
        WallVectorProfile(name="default"),
        name="thick_lines",
        med_t_min=0.9,
        med_t_strong_min=1.2,
        thinness_min=30.0,
        hough_thr_factor=0.03,
        open_kernel=1,
        close_kernel=2,
    ),
    "thin_lines": replace(
        WallVectorProfile(name="default"),
        name="thin_lines",
        min_area_factor=0.00005,
        med_t_min=0.25,
        med_t_strong_min=0.55,
        thinness_max=450.0,
        min_wall_peri_factor=0.02,
        hough_min_len_factor=0.02,
        hough_thr_factor=0.035,
        close_iterations=2,
    ),
    "noisy": replace(
        WallVectorProfile(name="default"),
        name="noisy",
        min_area_factor=0.00015,
        wall_area_factor=0.00035,
        med_t_min=0.7,
        med_t_strong_min=1.0,
        thinness_min=50.0,
        min_wall_peri_factor=0.04,
        hough_thr_factor=0.06,
        open_iterations=2,
        close_iterations=1,
    ),
    "high_detail": replace(
        WallVectorProfile(name="default"),
        name="high_detail",
        min_area_factor=0.00004,
        wall_area_factor=0.00018,
        aspect_min=1.1,
        min_wall_peri_factor=0.015,
        hough_min_len_factor=0.02,
        detail_aspect_max=5.5,
        close_iterations=2,
    ),
}


def _safe_profile_name(name: str | None) -> str:
    if name and name in WALL_VECTOR_PROFILES:
        return name
    return "default"


def _auto_profile_name(gray: np.ndarray) -> str:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    black_ratio = float(np.mean(bw > 0))
    comps = cv2.connectedComponentsWithStats(bw, 8)[2]
    small_components = 0
    if len(comps) > 1:
        areas = comps[1:, cv2.CC_STAT_AREA]
        small_components = int(np.sum(areas <= max(12, int(0.00003 * h * w))))
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    med_thickness = float(np.median(dist[dist > 0])) if np.any(dist > 0) else 0.0

    if small_components > 3500 or black_ratio > 0.16:
        return "noisy"
    if med_thickness >= 1.1:
        return "thick_lines"
    if med_thickness <= 0.45 or black_ratio < 0.045:
        return "thin_lines"
    if small_components > 1600:
        return "high_detail"
    return "default"


def resolve_wall_profile(
    gray: np.ndarray,
    requested: str = "default",
    auto_profile: bool = True,
) -> WallVectorProfile:
    name = _safe_profile_name(requested)
    if auto_profile:
        auto_name = _auto_profile_name(gray)
        if name == "default":
            name = auto_name
    return WALL_VECTOR_PROFILES[name]


def _normalize_plan_scale(
    mm_per_px_x: float,
    mm_per_px_y: float,
    scale_source: str,
) -> tuple[float, float, float]:
    """
    Plans should stay isotropic in XY.
    OCR can infer slightly different X/Y scales from mismatched chains, which
    visually stretches the drawing. Use one uniform plan scale.
    """
    sx = float(mm_per_px_x)
    sy = float(mm_per_px_y)
    ratio = min(sx, sy) / max(sx, sy, 1e-9)
    if scale_source in {"ocr_chain", "ocr_bbox_weak", "ocr"}:
        uniform = math.sqrt(max(sx, 1e-9) * max(sy, 1e-9))
        return uniform, uniform, ratio
    return sx, sy, ratio


def _read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def _downscale(img: np.ndarray, max_side: int = MAX_TRACE_SIDE) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return img, 1.0
    factor = max_side / long_side
    small = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return small, 1.0 / factor


def _px_to_mm(x: float, y: float, h: float, sx: float, sy: float) -> list[float]:
    """Image Y-down → CAD Y-up."""
    return [float(x * sx), float((h - y) * sy)]


def _ink_mask(gray: np.ndarray, profile: WallVectorProfile) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(profile.open_kernel, 1), max(profile.open_kernel, 1))
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(profile.close_kernel, 1), max(profile.close_kernel, 1))
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_kernel, iterations=max(profile.open_iterations, 0))
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=max(profile.close_iterations, 0),
    )
    return bw


def _approx_poly(cnt: np.ndarray, closed: bool = True) -> list[list[float]]:
    peri = float(cv2.arcLength(cnt, closed))
    eps = max(1.2, 0.0018 * peri)
    approx = cv2.approxPolyDP(cnt, eps, closed)
    pts = approx.reshape(-1, 2).astype(float).tolist()
    if closed and pts and (pts[0][0] != pts[-1][0] or pts[0][1] != pts[-1][1]):
        pts.append(list(pts[0]))
    return pts


def _fit_door_arc(poly: list[list[float]]) -> dict[str, float] | None:
    """Fit a circular arc to a polyline that looks like a door swing."""
    if len(poly) < 5:
        return None
    pts = np.array(poly[:-1] if poly[0] == poly[-1] else poly, dtype=np.float64)
    if len(pts) < 5:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    # Algebraic circle fit: x^2+y^2 + d x + e y + f = 0
    A = np.column_stack([x, y, np.ones(len(pts))])
    b = -(x * x + y * y)
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    d, e, f = sol
    cx, cy = -d / 2.0, -e / 2.0
    r2 = cx * cx + cy * cy - f
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    if r < 8 or r > 400:
        return None
    # Residual check — keep only good arcs
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    if float(np.median(resid)) > max(2.5, 0.08 * r):
        return None
    # Angular span
    angs = np.arctan2(y - cy, x - cx)
    start = float(np.min(angs))
    end = float(np.max(angs))
    span = end - start
    if span < math.radians(25) or span > math.radians(120):
        # try wrapping
        angs2 = (angs + 2 * math.pi) % (2 * math.pi)
        start = float(np.min(angs2))
        end = float(np.max(angs2))
        span = end - start
        if span < math.radians(25) or span > math.radians(120):
            return None
    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": float(r),
        "start_deg": math.degrees(start),
        "end_deg": math.degrees(end),
    }


def _detect_axis_circles(bw: np.ndarray) -> list[dict[str, float]]:
    """Hough circles ≈ GOST axis bubbles."""
    h, w = bw.shape[:2]
    min_r = max(6, int(0.006 * min(h, w)))
    max_r = max(min_r + 4, int(0.035 * min(h, w)))
    blur = cv2.GaussianBlur(bw, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, min_r * 3),
        param1=120,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )
    out: list[dict[str, float]] = []
    if circles is None:
        return out
    for c in np.round(circles[0]).astype(int):
        out.append({"cx": float(c[0]), "cy": float(c[1]), "r": float(c[2])})
    return out


def _segment_angle_deg(p1: list[float], p2: list[float]) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
    return 180.0 - ang if ang > 90.0 else ang


def _is_ortho_seg(p1: list[float], p2: list[float], tol_deg: float = 12.0) -> bool:
    ang = _segment_angle_deg(p1, p2)
    return ang <= tol_deg or abs(ang - 90.0) <= tol_deg


def _is_door_leaf_seg(
    p1: list[float],
    p2: list[float],
    min_len: float,
    max_len: float,
    diag_tol: float = 18.0,
) -> bool:
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if not (min_len <= length <= max_len):
        return False
    return abs(_segment_angle_deg(p1, p2) - 45.0) <= diag_tol


def _door_length_bounds(h: int, w: int) -> tuple[float, float]:
    door_min = max(18.0, 0.012 * min(h, w))
    door_max = max(door_min * 2, 0.12 * min(h, w))
    return door_min, door_max


def _contour_stats(
    cnt: np.ndarray,
    bw: np.ndarray,
    dist: np.ndarray,
    profile: WallVectorProfile,
) -> dict[str, Any] | None:
    h, w = bw.shape[:2]
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri < 16:
        return None
    x, y, cw, ch = cv2.boundingRect(cnt)
    aspect = max(cw, ch) / max(min(cw, ch), 1)
    thinness = (peri * peri) / max(area, 1.0)
    sample = cnt.reshape(-1, 2)[:: max(1, len(cnt) // 40)]
    thicknesses = [
        float(dist[int(py), int(px)])
        for px, py in sample
        if 0 <= py < h and 0 <= px < w
    ]
    med_t = float(np.median(thicknesses)) if thicknesses else 0.0
    poly_closed = _approx_poly(cnt, closed=True)
    if len(poly_closed) < 3:
        return None
    open_poly = _approx_poly(cnt, closed=False)
    img_area = float(h * w)
    min_area = max(20.0, profile.min_area_factor * img_area)
    wall_area = max(min_area * 1.5, profile.wall_area_factor * img_area)
    blob_area = profile.blob_area_factor * img_area
    wall_min_peri = max(28.0, profile.min_wall_peri_factor * min(h, w))
    mask = np.zeros(bw.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    ink_ratio = float(np.sum(bw[mask > 0] > 0)) / max(area, 1.0)
    return {
        "area": area,
        "peri": peri,
        "aspect": aspect,
        "thinness": thinness,
        "med_t": med_t,
        "poly_closed": poly_closed,
        "poly_open": open_poly,
        "blob_area": blob_area,
        "min_area": min_area,
        "wall_area": wall_area,
        "wall_min_peri": wall_min_peri,
        "ink_ratio": ink_ratio,
    }


def _reject_non_wall_structures(
    stats: dict[str, Any],
    profile: WallVectorProfile,
) -> str | None:
    area = float(stats["area"])
    peri = float(stats["peri"])
    aspect = float(stats["aspect"])
    med_t = float(stats["med_t"])
    blob_area = float(stats["blob_area"])
    ink_ratio = float(stats["ink_ratio"])
    if peri < profile.noise_peri_max and area < profile.noise_area_max:
        return "noise"
    if area >= blob_area:
        return "blob"
    if ink_ratio < 0.15 and area > 200:
        return "text_box"
    if area > float(stats["min_area"]) * 0.5 and aspect > profile.stair_aspect_min and med_t < profile.stair_med_t_max:
        return "stair_hatch"
    return None


def _classify_contour_as_wall_or_not(
    stats: dict[str, Any],
    profile: WallVectorProfile,
) -> str:
    area = float(stats["area"])
    peri = float(stats["peri"])
    aspect = float(stats["aspect"])
    thinness = float(stats["thinness"])
    med_t = float(stats["med_t"])
    min_area = float(stats["min_area"])
    wall_area = float(stats["wall_area"])
    blob_area = float(stats["blob_area"])
    wall_min_peri = float(stats["wall_min_peri"])

    if (
        med_t >= profile.med_t_min
        and area >= min_area
        and area < blob_area
        and profile.thinness_min <= thinness <= profile.thinness_max
        and aspect >= profile.aspect_min
        and peri >= wall_min_peri
        and (
            (area >= wall_area and aspect >= max(profile.aspect_min, 1.15))
            or (med_t >= profile.med_t_strong_min and peri >= 40)
        )
    ):
        return "wall"
    if area >= min_area * 0.35 and aspect < profile.detail_aspect_max and area < wall_area * 8:
        return "detail"
    return "skip"


def _extract_hough_wall_segments(
    bw: np.ndarray,
    profile: WallVectorProfile,
    door_min: float,
    door_max: float,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    h, w = bw.shape[:2]
    edges = cv2.Canny(bw, 50, 150)
    min_len = max(22, int(profile.hough_min_len_factor * min(h, w)))
    threshold = max(25, int(profile.hough_thr_factor * min(h, w)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_len,
        maxLineGap=profile.hough_max_gap,
    )
    walls: list[list[list[float]]] = []
    door_leaves: list[list[list[float]]] = []
    if lines is None:
        return walls, door_leaves
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        p1 = [float(x1), float(y1)]
        p2 = [float(x2), float(y2)]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if _is_door_leaf_seg(p1, p2, door_min, door_max):
            door_leaves.append([p1, p2])
            continue
        if not _is_ortho_seg(p1, p2, tol_deg=profile.ortho_tol_deg):
            continue
        if length >= min_len * 1.2:
            walls.append([p1, p2])
    return walls, door_leaves


def _extract_strokes(bw: np.ndarray, profile: WallVectorProfile) -> dict[str, Any]:
    """
    Contour-vectorize CAD ink as *linework* (not filled room blobs).

    fix_3: reject door swings / diagonal leaves early; walls stay ortho-capable.
    """
    h, w = bw.shape[:2]
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    blob_area = profile.blob_area_factor * float(h * w)

    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    walls: list[list[list[float]]] = []
    details: list[list[list[float]]] = []
    hatches: list[list[list[float]]] = []
    arcs: list[dict[str, float]] = []
    door_leaves: list[list[list[float]]] = []
    strokes: list[list[list[float]]] = []

    door_min, door_max = _door_length_bounds(h, w)

    for cnt in contours:
        stats = _contour_stats(cnt, bw, dist, profile)
        if not stats:
            continue
        area = float(stats["area"])
        peri = float(stats["peri"])
        aspect = float(stats["aspect"])
        thinness = float(stats["thinness"])
        poly = stats["poly_closed"]
        open_poly = stats["poly_open"]

        # Door swing / thin non-rect curve (fix_3) — before walls
        if thinness > 120 and aspect < 2.5 and 25 < peri < 0.4 * (h + w):
            arc = _fit_door_arc(open_poly)
            if arc is not None:
                arcs.append(arc)
                continue
            if len(open_poly) >= 2:
                p1, p2 = open_poly[0], open_poly[-1]
                if _is_door_leaf_seg(p1, p2, door_min, door_max):
                    door_leaves.append([p1, p2])
                    continue
            # Arc-like by enclosing circle
            if len(cnt) > 20:
                (_cx, _cy), radius = cv2.minEnclosingCircle(cnt)
                area_ratio = area / max(math.pi * radius * radius, 1.0)
                if area_ratio < 0.45 and 15 < radius < 220:
                    if arc is None:
                        arc = _fit_door_arc(open_poly)
                    if arc is not None:
                        arcs.append(arc)
                        continue

        reject_reason = _reject_non_wall_structures(stats, profile)
        if reject_reason == "noise":
            continue
        if reject_reason == "blob":
            continue
        if reject_reason == "text_box":
            continue
        if reject_reason == "stair_hatch":
            hatches.append(poly)
            continue

        # Peel diagonal 2-point polylines out of wall candidates
        if len(poly) <= 3:
            p1, p2 = poly[0], poly[-2] if poly[0] == poly[-1] and len(poly) > 2 else poly[-1]
            if _is_door_leaf_seg(p1, p2, door_min, door_max):
                door_leaves.append([p1, p2])
                continue

        kind = _classify_contour_as_wall_or_not(stats, profile)
        if kind == "wall":
            walls.append(poly)
            continue
        if kind == "detail":
            details.append(poly)
        elif peri >= 28:
            strokes.append(poly)

    hough_walls, hough_door_leaves = _extract_hough_wall_segments(bw, profile, door_min, door_max)
    walls.extend(hough_walls[:400])
    door_leaves.extend(hough_door_leaves)

    circles = _detect_axis_circles(bw)

    if not walls and details:
        candidates = []
        for p in details:
            xs = [q[0] for q in p]
            ys = [q[1] for q in p]
            bw_ = max(xs) - min(xs)
            bh_ = max(ys) - min(ys)
            a = bw_ * bh_
            if a <= 0 or a >= blob_area:
                continue
            aspect = max(bw_, bh_) / max(min(bw_, bh_), 1)
            if aspect >= profile.aspect_min:
                candidates.append((aspect * min(bw_, bh_), p))
        candidates.sort(key=lambda t: t[0], reverse=True)
        promote_n = max(1, min(60, len(candidates) // 2)) if candidates else 0
        walls = [p for _, p in candidates[:promote_n]]
        promoted = {id(p) for _, p in candidates[:promote_n]}
        details = [p for p in details if id(p) not in promoted]

    logger.info(
        "exact_trace contours=%s walls=%s details=%s hatches=%s arcs=%s leaves=%s circles=%s",
        len(contours),
        len(walls),
        len(details),
        len(hatches),
        len(arcs),
        len(door_leaves),
        len(circles),
    )
    return {
        "walls": walls,
        "details": details,
        "hatches": hatches,
        "arcs": arcs,
        "door_leaves": door_leaves,
        "strokes": strokes,
        "circles": circles,
        "size": (w, h),
    }


def _build_dim_chains(
    labels: list[dict[str, Any]],
    sh: float,
    sx: float,
    sy: float,
    up: float,
) -> list[dict[str, Any]]:
    """Place dimension segments from OCR text boxes (not invented lengths).

    Older logic spanned ±value_mm/2 around the glyph center. That puts green
    ticks in the wrong place vs the scan: OCR text is right, lines are wrong.
    Use the OCR bounding box long axis instead (approx. where the printed
    dimension sits).
    """
    dims: list[dict[str, Any]] = []
    for lab in labels:
        v = int(lab.get("value_mm") or 0)
        if v < 200 or bool(lab.get("is_area")):
            continue
        ocr_scale = float(lab.get("ocr_scale") or 1.0)
        box = lab.get("box")
        if not box or len(box) < 2:
            continue
        xs = [float(p[0]) / ocr_scale / up for p in box]
        ys = [float(p[1]) / ocr_scale / up for p in box]
        cx = float(sum(xs) / len(xs))
        cy = float(sum(ys) / len(ys))
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        # Prefer elongated text boxes; skip near-square (room areas etc.)
        if max(w, h) < 8:
            continue
        if w >= h:
            # Horizontal dim label — line along text baseline
            y = cy
            p1 = _px_to_mm(min(xs), y, sh, sx, sy)
            p2 = _px_to_mm(max(xs), y, sh, sx, sy)
            orient = "h"
        else:
            x = cx
            p1 = _px_to_mm(x, min(ys), sh, sx, sy)
            p2 = _px_to_mm(x, max(ys), sh, sx, sy)
            orient = "v"
        # Skip tiny segments (OCR box too small to mean a dim line)
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 80:
            continue
        dims.append(
            {
                "p1": p1,
                "p2": p2,
                "value_mm": v,
                "role": "ocr_box",
                "orient": orient,
                "source": "ocr_box",
            }
        )
    return dims


def _axes_from_circles_and_ocr(
    circles_px: list[dict[str, float]],
    axis_marks: list[dict[str, Any]],
    sh: float,
    sw: float,
    sx: float,
    sy: float,
    up: float,
) -> list[dict[str, Any]]:
    axes: list[dict[str, Any]] = []
    # Prefer OCR marks with nearby circle bubbles
    used_circles: set[int] = set()
    for mark in axis_marks:
        name = str(mark.get("name") or "").strip()
        if not name:
            continue
        ocr_scale = float(mark.get("ocr_scale") or 1.0)
        # axis marks from RapidOCR already in original px when ocr_to_orig applied
        cx = float(mark["cx"]) / up if up != 1.0 else float(mark["cx"])
        cy = float(mark["cy"]) / up if up != 1.0 else float(mark["cy"])
        # If mark coords are already original, divide by ocr path differently —
        # extract_dimensions returns original-space when ocr_to_orig is set.
        # Here marks come pre-scaled to original; convert to process space:
        cx_p, cy_p = cx / up, cy / up

        best_i, best_d = None, 1e18
        for i, c in enumerate(circles_px):
            if i in used_circles:
                continue
            d = math.hypot(c["cx"] - cx_p, c["cy"] - cy_p)
            if d < best_d:
                best_d, best_i = d, i
        bubble = None
        if best_i is not None and best_d < max(40.0, 0.04 * min(sw, sh)):
            used_circles.add(best_i)
            c = circles_px[best_i]
            bubble = {
                "cx": _px_to_mm(c["cx"], c["cy"], sh, sx, sy)[0],
                "cy": _px_to_mm(c["cx"], c["cy"], sh, sx, sy)[1],
                "r": float(c["r"]) * ((sx + sy) * 0.5),
            }
            cx_p, cy_p = c["cx"], c["cy"]

        mm = _px_to_mm(cx_p, cy_p, sh, sx, sy)
        if name.isdigit() or (len(name) <= 2 and name.isascii() and name.isalnum()):
            # vertical axis (numbered)
            axes.append(
                {
                    "name": name,
                    "x": mm[0],
                    "y0": 0.0,
                    "y1": sh * sy,
                    "bubble": bubble,
                    "ocr": True,
                }
            )
        else:
            axes.append(
                {
                    "name": name,
                    "y": mm[1],
                    "x0": 0.0,
                    "x1": sw * sx,
                    "bubble": bubble,
                    "ocr": True,
                }
            )

    # Remaining circles without OCR — still export as bubbles
    for i, c in enumerate(circles_px):
        if i in used_circles:
            continue
        mm = _px_to_mm(c["cx"], c["cy"], sh, sx, sy)
        axes.append(
            {
                "name": "",
                "bubble": {
                    "cx": mm[0],
                    "cy": mm[1],
                    "r": float(c["r"]) * ((sx + sy) * 0.5),
                },
                "x": mm[0],
                "y0": mm[1] - 800,
                "y1": mm[1] + 800,
            }
        )
    return axes


def _openings_from_yolo_and_arcs(
    yolo: dict[str, Any] | None,
    arcs_px: list[dict[str, float]],
    door_leaves_px: list[list[list[float]]],
    sh: float,
    sx: float,
    sy: float,
    sheet_idx: int,
) -> list[dict[str, Any]]:
    openings: list[dict[str, Any]] = []
    if yolo:
        for door in yolo.get("doors") or []:
            p1 = _px_to_mm(door["p1"][0], door["p1"][1], sh, sx, sy)
            p2 = _px_to_mm(door["p2"][0], door["p2"][1], sh, sx, sy)
            width_mm = None
            if door.get("width_px") is not None:
                width_mm = float(door["width_px"]) * ((sx + sy) * 0.5)
            openings.append(
                {
                    "kind": "door",
                    "p1": p1,
                    "p2": p2,
                    "width_mm": width_mm,
                    "sheet": sheet_idx,
                }
            )
        for win in yolo.get("windows") or []:
            p1 = _px_to_mm(win["p1"][0], win["p1"][1], sh, sx, sy)
            p2 = _px_to_mm(win["p2"][0], win["p2"][1], sh, sx, sy)
            width_mm = None
            if win.get("width_px") is not None:
                width_mm = float(win["width_px"]) * ((sx + sy) * 0.5)
            openings.append(
                {
                    "kind": "window",
                    "p1": p1,
                    "p2": p2,
                    "width_mm": width_mm,
                    "sheet": sheet_idx,
                }
            )

    for arc in arcs_px:
        center = _px_to_mm(arc["cx"], arc["cy"], sh, sx, sy)
        openings.append(
            {
                "kind": "door",
                "arc": {
                    "cx": center[0],
                    "cy": center[1],
                    "r": float(arc["r"]) * ((sx + sy) * 0.5),
                    "start_deg": float(arc["start_deg"]),
                    "end_deg": float(arc["end_deg"]),
                },
                "sheet": sheet_idx,
            }
        )

    for leaf in door_leaves_px:
        if len(leaf) < 2:
            continue
        p1 = _px_to_mm(leaf[0][0], leaf[0][1], sh, sx, sy)
        p2 = _px_to_mm(leaf[-1][0], leaf[-1][1], sh, sx, sy)
        openings.append(
            {
                "kind": "door",
                "leaf": [p1, p2],
                "p1": p1,
                "p2": p2,
                "sheet": sheet_idx,
            }
        )
    return openings


def extract_exact_trace(
    source_images: list[Path],
    use_yolo_openings: bool = True,
    use_yolo_stairs: bool = True,
    vectorization_profile: str = "default",
    vectorization_auto_profile: bool = True,
    use_sam: bool = False,
) -> dict[str, Any]:
    """Primary drawing-faithful extraction for one or more sheets."""
    from app.pipeline.ocr import extract_dimensions_and_axes
    from app.pipeline.ocr_dims import calibrate_span_mm, extract_dimension_labels
    from app.pipeline.preprocess import prepare_for_inference

    if not source_images:
        raise RuntimeError("No source images for exact trace")

    walls: list[list[list[float]]] = []
    partitions: list[list[list[float]]] = []
    details: list[list[list[float]]] = []
    strokes: list[list[list[float]]] = []
    hatches: list[list[list[float]]] = []
    stairs: list[list[list[float]]] = []
    openings: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    dimensions: list[dict[str, Any]] = []
    texts: list[dict[str, Any]] = []
    circles: list[dict[str, float]] = []
    sheets: list[dict[str, Any]] = []
    confidences: list[float] = []
    warnings: list[str] = []
    scale_sources: list[str] = []
    ocr_values: list[int] = []
    scale_ratios: list[float] = []
    ocr_rejected = 0
    sam_refined = 0
    underlay: dict[str, Any] | None = None

    resolved_profile_name = _safe_profile_name(vectorization_profile)
    active_profile_name = resolved_profile_name

    for sheet_idx, src in enumerate(source_images):
        if src.parent.name in {"drawings", "cleaned"}:
            prep_dir = src.parent.parent / "prep"
        else:
            prep_dir = src.parent / "prep"
        prep = prepare_for_inference(src, prep_dir)

        gray = _read_gray(prep["ocr_path"])
        if gray is None:
            gray = _read_gray(src)
        if gray is None:
            warnings.append(f"Cannot read {src.name}")
            continue
        small, up = _downscale(gray)
        sh, sw = small.shape[:2]
        wall_profile = resolve_wall_profile(
            small,
            requested=resolved_profile_name,
            auto_profile=vectorization_auto_profile,
        )
        active_profile_name = wall_profile.name
        bw = _ink_mask(small, wall_profile)
        traced = _extract_strokes(bw, wall_profile)

        labels = extract_dimension_labels(Path(prep["ocr_path"]))
        dims_ocr_early, _ = extract_dimensions_and_axes(
            Path(prep["ocr_path"]),
            ocr_to_orig=float(prep["ocr_scale"]),
        )
        cleaned_walls, n_rej = _reject_ocr_coincident_walls(
            traced["walls"],
            dims_ocr_early,
            up=up,
        )
        traced["walls"] = cleaned_walls
        ocr_rejected += n_rej

        # Build synthetic segments for scale calib from wall contours
        segs_px: list[tuple[int, int, int, int]] = []
        for poly in traced["walls"][:40]:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            if max(xs) - min(xs) >= max(ys) - min(ys):
                segs_px.append((int(min(xs)), int(np.median(ys)), int(max(xs)), int(np.median(ys))))
            else:
                segs_px.append((int(np.median(xs)), int(min(ys)), int(np.median(xs)), int(max(ys))))

        bbox_w = float(sw)
        bbox_h = float(sh)
        if traced["walls"]:
            allx = [p[0] for poly in traced["walls"] for p in poly]
            ally = [p[1] for poly in traced["walls"] for p in poly]
            bbox_w = max(max(allx) - min(allx), 1.0)
            bbox_h = max(max(ally) - min(ally), 1.0)

        ocr_scale0 = float(labels[0]["ocr_scale"]) if labels else 1.0
        ocr_to_proc = 1.0 / max(ocr_scale0 * up, 1e-9)
        calib = calibrate_span_mm(
            labels,
            bbox_w,
            bbox_h,
            segments_px=segs_px,
            ocr_to_proc=ocr_to_proc,
        )
        if calib:
            raw_sx = float(calib["mm_per_px_x"])
            raw_sy = float(calib["mm_per_px_y"])
            score = float(calib["confidence"])
            scale_source = str(calib.get("scale_source") or "ocr_chain")
            sx, sy, scale_ratio = _normalize_plan_scale(raw_sx, raw_sy, scale_source)
            if calib.get("warning"):
                warnings.append(str(calib["warning"]))
            if scale_ratio < 0.92:
                warnings.append(
                    f"Anisotropic OCR scale normalized for plan XY: x={raw_sx:.4f}, y={raw_sy:.4f}"
                )
            scale_ratios.append(scale_ratio)
            ocr_values.extend(int(v) for v in (calib.get("ocr_values") or []))
        else:
            long_px = max(bbox_w, bbox_h)
            sx = sy = DEFAULT_SPAN_MM / max(long_px, 1)
            score = 0.2
            scale_source = "provisional"
            scale_ratios.append(1.0)
            warnings.append(
                "Размерные надписи не сопоставлены. Условный масштаб 12 000 мм — укажите контрольный отрезок."
            )

        confidences.append(score)
        scale_sources.append(scale_source)

        def to_mm_poly(poly: list[list[float]]) -> list[list[float]]:
            return [_px_to_mm(p[0], p[1], sh, sx, sy) for p in poly]

        if underlay is None:
            underlay = {
                "min": [0.0, 0.0],
                "max": [float(sw) * sx, float(sh) * sy],
                "sheet": sheet_idx,
                "source": src.name,
            }

        for poly in traced["walls"]:
            mm = to_mm_poly(poly)
            if len(mm) >= 2:
                walls.append(mm)
        for poly in traced["details"]:
            mm = to_mm_poly(poly)
            if len(mm) < 2:
                continue
            details.append(mm)
            xs = [p[0] for p in mm]
            ys = [p[1] for p in mm]
            w = max(xs) - min(xs)
            hgt = max(ys) - min(ys)
            aspect = max(w, hgt) / max(min(w, hgt), 1.0)
            if aspect >= 3.0 and min(w, hgt) < max(w, hgt) * 0.25:
                partitions.append(mm)
        for poly in traced["hatches"]:
            mm = to_mm_poly(poly)
            if len(mm) >= 2:
                hatches.append(mm)
        # Skip raw strokes — they duplicate walls and cause grey+red overlap
        _ = traced.get("strokes")

        for c in traced["circles"]:
            mm = _px_to_mm(c["cx"], c["cy"], sh, sx, sy)
            circles.append(
                {
                    "cx": mm[0],
                    "cy": mm[1],
                    "r": float(c["r"]) * ((sx + sy) * 0.5),
                    "layer": "axes",
                }
            )

        # OCR texts at original locations
        dims_ocr, axis_marks = extract_dimensions_and_axes(
            Path(prep["ocr_path"]),
            ocr_to_orig=float(prep["ocr_scale"]),
        )
        # axis_marks / dims are in original pixels
        for lab in dims_ocr[:60]:
            v = int(lab.get("value_mm") or 0)
            text = str(lab.get("text") or v)
            kind = _classify_text_kind(text)
            if kind == "dimension":
                cleaned = (
                    text.replace(" ", "")
                    .replace(",", "")
                    .replace(".", "")
                    .replace("мм", "")
                    .replace("mm", "")
                )
                if not (cleaned.isdigit() and len(cleaned) >= 2):
                    continue
                if v < 100 and len(cleaned) < 3:
                    continue
            pos = _px_to_mm(lab["cx"] / up, lab["cy"] / up, sh, sx, sy)
            texts.append(
                {
                    "text": text,
                    "pos": pos,
                    "height": max(80.0, min(sw, sh) * sx * 0.018),
                    "sheet": sheet_idx,
                    "source": "ocr",
                    "kind": kind,
                    "rotation": float(lab.get("rotation") or 0.0),
                }
            )

        dimensions.extend(_build_dim_chains(labels, sh, sx, sy, up))
        # Overall dims
        if walls:
            pts = [p for poly in walls for p in poly]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            dimensions.append(
                {
                    "p1": [min(xs), min(ys) - max(400.0, (max(ys) - min(ys)) * 0.04)],
                    "p2": [max(xs), min(ys) - max(400.0, (max(ys) - min(ys)) * 0.04)],
                    "value_mm": int(round(max(xs) - min(xs))),
                    "role": "overall_x",
                    "source": scale_source,
                }
            )
            dimensions.append(
                {
                    "p1": [min(xs) - max(400.0, (max(xs) - min(xs)) * 0.04), min(ys)],
                    "p2": [min(xs) - max(400.0, (max(xs) - min(xs)) * 0.04), max(ys)],
                    "value_mm": int(round(max(ys) - min(ys))),
                    "role": "overall_y",
                    "source": scale_source,
                }
            )

        # Axis marks: convert original px → process px
        marks_proc = []
        for m in axis_marks:
            marks_proc.append(
                {
                    **m,
                    "cx": float(m["cx"]) / up,
                    "cy": float(m["cy"]) / up,
                    "ocr_scale": 1.0,
                }
            )
        axes.extend(
            _axes_from_circles_and_ocr(
                traced["circles"], marks_proc, sh, sw, sx, sy, up=1.0
            )
        )

        yolo = None
        run_yolo = bool(use_yolo_openings or use_yolo_stairs)
        if run_yolo:
            try:
                from app.pipeline.segmentation import segment_floorplan

                yolo = segment_floorplan(prep["yolo_path"], prep["orig_h"], prep["orig_w"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("YOLO optional skip: %s", exc)

        if yolo and use_sam:
            try:
                from app.pipeline.sam_refiner import refine_yolo_masks_with_sam

                for key in ("walls", "stairs"):
                    masks = list(yolo.get(key) or [])
                    if not masks:
                        continue
                    refined = refine_yolo_masks_with_sam(Path(prep["yolo_path"]), masks)
                    if refined:
                        yolo[key] = refined
                        sam_refined += len(refined)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SAM refine skipped: %s", exc)

        if yolo and use_yolo_stairs:
            for stair in yolo.get("stairs") or []:
                mm = to_mm_poly([[float(p[0]) / up, float(p[1]) / up] for p in stair])
                if len(mm) >= 3:
                    stairs.append(mm)

        # Scale YOLO boxes (original px) into process space for openings helper
        yolo_proc = None
        if yolo and use_yolo_openings:
            yolo_proc = {"doors": [], "windows": []}
            for door in yolo.get("doors") or []:
                item = {
                    "p1": [door["p1"][0] / up, door["p1"][1] / up],
                    "p2": [door["p2"][0] / up, door["p2"][1] / up],
                }
                if door.get("width_px") is not None:
                    item["width_px"] = float(door["width_px"]) / up
                yolo_proc["doors"].append(item)
            for win in yolo.get("windows") or []:
                item = {
                    "p1": [win["p1"][0] / up, win["p1"][1] / up],
                    "p2": [win["p2"][0] / up, win["p2"][1] / up],
                }
                if win.get("width_px") is not None:
                    item["width_px"] = float(win["width_px"]) / up
                yolo_proc["windows"].append(item)

        openings.extend(
            _openings_from_yolo_and_arcs(
                yolo_proc,
                traced["arcs"],
                traced.get("door_leaves") or [],
                sh,
                sx,
                sy,
                sheet_idx,
            )
        )

        sheets.append(
            {
                "index": sheet_idx,
                "source": src.name,
                "walls": len(traced["walls"]),
                "strokes": len(traced.get("strokes") or []),
                "arcs": len(traced["arcs"]),
                "door_leaves": len(traced.get("door_leaves") or []),
                "circles": len(traced["circles"]),
                "hatches": len(traced["hatches"]),
                "stairs": len(yolo.get("stairs") or []) if (yolo and use_yolo_stairs) else 0,
                "wall_profile": wall_profile.name,
                "scale_source": scale_source,
            }
        )

    from app.pipeline.postprocess import postprocess_walls

    walls = _strip_walls_overlapping_stairs(walls, stairs)
    walls, partitions, closedness = postprocess_walls(
        walls,
        partitions,
        snap_tol_deg=WALL_VECTOR_PROFILES[active_profile_name].snap_tol_deg,
        min_fragment_factor=0.012 if active_profile_name != "high_detail" else 0.008,
        overlap_iou=0.82 if active_profile_name != "thin_lines" else 0.9,
    )
    for leaf in closedness.get("door_leaves") or []:
        openings.append({**leaf, "sheet": 0})

    before_open = len(openings)
    openings = _filter_openings_near_walls(openings, walls, partitions)
    if before_open != len(openings):
        logger.info(
            "Openings near-wall filter: %s → %s",
            before_open,
            len(openings),
        )

    mean_conf = sum(confidences) / max(len(confidences), 1)
    priority = {"ocr_box": 3, "ocr_chain": 3, "ocr_bbox_weak": 2, "ocr": 2, "provisional": 1}
    scale_source = max(scale_sources, key=lambda s: priority.get(s, 0)) if scale_sources else "provisional"
    needs_scale = mean_conf < 0.55 or scale_source in {"provisional", "ocr_bbox_weak"}

    return {
        "mode": "exact_trace",
        "walls": walls,
        "partitions": partitions,
        "details": details,
        "strokes": strokes,
        "hatches": hatches,
        "stairs": stairs,
        "circles": circles,
        "openings": openings,
        "axes": axes,
        "dimensions": dimensions,
        "texts": texts,
        "sheets": sheets,
        "units": "mm",
        "source_count": len(source_images),
        "scale_confidence": round(mean_conf, 3),
        "scale_source": scale_source,
        "scale_xy_ratio_raw": round(min(scale_ratios) if scale_ratios else 1.0, 3),
        "scale_warning": warnings[0] if warnings else None,
        "ocr_values": ocr_values[:12],
        "needs_scale": needs_scale,
        "fill_walls": False,
        "vectorization_profile": active_profile_name,
        "vectorization_profile_requested": resolved_profile_name,
        "vectorization_auto_profile": vectorization_auto_profile,
        "closedness": {k: v for k, v in closedness.items() if k != "door_leaves"},
        "trace_note": "ortho-snapped walls; door leaves/arcs on openings",
        "ocr_wall_blobs_rejected": ocr_rejected,
        "sam_used": bool(use_sam and sam_refined),
        "sam_refined": sam_refined,
        "underlay": underlay,
    }
