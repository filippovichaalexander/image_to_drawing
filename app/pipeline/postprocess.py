"""Post-process wall contours: smooth spikes, close gaps, verify closedness."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segment_length(poly: list[list[float]]) -> float:
    if len(poly) < 2:
        return 0.0
    return sum(_dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def _bbox(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1.0)


def _is_closed(poly: list[list[float]], tol: float) -> bool:
    if len(poly) < 3:
        return False
    return _dist(poly[0], poly[-1]) <= tol


def _ensure_closed(poly: list[list[float]], tol: float) -> list[list[float]]:
    if not poly:
        return poly
    out = [list(p) for p in poly]
    if _dist(out[0], out[-1]) > tol:
        out.append(list(out[0]))
    elif _dist(out[0], out[-1]) > 1e-6:
        out[-1] = list(out[0])
    return out


def _remove_spikes(poly: list[list[float]], min_edge: float, max_turn_cos: float = -0.2) -> list[list[float]]:
    """
    Drop vertices that create sharp outward spikes / tiny zigzags.
    max_turn_cos: keep turns where cos(angle) > threshold (drop near-180° folds).
    """
    if len(poly) < 4:
        return poly
    closed = _dist(poly[0], poly[-1]) < min_edge * 0.5
    pts = poly[:-1] if closed else list(poly)
    if len(pts) < 3:
        return poly

    kept: list[list[float]] = []
    n = len(pts)
    for i in range(n):
        prev = pts[(i - 1) % n]
        cur = pts[i]
        nxt = pts[(i + 1) % n]
        e1 = _dist(prev, cur)
        e2 = _dist(cur, nxt)
        if e1 < min_edge * 0.35 and e2 < min_edge * 0.35:
            continue  # tiny stub
        v1x, v1y = cur[0] - prev[0], cur[1] - prev[1]
        v2x, v2y = nxt[0] - cur[0], nxt[1] - cur[1]
        n1 = math.hypot(v1x, v1y) or 1.0
        n2 = math.hypot(v2x, v2y) or 1.0
        cos_a = (v1x * v2x + v1y * v2y) / (n1 * n2)
        # Spike: coming back almost opposite on short edges
        if cos_a < max_turn_cos and min(e1, e2) < min_edge:
            continue
        kept.append(list(cur))

    if len(kept) < 3:
        return poly
    if closed:
        kept.append(list(kept[0]))
    return kept


def _snap_collinear(poly: list[list[float]], tol: float) -> list[list[float]]:
    """Merge nearly collinear consecutive edges (simplify polyline)."""
    if len(poly) < 4:
        return poly
    closed = _dist(poly[0], poly[-1]) < tol
    pts = poly[:-1] if closed else list(poly)
    if len(pts) < 3:
        return poly

    out = [list(pts[0])]
    for i in range(1, len(pts) - 1):
        a = out[-1]
        b = pts[i]
        c = pts[i + 1]
        # Area of triangle / base ≈ deviation from line
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        base = _dist(a, c) or 1.0
        if area2 / base <= tol:
            continue
        out.append(list(b))
    out.append(list(pts[-1]))
    if closed:
        out = _ensure_closed(out, tol)
    return out


def _close_endpoint_gaps(segments: list[list[list[float]]], snap: float) -> list[list[list[float]]]:
    """Snap nearby free endpoints of open 2-point walls to each other / to closed polys."""
    closed: list[list[list[float]]] = []
    opens: list[list[list[float]]] = []
    for poly in segments:
        if len(poly) >= 3 and _is_closed(poly, snap):
            closed.append(poly)
        elif len(poly) >= 2:
            opens.append([list(poly[0]), list(poly[-1])] if len(poly) > 2 else [list(p) for p in poly])

    # Collect snap targets from closed vertices
    anchors: list[list[float]] = []
    for poly in closed:
        anchors.extend(poly[:-1] if _is_closed(poly, snap) else poly)

    for seg in opens:
        for i, pt in enumerate(seg):
            best = None
            best_d = snap
            for a in anchors:
                d = _dist(pt, a)
                if d < best_d:
                    best_d = d
                    best = a
            # Also snap to other open endpoints
            for other in opens:
                if other is seg:
                    continue
                for q in other:
                    d = _dist(pt, q)
                    if d < best_d:
                        best_d = d
                        best = q
            if best is not None:
                seg[i] = list(best)
        closed.append(seg)
    return closed


def check_closedness(
    walls: list[list[list[float]]],
    tol_mm: float = 50.0,
) -> dict[str, Any]:
    """Report outer / room closedness for geometry QA."""
    closed_polys = [w for w in walls if len(w) >= 4 and _is_closed(w, tol_mm)]
    open_segs = [w for w in walls if not (len(w) >= 4 and _is_closed(w, tol_mm))]

    outer = None
    outer_area = -1.0
    for poly in closed_polys:
        # shoelace
        pts = poly[:-1]
        area = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area += x1 * y2 - x2 * y1
        area = abs(area) * 0.5
        if area > outer_area:
            outer_area = area
            outer = poly

    rooms = [p for p in closed_polys if p is not outer]
    return {
        "outer_closed": outer is not None,
        "outer_area_mm2": round(outer_area) if outer is not None else 0,
        "rooms_closed": len(rooms),
        "open_segments": len(open_segs),
        "all_closed_ok": outer is not None and len(open_segs) == 0,
    }


def simplify_polygon(pts: list[list[float]], epsilon_factor: float = 0.002) -> list[list[float]]:
    """Ramer–Douglas–Peucker simplification (OpenCV approxPolyDP)."""
    if len(pts) < 3:
        return [list(p) for p in pts]
    arr = np.array(pts, dtype=np.float32)
    closed = bool(_dist(pts[0], pts[-1]) < 1.0)
    peri = float(cv2.arcLength(arr, closed))
    epsilon = max(1.0, epsilon_factor * peri)
    approx = cv2.approxPolyDP(arr, epsilon, closed)
    out = approx.reshape(-1, 2).tolist()
    if closed and out and _dist(out[0], out[-1]) > 1e-3:
        out.append(list(out[0]))
    return out


def snap_to_orthogonal(poly: list[list[float]], tol_deg: float = 8.0) -> list[list[float]]:
    """
    Snap nearly-horizontal / nearly-vertical edges to exact H/V.
    Removes jagged stair-step pixel contours on CAD plans (fix_3).
    True diagonals (door leaves) are left unchanged.
    """
    if len(poly) < 2:
        return [list(p) for p in poly]

    closed = _dist(poly[0], poly[-1]) < 1e-3
    pts = [list(p) for p in (poly[:-1] if closed else poly)]
    if len(pts) < 2:
        return [list(p) for p in poly]

    out: list[list[float]] = [list(pts[0])]
    for i in range(1, len(pts)):
        x0, y0 = out[-1]
        x1, y1 = pts[i]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        if ang > 90.0:
            ang = 180.0 - ang
        if ang <= tol_deg:
            # horizontal
            out.append([x1, y0])
        elif abs(ang - 90.0) <= tol_deg:
            # vertical
            out.append([x0, y1])
        else:
            out.append([x1, y1])

    if closed and out:
        # Force last→first also ortho when possible
        x0, y0 = out[-1]
        x1, y1 = out[0]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length > 1e-6:
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            if ang > 90.0:
                ang = 180.0 - ang
            if ang <= tol_deg:
                out[-1] = [x0, y1] if abs(dy) < abs(dx) else [x0, y0]
                out[0] = [x1, out[-1][1]]
            elif abs(ang - 90.0) <= tol_deg:
                out[-1] = [x1, y0]
                out[0] = [out[-1][0], y1]
        out.append(list(out[0]))
    return out


def peel_diagonal_door_leaves(
    walls: list[list[list[float]]],
    min_len: float = 200.0,
    max_len: float = 1800.0,
    diag_tol_deg: float = 18.0,
) -> tuple[list[list[list[float]]], list[dict[str, Any]]]:
    """
    Move short diagonal wall segments (GOST door leaves) out of walls → openings.
    Also drops residual short diagonal edges from longer polylines.
    """
    kept: list[list[list[float]]] = []
    doors: list[dict[str, Any]] = []
    for poly in walls:
        if len(poly) == 2:
            p1, p2 = poly[0], poly[1]
            length = _dist(p1, p2)
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            ang = abs(math.degrees(math.atan2(dy, dx))) % 180.0
            if ang > 90.0:
                ang = 180.0 - ang
            if min_len <= length <= max_len and abs(ang - 45.0) <= diag_tol_deg:
                doors.append({"kind": "door", "leaf": [list(p1), list(p2)]})
                continue
            kept.append(poly)
            continue

        # Multi-point: keep only near-ortho edges; peel door-length diagonals
        if len(poly) < 2:
            continue
        closed = _dist(poly[0], poly[-1]) < 1e-3
        pts = poly[:-1] if closed else list(poly)
        new_pts: list[list[float]] = [list(pts[0])]
        for i in range(1, len(pts)):
            a, b = new_pts[-1], pts[i]
            length = _dist(a, b)
            ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180.0
            if ang > 90.0:
                ang = 180.0 - ang
            is_ortho = ang <= 12.0 or abs(ang - 90.0) <= 12.0
            is_door = min_len <= length <= max_len and abs(ang - 45.0) <= diag_tol_deg
            if is_door:
                doors.append({"kind": "door", "leaf": [list(a), list(b)]})
                # Flush current ortho chain; start a new one at the far hinge/leaf tip
                if len(new_pts) >= 2:
                    kept.append(new_pts)
                new_pts = [list(b)]
                continue
            if is_ortho or length < min_len * 0.5:
                new_pts.append(list(b))
            else:
                # Non-ortho junk: break chain
                if len(new_pts) >= 2:
                    kept.append(new_pts)
                new_pts = [list(b)]
                continue
        if len(new_pts) >= 2:
            if closed and len(new_pts) >= 3 and _dist(new_pts[0], new_pts[-1]) > 1e-3:
                # only close if final edge would be ortho
                a, b = new_pts[-1], new_pts[0]
                ang = abs(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))) % 180.0
                if ang > 90.0:
                    ang = 180.0 - ang
                if ang <= 12.0 or abs(ang - 90.0) <= 12.0:
                    new_pts.append(list(new_pts[0]))
            kept.append(new_pts)
    return kept, doors


def _filter_tiny_wall_fragments(
    walls: list[list[list[float]]],
    min_length: float,
    min_box_span: float,
) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    for poly in walls:
        if len(poly) < 2:
            continue
        span_x0, span_y0, span_x1, span_y1 = _bbox(poly)
        span = max(span_x1 - span_x0, span_y1 - span_y0)
        if len(poly) == 2 and _segment_length(poly) < min_length:
            continue
        if span < min_box_span and _segment_length(poly) < min_length * 1.5:
            continue
        out.append(poly)
    return out


def _dedupe_wall_overlaps(
    walls: list[list[list[float]]],
    overlap_iou: float = 0.82,
    endpoint_tol: float = 80.0,
) -> list[list[list[float]]]:
    kept: list[list[list[float]]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for poly in sorted(walls, key=_segment_length, reverse=True):
        box = _bbox(poly)
        duplicate = False
        for prev, prev_box in zip(kept, boxes):
            if _bbox_iou(box, prev_box) >= overlap_iou:
                duplicate = True
                break
            if len(poly) == 2 and len(prev) == 2:
                if _dist(poly[0], prev[0]) <= endpoint_tol and _dist(poly[1], prev[1]) <= endpoint_tol:
                    duplicate = True
                    break
                if _dist(poly[0], prev[1]) <= endpoint_tol and _dist(poly[1], prev[0]) <= endpoint_tol:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(poly)
            boxes.append(box)
    return kept


def wall_polygons_to_lines(
    walls: list[list[list[float]]],
    min_thickness: float = 80.0,
    max_thickness: float = 600.0,
) -> list[dict[str, Any]]:
    """
    Convert thick wall polygons to centerline segments + thickness (PCA axis).
    Thin / oversized blobs are skipped.
    """
    lines: list[dict[str, Any]] = []
    for poly in walls:
        if len(poly) < 3:
            continue
        pts = np.array(poly[:-1] if _is_closed(poly, 1.0) else poly, dtype=np.float64)
        if len(pts) < 3:
            continue
        mean = pts.mean(axis=0)
        centered = pts - mean
        try:
            cov = np.cov(centered.T)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        main_axis = eig_vecs[:, int(np.argmax(eig_vals))]
        proj = centered @ main_axis
        min_t, max_t = float(np.min(proj)), float(np.max(proj))
        if max_t - min_t < 200:
            continue
        perp = np.array([-main_axis[1], main_axis[0]])
        perp_proj = centered @ perp
        thickness = float(np.max(perp_proj) - np.min(perp_proj))
        if not (min_thickness <= thickness <= max_thickness):
            # Still emit as open polyline outline when thickness filter fails
            continue
        p1 = (mean + min_t * main_axis).tolist()
        p2 = (mean + max_t * main_axis).tolist()
        lines.append({"p1": p1, "p2": p2, "thickness": thickness})
    return lines


def postprocess_walls(
    walls: list[list[list[float]]],
    partitions: list[list[list[float]]] | None = None,
    *,
    snap_tol_deg: float = 12.0,
    min_fragment_factor: float = 0.012,
    overlap_iou: float = 0.82,
) -> tuple[list[list[list[float]]], list[list[list[float]]], dict[str, Any]]:
    """
    Smooth closed contours, snap gaps, return cleaned walls/partitions + QA report.
    """
    partitions = list(partitions or [])
    if not walls:
        return walls, partitions, check_closedness([])

    # Characteristic size for tolerances
    xs = [p[0] for w in walls for p in w]
    ys = [p[1] for w in walls for p in w]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    snap = max(30.0, span * 0.008)
    min_edge = max(80.0, span * 0.015)
    col_tol = max(20.0, span * 0.004)
    min_fragment_len = max(120.0, span * min_fragment_factor)
    min_box_span = max(60.0, span * 0.006)

    cleaned: list[list[list[float]]] = []
    for poly in walls:
        if len(poly) >= 4 and (_is_closed(poly, snap) or len(poly) > 4):
            p = _ensure_closed(poly, snap)
            p = simplify_polygon(p, epsilon_factor=0.006)
            p = _remove_spikes(p, min_edge=min_edge)
            p = snap_to_orthogonal(p, tol_deg=max(snap_tol_deg + 3.0, snap_tol_deg))
            p = _snap_collinear(p, tol=col_tol)
            p = snap_to_orthogonal(p, tol_deg=snap_tol_deg)
            p = _ensure_closed(p, snap)
            cleaned.append(p)
        elif len(poly) >= 2:
            p = snap_to_orthogonal([list(q) for q in poly], tol_deg=max(snap_tol_deg + 3.0, snap_tol_deg))
            cleaned.append(p)
        else:
            cleaned.append([list(p) for p in poly])

    cleaned = _close_endpoint_gaps(cleaned, snap=snap)
    cleaned = _filter_tiny_wall_fragments(cleaned, min_length=min_fragment_len, min_box_span=min_box_span)
    cleaned = _dedupe_wall_overlaps(cleaned, overlap_iou=overlap_iou, endpoint_tol=snap)
    # Peel door leaves after ortho snap (diagonals survive snap)
    cleaned, door_leaves = peel_diagonal_door_leaves(cleaned)

    # Partitions: only endpoint snap
    parts_out: list[list[list[float]]] = []
    for poly in partitions:
        if len(poly) == 2:
            parts_out.append([list(poly[0]), list(poly[1])])
        else:
            parts_out.append([list(p) for p in poly])
    if parts_out:
        # snap partition ends to wall vertices
        anchors = [p for w in cleaned for p in (w[:-1] if _is_closed(w, snap) else w)]
        for seg in parts_out:
            for i, pt in enumerate(seg):
                best, best_d = None, snap
                for a in anchors:
                    d = _dist(pt, a)
                    if d < best_d:
                        best_d, best = d, a
                if best is not None:
                    seg[i] = list(best)

    report = check_closedness(cleaned, tol_mm=snap)
    report["snap_mm"] = round(snap, 1)
    report["postprocessed"] = True
    report["orthogonal_snap"] = True
    report["door_leaves_peeled"] = len(door_leaves)
    report["tiny_fragments_filtered"] = True
    report["wall_overlap_dedup"] = True
    report["centerlines"] = len(wall_polygons_to_lines(cleaned))
    report["door_leaves"] = door_leaves
    return cleaned, parts_out, report
