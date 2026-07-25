"""Post-process wall contours: smooth spikes, close gaps, verify closedness."""

from __future__ import annotations

import math
from typing import Any


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def postprocess_walls(
    walls: list[list[list[float]]],
    partitions: list[list[list[float]]] | None = None,
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

    cleaned: list[list[list[float]]] = []
    for poly in walls:
        if len(poly) >= 4 and (_is_closed(poly, snap) or len(poly) > 4):
            p = _ensure_closed(poly, snap)
            p = _remove_spikes(p, min_edge=min_edge)
            p = _snap_collinear(p, tol=col_tol)
            p = _ensure_closed(p, snap)
            cleaned.append(p)
        else:
            cleaned.append([list(p) for p in poly])

    cleaned = _close_endpoint_gaps(cleaned, snap=snap)

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
    return cleaned, parts_out, report
