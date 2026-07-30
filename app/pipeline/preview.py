"""Interactive preview payload: contour metrics for live client-side filtering."""

from __future__ import annotations

import math
from typing import Any


DEFAULT_FILTERS: dict[str, float] = {
    "med_t_min": 0.55,
    "thinness_min": 40.0,
    "thinness_max": 320.0,
    "aspect_min": 1.2,
    "ink_ratio_min": 0.15,
    "min_length": 200.0,
}


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def calc_perimeter(poly: list[list[float]]) -> float:
    if len(poly) < 2:
        return 0.0
    return sum(_dist(poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def calc_area(poly: list[list[float]]) -> float:
    if len(poly) < 3:
        return 0.0
    pts = poly[:-1] if _dist(poly[0], poly[-1]) < 1e-6 else poly
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def calc_aspect(poly: list[list[float]]) -> float:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return max(w, h) / max(min(w, h), 1e-6)


def calc_thinness(poly: list[list[float]]) -> float:
    area = calc_area(poly)
    peri = calc_perimeter(poly)
    if area < 1e-6:
        # Open segment: neutral thinness so length/aspect filters dominate
        return 80.0 if peri > 0 else 0.0
    return (peri * peri) / area


def calc_med_t(poly: list[list[float]]) -> float:
    """
    Approximate stroke thickness from geometry.
    Closed ring ≈ area/perimeter; open segment ≈ small constant proxy.
    """
    peri = calc_perimeter(poly)
    area = calc_area(poly)
    if peri < 1e-6:
        return 0.0
    if area < 1e-6:
        return 0.45
    return max(0.05, min(8.0, (2.0 * area) / peri))


def calc_ink_ratio(poly: list[list[float]]) -> float:
    """Proxy fill density from compactness (not true raster ink)."""
    area = calc_area(poly)
    peri = calc_perimeter(poly)
    if peri < 1e-6:
        return 0.0
    if area < 1e-6:
        return 0.35
    # circle-like fill denser; elongated walls lower
    compact = (4.0 * math.pi * area) / max(peri * peri, 1e-6)
    return float(max(0.05, min(1.0, 0.25 + compact)))


def contour_properties(poly: list[list[float]]) -> dict[str, float]:
    peri = calc_perimeter(poly)
    area = calc_area(poly)
    return {
        "area": round(area, 3),
        "perimeter": round(peri, 3),
        "length": round(peri, 3),
        "thinness": round(calc_thinness(poly), 3),
        "aspect": round(calc_aspect(poly), 3),
        "med_t": round(calc_med_t(poly), 3),
        "ink_ratio": round(calc_ink_ratio(poly), 3),
        "n_points": float(len(poly)),
    }


def _bounds_from_points(points: list[list[float]]) -> dict[str, list[float]]:
    if not points:
        return {"min": [0.0, 0.0], "max": [12000.0, 8000.0]}
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return {"min": [min(xs), min(ys)], "max": [max(xs), max(ys)]}


def build_preview_payload(geometry: dict[str, Any]) -> dict[str, Any]:
    """Build client preview JSON: filterable contours + static annotations."""
    contours: list[dict[str, Any]] = []
    all_pts: list[list[float]] = []

    def add_group(key: str, default_kind: str, included: bool) -> None:
        for idx, poly in enumerate(geometry.get(key) or []):
            if not isinstance(poly, list) or len(poly) < 2:
                continue
            pts = [[float(p[0]), float(p[1])] for p in poly]
            props = contour_properties(pts)
            all_pts.extend(pts)
            contours.append(
                {
                    "id": f"{key}_{idx}",
                    "source": key,
                    "kind": default_kind,
                    "included_default": included,
                    "points": pts,
                    "properties": props,
                }
            )

    add_group("walls", "wall", True)
    add_group("partitions", "partition", True)
    add_group("details", "detail", False)
    add_group("hatches", "hatch", False)
    add_group("strokes", "stroke", False)

    profile = str(geometry.get("vectorization_profile") or "default")
    filters = dict(DEFAULT_FILTERS)
    # Soft defaults by profile
    if profile == "thin_lines":
        filters.update({"med_t_min": 0.25, "thinness_max": 450.0, "min_length": 150.0})
    elif profile == "noisy":
        filters.update({"med_t_min": 0.7, "thinness_min": 50.0, "min_length": 320.0, "aspect_min": 1.3})
    elif profile == "thick_lines":
        filters.update({"med_t_min": 0.9, "thinness_min": 30.0})
    elif profile == "high_detail":
        filters.update({"med_t_min": 0.35, "thinness_min": 30.0, "aspect_min": 1.1, "min_length": 120.0})

    return {
        "mode": "preview",
        "units": geometry.get("units") or "mm",
        "vectorization_profile": profile,
        "filters_default": filters,
        "contours": contours,
        "openings": geometry.get("openings") or [],
        "axes": geometry.get("axes") or [],
        "dimensions": geometry.get("dimensions") or [],
        "texts": geometry.get("texts") or [],
        "circles": geometry.get("circles") or [],
        "bounds": _bounds_from_points(all_pts),
        "counts": {
            "contours": len(contours),
            "walls_default": sum(1 for c in contours if c["included_default"] and c["kind"] == "wall"),
        },
    }


def contour_passes_filters(props: dict[str, Any], filters: dict[str, float]) -> bool:
    med_t = float(props.get("med_t") or 0.0)
    thinness = float(props.get("thinness") or 0.0)
    aspect = float(props.get("aspect") or 0.0)
    ink = float(props.get("ink_ratio") or 0.0)
    length = float(props.get("length") or props.get("perimeter") or 0.0)
    area = float(props.get("area") or 0.0)
    # Open line segments: skip density/thinness max (geometry-only)
    if area < 1e-3:
        return (
            aspect >= float(filters.get("aspect_min", DEFAULT_FILTERS["aspect_min"]))
            and length >= float(filters.get("min_length", DEFAULT_FILTERS["min_length"]))
            and med_t >= float(filters.get("med_t_min", DEFAULT_FILTERS["med_t_min"])) * 0.5
        )
    return (
        med_t >= float(filters.get("med_t_min", DEFAULT_FILTERS["med_t_min"]))
        and thinness >= float(filters.get("thinness_min", DEFAULT_FILTERS["thinness_min"]))
        and thinness <= float(filters.get("thinness_max", DEFAULT_FILTERS["thinness_max"]))
        and aspect >= float(filters.get("aspect_min", DEFAULT_FILTERS["aspect_min"]))
        and ink >= float(filters.get("ink_ratio_min", DEFAULT_FILTERS["ink_ratio_min"]))
        and length >= float(filters.get("min_length", DEFAULT_FILTERS["min_length"]))
    )


def apply_filters_to_geometry(
    geometry: dict[str, Any],
    filters: dict[str, float] | None = None,
    preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a geometry copy with walls replaced by filter-passing contours."""
    out = dict(geometry)
    f = dict(DEFAULT_FILTERS)
    if filters:
        f.update({k: float(v) for k, v in filters.items() if v is not None})

    payload = preview or build_preview_payload(geometry)
    walls: list[list[list[float]]] = []
    details: list[list[list[float]]] = []
    hatches: list[list[list[float]]] = []
    for c in payload.get("contours") or []:
        props = c.get("properties") or {}
        pts = c.get("points") or []
        if len(pts) < 2:
            continue
        if contour_passes_filters(props, f):
            walls.append(pts)
        elif c.get("source") == "hatches":
            hatches.append(pts)
        elif c.get("source") in {"details", "strokes"}:
            details.append(pts)

    out["walls"] = walls
    out["details"] = details
    out["hatches"] = hatches
    out["preview_filters"] = f
    out["preview_wall_count"] = len(walls)
    return out
