"""Stage 4: mm/px from OCR dimension labels (+ optional wall polygons)."""

from __future__ import annotations

from typing import Any

DEFAULT_SPAN_MM = 12000.0


def _wall_segments_from_polys(
    wall_polygons: list[list[list[float]]],
) -> list[tuple[float, float, float, float]]:
    """Turn wall polygons into edge segments for chain matching."""
    segs: list[tuple[float, float, float, float]] = []
    for poly in wall_polygons:
        if len(poly) < 2:
            continue
        n = len(poly)
        closed = (
            abs(poly[0][0] - poly[-1][0]) < 1.5
            and abs(poly[0][1] - poly[-1][1]) < 1.5
        )
        limit = n - 1 if closed or n > 2 else 1
        for i in range(limit):
            a = poly[i]
            b = poly[(i + 1) % n] if closed else poly[min(i + 1, n - 1)]
            segs.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    # Also add overall bbox edges (overall dimension chains)
    xs = [p[0] for poly in wall_polygons for p in poly]
    ys = [p[1] for poly in wall_polygons for p in poly]
    if xs and ys:
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        segs.extend(
            [
                (x0, y0, x1, y0),
                (x1, y0, x1, y1),
                (x1, y1, x0, y1),
                (x0, y1, x0, y0),
            ]
        )
    return segs


def calibrate_scale(
    dimensions: list[dict[str, Any]],
    wall_polygons: list[list[list[float]]],
    orig_w: int,
    orig_h: int,
) -> dict[str, Any] | None:
    """
    Prefer OCR label ↔ wall/dim-chain length. Falls back to bbox mapping (weak).
    Returns None if no usable OCR sizes (caller must warn / ask control length).
    """
    if not dimensions:
        return None

    from app.pipeline.ocr_dims import calibrate_span_mm

    xs = [p[0] for poly in wall_polygons for p in poly] or [0.0, float(orig_w)]
    ys = [p[1] for poly in wall_polygons for p in poly] or [0.0, float(orig_h)]
    bbox_w = max(max(xs) - min(xs), 1.0)
    bbox_h = max(max(ys) - min(ys), 1.0)
    segs = _wall_segments_from_polys(wall_polygons)

    # Labels already in original pixel space
    labels = []
    for d in dimensions:
        labels.append(
            {
                "value_mm": int(d["value_mm"]),
                "cx": float(d["cx"]),
                "cy": float(d["cy"]),
                "conf": float(d.get("conf") or 0.5),
                "text": d.get("text"),
                "is_area": False,
                "ocr_scale": 1.0,
            }
        )

    calib = calibrate_span_mm(
        labels,
        bbox_w_px=bbox_w,
        bbox_h_px=bbox_h,
        segments_px=segs or None,
        ocr_to_proc=1.0,
    )
    if not calib:
        return None

    mm_x = float(calib["mm_per_px_x"])
    mm_y = float(calib["mm_per_px_y"])
    return {
        "mm_per_px": (mm_x + mm_y) / 2.0,
        "mm_per_px_x": mm_x,
        "mm_per_px_y": mm_y,
        "source": calib.get("scale_source", "ocr"),
        "confidence": float(calib.get("confidence") or 0.5),
        "ocr_values": calib.get("ocr_values") or [],
        "warning": calib.get("warning"),
        "from_chains": bool(calib.get("from_chains")),
    }


def provisional_scale(orig_w: int, orig_h: int) -> dict[str, Any]:
    """Conditional 12 000 mm span — always flagged for user confirmation."""
    long_px = max(orig_w, orig_h, 1)
    mm_per_px = DEFAULT_SPAN_MM / long_px
    return {
        "mm_per_px": mm_per_px,
        "mm_per_px_x": mm_per_px,
        "mm_per_px_y": mm_per_px,
        "source": "provisional",
        "confidence": 0.2,
        "ocr_values": [],
        "warning": (
            "Размерные надписи не найдены. Применён условный масштаб 12 000 мм — "
            "укажите контрольный отрезок."
        ),
        "from_chains": False,
    }
