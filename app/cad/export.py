"""Assemble DXF: true-mm geometry in modelspace + GOST frame on paperspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ezdxf

from app.cad.dimensions import add_gost_dimension
from app.cad.gost import LAYER_NAMES
from app.cad.stamp import add_gost_stamp_and_frame, ensure_gost_style, ensure_layers, pick_sheet_format


def _offset_pair(
    p1: tuple[float, float],
    p2: tuple[float, float],
    half_t: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return two parallel edges (a1-a2 and b1-b2) for a thick wall segment."""
    import math

    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length * half_t, dx / length * half_t
    return (
        (x1 + nx, y1 + ny),
        (x2 + nx, y2 + ny),
        (x1 - nx, y1 - ny),
        (x2 - nx, y2 - ny),
    )


def _add_thick_segment(msp: Any, p1: tuple[float, float], p2: tuple[float, float], half_t: float, layer: str) -> None:
    a1, a2, b1, b2 = _offset_pair(p1, p2, half_t)
    msp.add_line(a1, a2, dxfattribs={"layer": layer})
    msp.add_line(b1, b2, dxfattribs={"layer": layer})
    msp.add_line(a1, b1, dxfattribs={"layer": layer})
    msp.add_line(a2, b2, dxfattribs={"layer": layer})


def build_dxf(geometry: dict[str, Any], metadata: dict[str, Any], out_path: Path) -> Path:
    # R2000 DXF — widely compatible (AutoCAD, LibreCAD, etc.)
    doc = ezdxf.new("R2000")
    ensure_layers(doc)
    ensure_gost_style(doc)
    msp = doc.modelspace()

    walls = geometry.get("walls") or []
    xs = [p[0] for wall in walls for p in wall] or [0, 12000]
    ys = [p[1] for wall in walls for p in wall] or [0, 8000]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    content_w = max(max_x - min_x, 1.0)
    content_h = max(max_y - min_y, 1.0)

    # --- Modelspace: real millimetres (do NOT shrink into A3 paper) ---
    # Outer closed walls: double line (thickness). Internal: centerlines only
    # (drawing every segment as a thick box made the preview chaotic).
    wall_t = max(80.0, min(content_w, content_h) * 0.012)

    for wall in walls:
        pts = [(float(p[0]), float(p[1])) for p in wall]
        if len(pts) >= 3 and pts[0] == pts[-1]:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["walls"]})
            # Parallel inset for wall thickness (rectangular envelope only)
            xs_ = [p[0] for p in pts[:-1]]
            ys_ = [p[1] for p in pts[:-1]]
            if xs_ and ys_:
                ix0, ix1 = min(xs_) + wall_t, max(xs_) - wall_t
                iy0, iy1 = min(ys_) + wall_t, max(ys_) - wall_t
                if ix1 > ix0 and iy1 > iy0:
                    msp.add_lwpolyline(
                        [(ix0, iy0), (ix1, iy0), (ix1, iy1), (ix0, iy1), (ix0, iy0)],
                        dxfattribs={"layer": LAYER_NAMES["walls"]},
                    )
        elif len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["walls"]})

    for part in geometry.get("partitions") or []:
        pts = [(float(p[0]), float(p[1])) for p in part]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["partitions"]})

    for opening in geometry.get("openings") or []:
        if "p1" not in opening or "p2" not in opening:
            continue
        x0, x1 = sorted((float(opening["p1"][0]), float(opening["p2"][0])))
        y0, y1 = sorted((float(opening["p1"][1]), float(opening["p2"][1])))
        msp.add_lwpolyline(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
            dxfattribs={"layer": LAYER_NAMES["openings"]},
        )
        if str(opening.get("type", "")).lower() == "door":
            msp.add_line((x0, y0), (x1, y1), dxfattribs={"layer": LAYER_NAMES["openings"]})

    for axis in geometry.get("axes") or []:
        if "x" in axis:
            x = float(axis["x"])
            p1 = (x, float(axis.get("y0", min_y)))
            p2 = (x, float(axis.get("y1", max_y)))
            msp.add_line(p1, p2, dxfattribs={"layer": LAYER_NAMES["axes"], "linetype": "CENTER"})
            # Prefer ASCII in R2000 DXF (Cyrillic becomes \\U+XXXX otherwise)
            name = str(axis.get("name", ""))
            latin = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E"}.get(name, name)
            t = msp.add_text(
                latin,
                height=max(200.0, content_h * 0.015),
                dxfattribs={"layer": LAYER_NAMES["axes"], "style": "GOST_ST"},
            )
            t.set_placement((p2[0], p2[1] + 100))
        elif "y" in axis:
            y = float(axis["y"])
            p1 = (float(axis.get("x0", min_x)), y)
            p2 = (float(axis.get("x1", max_x)), y)
            msp.add_line(p1, p2, dxfattribs={"layer": LAYER_NAMES["axes"], "linetype": "CENTER"})
            # Prefer ASCII in R2000 DXF (Cyrillic becomes \\U+XXXX otherwise)
            name = str(axis.get("name", ""))
            latin = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E"}.get(name, name)
            t = msp.add_text(
                latin,
                height=max(200.0, content_h * 0.015),
                dxfattribs={"layer": LAYER_NAMES["axes"], "style": "GOST_ST"},
            )
            t.set_placement((p1[0] - 400, p1[1]))

    # Overall GOST dims only (OCR sizes are plain text labels — avoids chaotic dim chains)
    for dim in geometry.get("dimensions") or []:
        if dim.get("role") not in {"overall_x", "overall_y"}:
            continue
        p1 = (float(dim["p1"][0]), float(dim["p1"][1]))
        p2 = (float(dim["p2"][0]), float(dim["p2"][1]))
        offset = max(300.0, min(content_w, content_h) * 0.03)
        add_gost_dimension(msp, p1, p2, value_mm=dim.get("value_mm"), offset=offset)

    for label in geometry.get("texts") or []:
        pos = label.get("pos") or [0, 0]
        t = msp.add_text(
            str(label.get("text", "")),
            height=float(label.get("height") or max(120.0, content_h * 0.012)),
            dxfattribs={"layer": LAYER_NAMES["dims"], "style": "GOST_ST"},
        )
        t.set_placement((float(pos[0]), float(pos[1])))

    # --- Paperspace: GOST frame / stamp for plotting ---
    fmt, orient, sheet_w, sheet_h = pick_sheet_format(400, 300)
    try:
        layout = doc.paperspace()
    except Exception:  # noqa: BLE001
        layout = doc.layout("Layout1")
    add_gost_stamp_and_frame(doc, layout, format_name=fmt, orientation=orient, metadata=metadata)
    layout.add_text(
        "Модель: размеры в мм (1:1). Рамка — пространство листа.",
        height=2.5,
        dxfattribs={"layer": LAYER_NAMES["stamp_text"], "style": "GOST_ST"},
    ).set_placement((25, sheet_h - 10))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_path)
    return out_path
