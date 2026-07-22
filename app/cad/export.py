"""Assemble DXF document from geometry + GOST frame/stamp."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ezdxf

from app.cad.dimensions import add_gost_dimension
from app.cad.gost import LAYER_NAMES
from app.cad.stamp import add_gost_stamp_and_frame, ensure_gost_style, ensure_layers, pick_sheet_format


def build_dxf(geometry: dict[str, Any], metadata: dict[str, Any], out_path: Path) -> Path:
    # R2000: best LibreDWG encode compatibility for DXF→DWG
    doc = ezdxf.new("R2000")
    ensure_layers(doc)
    ensure_gost_style(doc)
    msp = doc.modelspace()

    walls = geometry.get("walls") or []
    xs = [p[0] for wall in walls for p in wall] or [0, 12000]
    ys = [p[1] for wall in walls for p in wall] or [0, 8000]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    content_w = max_x - min_x
    content_h = max_y - min_y

    fmt, orient, sheet_w, sheet_h = pick_sheet_format(content_w, content_h)
    add_gost_stamp_and_frame(doc, msp, format_name=fmt, orientation=orient, metadata=metadata)

    # Place drawing content above the stamp with margins inside inner frame
    origin_x = 20 + 15
    origin_y = 5 + 55 + 20
    # Fit content into drawable area
    drawable_w = sheet_w - 5 - origin_x - 15
    drawable_h = sheet_h - 5 - origin_y - 15
    scale = 1.0
    if content_w > 0 and content_h > 0:
        scale = min(drawable_w / content_w, drawable_h / content_h, 1.0)

    def map_pt(x: float, y: float) -> tuple[float, float]:
        return origin_x + (x - min_x) * scale, origin_y + (y - min_y) * scale

    for wall in walls:
        pts = [map_pt(p[0], p[1]) for p in wall]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["walls"]})

    for axis in geometry.get("axes") or []:
        if "x" in axis:
            x = axis["x"]
            p1 = map_pt(x, axis.get("y0", min_y))
            p2 = map_pt(x, axis.get("y1", max_y))
            msp.add_line(p1, p2, dxfattribs={"layer": LAYER_NAMES["axes"], "linetype": "CENTER"})
            t = msp.add_text(str(axis.get("name", "")), height=3.0, dxfattribs={"layer": LAYER_NAMES["axes"], "style": "GOST_ST"})
            t.set_placement((p2[0], p2[1] + 5))
        elif "y" in axis:
            y = axis["y"]
            p1 = map_pt(axis.get("x0", min_x), y)
            p2 = map_pt(axis.get("x1", max_x), y)
            msp.add_line(p1, p2, dxfattribs={"layer": LAYER_NAMES["axes"], "linetype": "CENTER"})
            t = msp.add_text(str(axis.get("name", "")), height=3.0, dxfattribs={"layer": LAYER_NAMES["axes"], "style": "GOST_ST"})
            t.set_placement((p1[0] - 10, p1[1]))

    for dim in geometry.get("dimensions") or []:
        p1 = map_pt(dim["p1"][0], dim["p1"][1])
        p2 = map_pt(dim["p2"][0], dim["p2"][1])
        add_gost_dimension(msp, p1, p2, value_mm=dim.get("value_mm"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(out_path)
    return out_path
