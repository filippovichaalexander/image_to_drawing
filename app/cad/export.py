"""Assemble DXF: true-mm geometry in modelspace + GOST frame on paperspace."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import ezdxf
from ezdxf.enums import TextEntityAlignment

from app.cad.dimensions import add_gost_dimension
from app.cad.gost import LAYER_NAMES
from app.cad.stamp import add_gost_stamp_and_frame, ensure_gost_style, ensure_layers, pick_sheet_format


def _latin_axis(name: str) -> str:
    return {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E"}.get(name, name)


def _add_axis_bubble(msp: Any, cx: float, cy: float, r: float, name: str, content_h: float) -> None:
    layer = LAYER_NAMES["axes"]
    msp.add_circle((cx, cy), radius=max(r, 120.0), dxfattribs={"layer": layer})
    if not name:
        return
    t = msp.add_text(
        _latin_axis(name),
        height=max(r * 0.9, content_h * 0.012, 140.0),
        dxfattribs={"layer": layer, "style": "GOST_ST"},
    )
    t.set_placement((cx, cy), align=TextEntityAlignment.MIDDLE_CENTER)


def _add_wall_hatch(msp: Any, pts: list[tuple[float, float]]) -> None:
    """Solid-ish hatch for closed wall polygons (visual thickness)."""
    if len(pts) < 4:
        return
    try:
        hatch = msp.add_hatch(color=1, dxfattribs={"layer": LAYER_NAMES["walls"]})
        hatch.paths.add_polyline_path(pts, is_closed=True)
        hatch.set_solid_fill(color=1)
    except Exception:  # noqa: BLE001
        # Older / restricted DXF profiles — outline already drawn
        pass


def _add_door_arc(msp: Any, arc: dict[str, Any]) -> None:
    cx = float(arc["cx"])
    cy = float(arc["cy"])
    r = float(arc["r"])
    start = float(arc.get("start_deg", 0.0))
    end = float(arc.get("end_deg", 90.0))
    if abs(end - start) < 1.0:
        end = start + 90.0
    msp.add_arc(
        center=(cx, cy),
        radius=max(r, 50.0),
        start_angle=start,
        end_angle=end,
        dxfattribs={"layer": LAYER_NAMES["openings"]},
    )
    # Door leaf as radius line at start angle
    a0 = math.radians(start)
    msp.add_line(
        (cx, cy),
        (cx + r * math.cos(a0), cy + r * math.sin(a0)),
        dxfattribs={"layer": LAYER_NAMES["openings"]},
    )


def build_dxf(geometry: dict[str, Any], metadata: dict[str, Any], out_path: Path) -> Path:
    # R2000 DXF — widely compatible (AutoCAD, LibreCAD, etc.)
    doc = ezdxf.new("R2000")
    ensure_layers(doc)
    ensure_gost_style(doc)
    msp = doc.modelspace()

    walls = geometry.get("walls") or []
    strokes = geometry.get("strokes") or []
    details = geometry.get("details") or []
    xs = [p[0] for poly in (walls + strokes + details) for p in poly] or [0, 12000]
    ys = [p[1] for poly in (walls + strokes + details) for p in poly] or [0, 8000]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    content_w = max(max_x - min_x, 1.0)
    content_h = max(max_y - min_y, 1.0)
    fill_walls = bool(geometry.get("fill_walls"))

    # --- Walls as polylines (+ optional hatch fill for thickness) ---
    for wall in walls:
        pts = [(float(p[0]), float(p[1])) for p in wall]
        if len(pts) >= 3:
            closed = pts[0] == pts[-1] or (
                abs(pts[0][0] - pts[-1][0]) < 1e-3 and abs(pts[0][1] - pts[-1][1]) < 1e-3
            )
            msp.add_lwpolyline(
                pts,
                close=closed,
                dxfattribs={"layer": LAYER_NAMES["walls"]},
            )
            if fill_walls and closed and len(pts) >= 4:
                _add_wall_hatch(msp, pts[:-1] if pts[0] == pts[-1] else pts)
        elif len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["walls"]})

    for part in geometry.get("partitions") or []:
        pts = [(float(p[0]), float(p[1])) for p in part]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["partitions"]})

    for poly in details:
        pts = [(float(p[0]), float(p[1])) for p in poly]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["detail"]})

    for poly in strokes:
        pts = [(float(p[0]), float(p[1])) for p in poly]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["detail"]})

    for poly in geometry.get("stairs") or []:
        pts = [(float(p[0]), float(p[1])) for p in poly]
        if len(pts) >= 2:
            msp.add_lwpolyline(
                pts,
                close=len(pts) >= 3,
                dxfattribs={"layer": LAYER_NAMES.get("stairs", LAYER_NAMES["detail"])},
            )

    for poly in geometry.get("hatches") or []:
        pts = [(float(p[0]), float(p[1])) for p in poly]
        if len(pts) >= 2:
            msp.add_lwpolyline(pts, dxfattribs={"layer": LAYER_NAMES["hatch"]})

    # Standalone circles (axis bubbles / fixtures)
    for circ in geometry.get("circles") or []:
        msp.add_circle(
            (float(circ["cx"]), float(circ["cy"])),
            radius=max(float(circ.get("r") or 100.0), 20.0),
            dxfattribs={"layer": LAYER_NAMES.get(str(circ.get("layer") or "axes"), LAYER_NAMES["axes"])},
        )

    for opening in geometry.get("openings") or []:
        kind = str(opening.get("kind") or opening.get("type") or "").lower()
        if opening.get("arc"):
            _add_door_arc(msp, opening["arc"])
            continue
        if opening.get("leaf") and len(opening["leaf"]) >= 2:
            a, b = opening["leaf"][0], opening["leaf"][-1]
            msp.add_line(
                (float(a[0]), float(a[1])),
                (float(b[0]), float(b[1])),
                dxfattribs={"layer": LAYER_NAMES["openings"]},
            )
            continue
        if "p1" not in opening or "p2" not in opening:
            continue
        x1, y1 = float(opening["p1"][0]), float(opening["p1"][1])
        x2, y2 = float(opening["p2"][0]), float(opening["p2"][1])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1.0
        ux, uy = dx / length, dy / length
        # Thickness across the opening (wall depth), not a fat YOLO AABB
        width = float(opening.get("width_mm") or 0.0)
        if width < 40:
            width = 120.0 if kind == "window" else 80.0
        half = width * 0.5
        nx, ny = -uy, ux
        # Rectangle along centerline
        corners = [
            (x1 + nx * half, y1 + ny * half),
            (x2 + nx * half, y2 + ny * half),
            (x2 - nx * half, y2 - ny * half),
            (x1 - nx * half, y1 - ny * half),
            (x1 + nx * half, y1 + ny * half),
        ]
        msp.add_lwpolyline(corners, dxfattribs={"layer": LAYER_NAMES["openings"]})
        if kind == "door":
            # Swing arc from hinge at p1
            r = max(length, 200.0)
            msp.add_arc(
                center=(x1, y1),
                radius=r,
                start_angle=0,
                end_angle=90,
                dxfattribs={"layer": LAYER_NAMES["openings"]},
            )
            msp.add_line((x1, y1), (x1 + r * ux, y1 + r * uy), dxfattribs={"layer": LAYER_NAMES["openings"]})
        elif kind == "window":
            # Triple line along the opening (GOST-ish window symbol)
            for frac in (-0.35, 0.0, 0.35):
                ox, oy = nx * half * frac, ny * half * frac
                msp.add_line(
                    (x1 + ox, y1 + oy),
                    (x2 + ox, y2 + oy),
                    dxfattribs={"layer": LAYER_NAMES["openings"]},
                )

    for axis in geometry.get("axes") or []:
        name = str(axis.get("name", ""))
        bubble = axis.get("bubble")
        if "x" in axis:
            x = float(axis["x"])
            p1 = (x, float(axis.get("y0", min_y - 1500)))
            p2 = (x, float(axis.get("y1", max_y + 1500)))
            msp.add_line(p1, p2, dxfattribs={"layer": LAYER_NAMES["axes"], "linetype": "CENTER"})
            if bubble:
                _add_axis_bubble(
                    msp,
                    float(bubble.get("cx", x)),
                    float(bubble.get("cy", p2[1])),
                    float(bubble.get("r") or max(180.0, content_h * 0.012)),
                    name,
                    content_h,
                )
            else:
                # Default GOST bubble at both ends
                br = max(180.0, content_h * 0.012)
                _add_axis_bubble(msp, x, p2[1] + br * 1.2, br, name, content_h)
                _add_axis_bubble(msp, x, p1[1] - br * 1.2, br, name, content_h)
        elif "y" in axis:
            y = float(axis["y"])
            p1 = (float(axis.get("x0", min_x - 1500)), y)
            p2 = (float(axis.get("x1", max_x + 1500)), y)
            msp.add_line(p1, p2, dxfattribs={"layer": LAYER_NAMES["axes"], "linetype": "CENTER"})
            if bubble:
                _add_axis_bubble(
                    msp,
                    float(bubble.get("cx", p1[0])),
                    float(bubble.get("cy", y)),
                    float(bubble.get("r") or max(180.0, content_h * 0.012)),
                    name,
                    content_h,
                )
            else:
                br = max(180.0, content_h * 0.012)
                _add_axis_bubble(msp, p1[0] - br * 1.2, y, br, name, content_h)
                _add_axis_bubble(msp, p2[0] + br * 1.2, y, br, name, content_h)
        elif bubble:
            _add_axis_bubble(
                msp,
                float(bubble["cx"]),
                float(bubble["cy"]),
                float(bubble.get("r") or 180.0),
                name,
                content_h,
            )

    # Export ALL dimension roles with GOST extension lines + 45° ticks
    for dim in geometry.get("dimensions") or []:
        if "p1" not in dim or "p2" not in dim:
            continue
        p1 = (float(dim["p1"][0]), float(dim["p1"][1]))
        p2 = (float(dim["p2"][0]), float(dim["p2"][1]))
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) < 50:
            continue
        role = str(dim.get("role") or "chain")
        if role in {"overall_x", "overall_y"}:
            offset = max(350.0, min(content_w, content_h) * 0.035)
        else:
            offset = max(180.0, min(content_w, content_h) * 0.015)
        add_gost_dimension(msp, p1, p2, value_mm=dim.get("value_mm"), offset=offset)

    for label in geometry.get("texts") or []:
        pos = label.get("pos") or [0, 0]
        kind = str(label.get("kind") or "dimension").lower()
        layer = LAYER_NAMES["dims"] if kind == "dimension" else LAYER_NAMES["detail"]
        t = msp.add_text(
            str(label.get("text", "")),
            height=float(label.get("height") or max(120.0, content_h * 0.012)),
            dxfattribs={"layer": layer, "style": "GOST_ST"},
        )
        rot = float(label.get("rotation") or 0.0)
        if hasattr(t.dxf, "rotation"):
            t.dxf.rotation = rot
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
