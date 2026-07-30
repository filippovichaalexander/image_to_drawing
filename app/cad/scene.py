"""Extract lightweight scene JSON from DXF for the WebGL viewer."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import ezdxf

_UESC = re.compile(r"\\[Uu]\+([0-9A-Fa-f]{4,6})")


def _dxf_text(raw: str) -> str:
    """Decode R2000 \\U+XXXX escapes so Cyrillic axis labels are readable."""
    if not raw:
        return ""
    return _UESC.sub(lambda m: chr(int(m.group(1), 16)), str(raw))


def _circle_points(cx: float, cy: float, r: float, n: int = 48) -> list[list[float]]:
    return [
        [cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)]
        for i in range(n + 1)
    ]


def _arc_points(
    cx: float,
    cy: float,
    r: float,
    start_deg: float,
    end_deg: float,
    n: int = 32,
) -> list[list[float]]:
    # ezdxf / AutoCAD: counterclockwise from start to end
    a0 = math.radians(start_deg)
    a1 = math.radians(end_deg)
    if a1 <= a0:
        a1 += 2 * math.pi
    span = a1 - a0
    steps = max(8, int(n * span / (2 * math.pi)) + 1)
    return [
        [cx + r * math.cos(a0 + span * i / steps), cy + r * math.sin(a0 + span * i / steps)]
        for i in range(steps + 1)
    ]


def _line_entity(e: Any) -> dict[str, Any] | None:
    layer = e.dxf.layer
    dxftype = e.dxftype()
    if dxftype == "LINE":
        s, t = e.dxf.start, e.dxf.end
        return {
            "layer": layer,
            "type": "line",
            "points": [[float(s.x), float(s.y)], [float(t.x), float(t.y)]],
        }
    if dxftype == "LWPOLYLINE":
        pts = [[float(p[0]), float(p[1])] for p in e.get_points("xy")]
        if len(pts) < 2:
            return None
        return {
            "layer": layer,
            "type": "polyline",
            "points": pts,
            "closed": bool(e.closed),
            # Never soft-fill wall outlines — GOST plans are hollow double-lines.
            # Only true HATCH entities (handled below) are filled.
            "filled": False,
        }
    if dxftype == "POLYLINE":
        pts = [[float(v.dxf.location.x), float(v.dxf.location.y)] for v in e.vertices]
        if len(pts) < 2:
            return None
        return {
            "layer": layer,
            "type": "polyline",
            "points": pts,
            "closed": bool(e.is_closed),
            "filled": False,
        }
    if dxftype == "CIRCLE":
        c = e.dxf.center
        r = float(e.dxf.radius)
        pts = _circle_points(float(c.x), float(c.y), r)
        return {
            "layer": layer,
            "type": "circle",
            "points": pts,
            "center": [float(c.x), float(c.y)],
            "radius": r,
            "closed": True,
        }
    if dxftype == "ARC":
        c = e.dxf.center
        r = float(e.dxf.radius)
        pts = _arc_points(
            float(c.x),
            float(c.y),
            r,
            float(e.dxf.start_angle),
            float(e.dxf.end_angle),
        )
        return {
            "layer": layer,
            "type": "arc",
            "points": pts,
            "center": [float(c.x), float(c.y)],
            "radius": r,
            "start_angle": float(e.dxf.start_angle),
            "end_angle": float(e.dxf.end_angle),
        }
    if dxftype == "HATCH":
        # Approximate hatch boundaries as closed polylines (filled)
        pts: list[list[float]] = []
        try:
            for path in e.paths:
                if hasattr(path, "vertices"):
                    pts = [[float(v[0]), float(v[1])] for v in path.vertices]
                    break
                if hasattr(path, "edges"):
                    # polyline path via edges — skip complex
                    continue
        except Exception:  # noqa: BLE001
            return None
        if len(pts) < 3:
            return None
        return {
            "layer": layer,
            "type": "polyline",
            "points": pts,
            "closed": True,
            "filled": True,
        }
    if dxftype == "TEXT":
        p = e.dxf.insert
        return {
            "layer": layer,
            "type": "text",
            "points": [[float(p.x), float(p.y)]],
            "text": _dxf_text(e.dxf.text),
            "height": float(getattr(e.dxf, "height", 2.5) or 2.5),
            "rotation": float(getattr(e.dxf, "rotation", 0.0) or 0.0),
        }
    return None


def dxf_to_scene(dxf_path: Path) -> dict[str, Any]:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    entities: list[dict[str, Any]] = []
    layers: set[str] = set()
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for e in msp:
        item = _line_entity(e)
        if not item:
            continue
        layers.add(item["layer"])
        entities.append(item)
        for x, y in item["points"]:
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
        if item["type"] == "text":
            h = float(item.get("height") or 0)
            x, y = item["points"][0]
            min_x, max_x = min(min_x, x), max(max_x, x + h * max(len(item.get("text") or ""), 1))
            min_y, max_y = min(min_y, y), max(max_y, y + h)

    if not entities:
        min_x = min_y = 0.0
        max_x = max_y = 100.0

    return {
        "layers": sorted(layers),
        "entities": entities,
        "bounds": {
            "min": [min_x, min_y],
            "max": [max_x, max_y],
        },
    }
