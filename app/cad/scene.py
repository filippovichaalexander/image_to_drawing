"""Extract lightweight scene JSON from DXF for the WebGL viewer."""

from __future__ import annotations

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


def _line_entity(e: Any) -> dict[str, Any] | None:
    layer = e.dxf.layer
    if e.dxftype() == "LINE":
        s, t = e.dxf.start, e.dxf.end
        return {
            "layer": layer,
            "type": "line",
            "points": [[float(s.x), float(s.y)], [float(t.x), float(t.y)]],
        }
    if e.dxftype() == "LWPOLYLINE":
        pts = [[float(p[0]), float(p[1])] for p in e.get_points("xy")]
        if len(pts) < 2:
            return None
        return {"layer": layer, "type": "polyline", "points": pts, "closed": bool(e.closed)}
    if e.dxftype() == "POLYLINE":
        pts = [[float(v.dxf.location.x), float(v.dxf.location.y)] for v in e.vertices]
        if len(pts) < 2:
            return None
        return {"layer": layer, "type": "polyline", "points": pts, "closed": bool(e.is_closed)}
    if e.dxftype() == "TEXT":
        p = e.dxf.insert
        return {
            "layer": layer,
            "type": "text",
            "points": [[float(p.x), float(p.y)]],
            "text": _dxf_text(e.dxf.text),
            "height": float(getattr(e.dxf, "height", 2.5) or 2.5),
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
            # Expand bounds a bit for text height
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
