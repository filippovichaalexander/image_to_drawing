"""Stage 4: multi-sheet dimension correlation / GOST module rounding."""

from __future__ import annotations

from typing import Any


def _snap_module(value_mm: float, module: int = 50) -> int:
    """Round to construction module (10/50/100 mm) — GOST-friendly."""
    return int(round(value_mm / module) * module)


def correlate_dimensions(geometry: dict[str, Any]) -> dict[str, Any]:
    geometry = dict(geometry)
    dims = []
    for d in geometry.get("dimensions", []):
        item = dict(d)
        raw = float(item.get("value_mm", 0))
        item["value_raw"] = raw
        item["value_mm"] = _snap_module(raw, 50)
        dims.append(item)
    geometry["dimensions"] = dims
    geometry["correlated"] = True
    return geometry
