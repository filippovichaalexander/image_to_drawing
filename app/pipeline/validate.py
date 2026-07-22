"""Stage 4: multi-sheet dimension correlation / GOST module rounding."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _snap_module(value_mm: float, module: int = 50) -> int:
    return int(round(value_mm / module) * module)


def correlate_dimensions(geometry: dict[str, Any]) -> dict[str, Any]:
    """
    Voting matrix across dimension candidates sharing approximate endpoints /
    axis labels, then snap to construction module (10/50/100 mm).
    """
    geometry = dict(geometry)
    dims_in = list(geometry.get("dimensions") or [])

    # Group by rounded length bucket for voting
    buckets: dict[int, list[float]] = defaultdict(list)
    for d in dims_in:
        raw = float(d.get("value_mm", 0))
        buckets[_snap_module(raw, 100)].append(raw)

    voted: dict[int, float] = {}
    for key, values in buckets.items():
        # Prefer median then snap to 50 mm module (ТЗ)
        median = sorted(values)[len(values) // 2]
        voted[key] = float(_snap_module(median, 50))

    dims_out = []
    for d in dims_in:
        item = dict(d)
        raw = float(item.get("value_mm", 0))
        key = _snap_module(raw, 100)
        item["value_raw"] = raw
        item["value_mm"] = int(voted.get(key, _snap_module(raw, 50)))
        dims_out.append(item)

    # Axis name normalization (Cyrillic / Latin lookalikes)
    axes = []
    for axis in geometry.get("axes") or []:
        a = dict(axis)
        name = str(a.get("name", "")).strip().upper().replace("ОСЬ", "").strip()
        a["name"] = name or a.get("name", "")
        axes.append(a)

    geometry["dimensions"] = dims_out
    geometry["axes"] = axes
    geometry["correlated"] = True
    geometry["vote_buckets"] = {str(k): v for k, v in voted.items()}
    return geometry
