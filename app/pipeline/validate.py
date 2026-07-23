"""Stage 4: multi-sheet dimension correlation / GOST module rounding (lightweight)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def _snap_module(value_mm: float, module: int = 50) -> int:
    return int(round(value_mm / module) * module)


def _norm_axis_name(name: str) -> str:
    raw = str(name or "").strip().upper()
    for token in ("ОСЬ", "AXIS", "ОС", "AX"):
        raw = raw.replace(token, "")
    # Latin/Cyrillic lookalikes often OCR'd on scans
    table = str.maketrans({"A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р", "C": "С", "T": "Т", "X": "Х"})
    raw = raw.translate(table).strip(" .-_:")
    return raw


def _sheet_role(sheet: dict[str, Any], index: int, metadata_sheet_name: str = "") -> str:
    """Heuristic plan vs elevation/section from filename / UI sheet name."""
    blob = " ".join(
        [
            str(sheet.get("source", "")),
            metadata_sheet_name,
            str(sheet.get("role", "")),
        ]
    ).lower()
    if any(k in blob for k in ("фасад", "facade", "elevation", "разрез", "section", "разр")):
        return "elevation"
    if any(k in blob for k in ("план", "plan", "этаж", "floor")):
        return "plan"
    return "plan" if index == 0 else "elevation"


def correlate_dimensions(
    geometry: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Voting matrix across dimension candidates, axis-name unify, and
    lightweight plan↔elevation height transfer by shared axis labels.
    """
    metadata = metadata or {}
    geometry = dict(geometry)
    dims_in = list(geometry.get("dimensions") or [])

    # Group by rounded length bucket for voting (ТЗ: матрица голосования)
    buckets: dict[int, list[float]] = defaultdict(list)
    for d in dims_in:
        raw = float(d.get("value_mm", 0))
        buckets[_snap_module(raw, 100)].append(raw)

    voted: dict[int, float] = {}
    for key, values in buckets.items():
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

    axes_in = list(geometry.get("axes") or [])
    axes_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    axes_out: list[dict[str, Any]] = []
    for axis in axes_in:
        a = dict(axis)
        a["name"] = _norm_axis_name(str(a.get("name", ""))) or str(a.get("name", ""))
        axes_out.append(a)
        if a["name"]:
            axes_by_name[a["name"]].append(a)

    # Collapse duplicate axis names: average coordinates across sheets
    unified_axes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for a in axes_out:
        name = a["name"]
        if not name or name in seen:
            continue
        seen.add(name)
        group = axes_by_name[name]
        merged = dict(group[0])
        if "x" in merged:
            xs = [float(g["x"]) for g in group if "x" in g]
            if xs:
                merged["x"] = sum(xs) / len(xs)
            y0s = [float(g["y0"]) for g in group if "y0" in g]
            y1s = [float(g["y1"]) for g in group if "y1" in g]
            if y0s:
                merged["y0"] = min(y0s)
            if y1s:
                merged["y1"] = max(y1s)
        elif "y" in merged:
            ys = [float(g["y"]) for g in group if "y" in g]
            if ys:
                merged["y"] = sum(ys) / len(ys)
        merged["sheets"] = sorted({int(g.get("sheet", 0)) for g in group})
        unified_axes.append(merged)

    # Height transfer: elevation overall_y → openings Z / plan opening height notes
    sheets_meta = list(geometry.get("sheets") or [])
    sheet_name = str(metadata.get("sheet_name", ""))
    roles = {
        int(s.get("index", i)): _sheet_role(s, int(s.get("index", i)), sheet_name)
        for i, s in enumerate(sheets_meta)
    }
    if not roles and dims_out:
        roles = {0: "plan"}

    elev_heights: list[float] = []
    for d in dims_out:
        sheet = int(d.get("sheet", 0))
        if roles.get(sheet, "plan") != "elevation":
            continue
        if d.get("role") == "overall_y" or abs(float(d["p2"][1]) - float(d["p1"][1])) > abs(
            float(d["p2"][0]) - float(d["p1"][0])
        ):
            elev_heights.append(float(d["value_mm"]))

    transferred = 0
    if elev_heights:
        # Prefer median elevation height, snap to 50 mm module
        z_mm = float(_snap_module(sorted(elev_heights)[len(elev_heights) // 2], 50))
        openings = []
        for op in geometry.get("openings") or []:
            item = dict(op)
            if "height_mm" not in item or not item.get("height_mm"):
                # Rough: openings get ~1/3 of facade height if window, 2.1m door default
                if str(item.get("type", "")).lower() == "door":
                    item["height_mm"] = 2100
                else:
                    item["height_mm"] = int(_snap_module(z_mm / 3.0, 50))
                item["z_from"] = "elevation"
                transferred += 1
            item["facade_height_mm"] = int(z_mm)
            openings.append(item)
        geometry["openings"] = openings
        geometry["elevation_height_mm"] = int(z_mm)

    geometry["dimensions"] = dims_out
    geometry["axes"] = unified_axes or axes_out
    geometry["correlated"] = True
    geometry["vote_buckets"] = {str(k): v for k, v in voted.items()}
    geometry["sheet_roles"] = {str(k): v for k, v in roles.items()}
    geometry["heights_transferred"] = transferred
    return geometry
