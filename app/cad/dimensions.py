"""GOST 2.307-2011 dimension lines with 45° ticks."""

from __future__ import annotations

import math
from typing import Any

from app.cad.gost import DIM_OFFSET_MM, DIM_TICK_MM, LAYER_NAMES


def _offset_point(x: float, y: float, nx: float, ny: float, dist: float) -> tuple[float, float]:
    return x + nx * dist, y + ny * dist


def add_gost_dimension(
    msp: Any,
    p1: tuple[float, float],
    p2: tuple[float, float],
    value_mm: float | int | None = None,
    offset: float = DIM_OFFSET_MM,
) -> None:
    """
    Draw extension lines, dimension line and 45° tick marks.
    Coordinates are in millimetres (model space).
    """
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    # outward normal
    nx, ny = -uy, ux

    # Place dimension away from object
    a1 = _offset_point(x1, y1, nx, ny, offset)
    a2 = _offset_point(x2, y2, nx, ny, offset)
    # Extension lines from object toward dim line (short gap near object)
    gap = max(1.0, offset * 0.05)
    e1 = _offset_point(x1, y1, nx, ny, gap)
    e2 = _offset_point(x2, y2, nx, ny, gap)
    layer = LAYER_NAMES["dims"]

    msp.add_line(e1, a1, dxfattribs={"layer": layer})
    msp.add_line(e2, a2, dxfattribs={"layer": layer})
    msp.add_line(a1, a2, dxfattribs={"layer": layer})

    # 45° ticks — scale with offset for large model-space drawings
    tick = max(DIM_TICK_MM, offset * 0.15)
    ang = math.radians(45)
    tx = math.cos(ang) * ux - math.sin(ang) * uy
    ty = math.sin(ang) * ux + math.cos(ang) * uy
    for ax, ay in (a1, a2):
        msp.add_line(
            (ax - tx * tick / 2, ay - ty * tick / 2),
            (ax + tx * tick / 2, ay + ty * tick / 2),
            dxfattribs={"layer": layer},
        )

    label = str(int(round(value_mm if value_mm is not None else length)))
    mx, my = (a1[0] + a2[0]) / 2, (a1[1] + a2[1]) / 2
    text_h = max(2.5, offset * 0.35)
    txp, typ = _offset_point(mx, my, nx, ny, text_h)
    text = msp.add_text(label, height=text_h, dxfattribs={"layer": layer, "style": "GOST_ST"})
    text.set_placement((txp, typ))
