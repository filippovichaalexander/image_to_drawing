"""Frame and title block (Form 3) per ГОСТ Р 21.101-2020."""

from __future__ import annotations

from typing import Any

import ezdxf
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment

from app.cad.gost import LAYER_NAMES, LINEWEIGHT


def ensure_layers(doc: ezdxf.document.Drawing) -> None:
    specs = {
        LAYER_NAMES["frame"]: {"color": colors.WHITE, "lineweight": LINEWEIGHT["medium"]},
        LAYER_NAMES["stamp_text"]: {"color": colors.WHITE, "lineweight": LINEWEIGHT["thin"]},
        LAYER_NAMES["walls"]: {"color": colors.RED, "lineweight": LINEWEIGHT["thick"]},
        LAYER_NAMES["partitions"]: {"color": colors.YELLOW, "lineweight": LINEWEIGHT["medium"]},
        LAYER_NAMES["axes"]: {"color": colors.CYAN, "lineweight": LINEWEIGHT["thin"], "linetype": "CENTER"},
        LAYER_NAMES["dims"]: {"color": colors.GREEN, "lineweight": LINEWEIGHT["thin"]},
        LAYER_NAMES["openings"]: {"color": colors.BLUE, "lineweight": LINEWEIGHT["medium"]},
        LAYER_NAMES["detail"]: {"color": colors.WHITE, "lineweight": LINEWEIGHT["thin"]},
        LAYER_NAMES["stairs"]: {"color": colors.WHITE, "lineweight": LINEWEIGHT["thin"]},
        LAYER_NAMES["hatch"]: {"color": 6, "lineweight": LINEWEIGHT["thin"]},
    }
    if "CENTER" not in doc.linetypes:
        doc.linetypes.add("CENTER", pattern="A,12.5,-2.5,0.5,-2.5", description="Axis center")

    for name, attrs in specs.items():
        if name not in doc.layers:
            dxfattribs = {
                "color": attrs["color"],
                "lineweight": attrs["lineweight"],
            }
            if "linetype" in attrs:
                dxfattribs["linetype"] = attrs["linetype"]
            doc.layers.add(name, **dxfattribs)


def ensure_gost_style(doc: ezdxf.document.Drawing) -> None:
    """Prefer ISOCPEUR / GOST-like; fall back to txt if font missing in runtime."""
    if "GOST_ST" not in doc.styles:
        try:
            doc.styles.add("GOST_ST", font="ISOCPEUR")
        except Exception:  # noqa: BLE001
            doc.styles.add("GOST_ST", font="txt")


_LANDSCAPE = {
    "A4": (297.0, 210.0),
    "A3": (420.0, 297.0),
    "A2": (594.0, 420.0),
    "A1": (841.0, 594.0),
}

_PORTRAIT = {
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
}


def pick_sheet_format(bounds_w: float, bounds_h: float) -> tuple[str, str, float, float]:
    """Choose sheet so content (+ margins + stamp) fits; prefer landscape."""
    need_w = bounds_w + 70
    need_h = bounds_h + 90
    for name in ("A4", "A3", "A2", "A1"):
        lw, lh = _LANDSCAPE[name]
        if need_w <= lw - 25 and need_h <= lh - 10:
            return name, "landscape", lw, lh
    for name in ("A3", "A2", "A1"):
        pw, ph = _PORTRAIT[name]
        if need_w <= pw - 25 and need_h <= ph - 10:
            return name, "portrait", pw, ph
    return "A1", "landscape", *_LANDSCAPE["A1"]


def add_gost_stamp_and_frame(
    doc: ezdxf.document.Drawing,
    msp: Any,
    format_name: str = "A3",
    orientation: str = "landscape",
    metadata: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """
    Outer/inner frame + Form 3 stamp (185×55) in model space (mm, 1:1).
    Returns sheet (width, height).
    """
    metadata = metadata or {}
    sizes = _PORTRAIT if orientation == "portrait" else _LANDSCAPE
    w, h = sizes.get(format_name, sizes["A3"])

    ensure_layers(doc)
    ensure_gost_style(doc)
    frame = LAYER_NAMES["frame"]
    text_layer = LAYER_NAMES["stamp_text"]

    # Outer sheet edge
    msp.add_lwpolyline(
        [(0, 0), (w, 0), (w, h), (0, h), (0, 0)],
        dxfattribs={"layer": frame, "lineweight": LINEWEIGHT["thin"]},
    )

    # Inner frame: 20 mm left (binding), 5 mm other sides — ГОСТ Р 21.101-2020
    x_min, y_min = 20.0, 5.0
    x_max, y_max = w - 5.0, h - 5.0
    msp.add_lwpolyline(
        [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)],
        dxfattribs={"layer": frame, "lineweight": LINEWEIGHT["medium"]},
    )

    # Form 3 stamp 185×55 bottom-right inside inner frame
    sx, sy = x_max - 185.0, y_min
    msp.add_lwpolyline(
        [(sx, sy), (x_max, sy), (x_max, sy + 55), (sx, sy + 55), (sx, sy)],
        dxfattribs={"layer": frame},
    )

    # Horizontal dividers (Form 3 simplified denser grid)
    for dy in (5, 10):
        msp.add_line((sx, sy + dy), (sx + 50, sy + dy), dxfattribs={"layer": frame})
    for dy in (15, 25, 40, 45, 50):
        msp.add_line((sx, sy + dy), (x_max, sy + dy), dxfattribs={"layer": frame})

    # Vertical dividers
    msp.add_line((sx + 50, sy), (sx + 50, sy + 55), dxfattribs={"layer": frame})
    msp.add_line((sx + 70, sy + 15), (sx + 70, sy + 55), dxfattribs={"layer": frame})
    msp.add_line((x_max - 35, sy), (x_max - 35, sy + 55), dxfattribs={"layer": frame})
    msp.add_line((x_max - 20, sy), (x_max - 20, sy + 15), dxfattribs={"layer": frame})
    msp.add_line((x_max - 10, sy), (x_max - 10, sy + 15), dxfattribs={"layer": frame})

    def text(
        value: str,
        pos: tuple[float, float],
        height: float = 2.5,
        center: bool = False,
    ) -> None:
        if not value:
            return
        entity = msp.add_text(
            value,
            height=height,
            dxfattribs={"layer": text_layer, "style": "GOST_ST"},
        )
        if center:
            entity.set_placement(pos, align=TextEntityAlignment.MIDDLE_CENTER)
        else:
            entity.set_placement(pos, align=TextEntityAlignment.LEFT)

    # Left signature column labels
    text("Изм.", (sx + 2, sy + 51), height=1.8)
    text("Кол.уч.", (sx + 12, sy + 51), height=1.8)
    text("Лист", (sx + 28, sy + 51), height=1.8)
    text("№ док.", (sx + 2, sy + 46), height=1.8)
    text("Подп.", (sx + 28, sy + 46), height=1.8)
    text("Дата", (sx + 40, sy + 46), height=1.8)

    text("Разраб.", (sx + 2, sy + 41), height=2.0)
    text(str(metadata.get("drawn_by", ""))[:18], (sx + 18, sy + 41), height=2.2)
    text("Пров.", (sx + 2, sy + 31), height=2.0)
    text(str(metadata.get("checked_by", ""))[:18], (sx + 18, sy + 31), height=2.2)
    text("Н.контр.", (sx + 2, sy + 21), height=2.0)
    text("ГИП", (sx + 2, sy + 11), height=2.0)

    # Project code / object / sheet
    text(str(metadata.get("code", "")), (sx + 117.5, sy + 47.5), height=5.0, center=True)
    text(str(metadata.get("object_name", ""))[:52], (sx + 72, sy + 28), height=2.5)
    text(str(metadata.get("sheet_name", ""))[:44], (sx + 72, sy + 18), height=2.5)
    text(str(metadata.get("company", ""))[:36], (sx + 72, sy + 8), height=2.2)

    # Stage / sheet / sheets count cells
    text("Стадия", (x_max - 27.5, sy + 12), height=1.6, center=True)
    text(str(metadata.get("stage", "Р")), (x_max - 27.5, sy + 5), height=3.5, center=True)
    text("Лист", (x_max - 15, sy + 12), height=1.6, center=True)
    text(str(metadata.get("sheet_num", "1")), (x_max - 15, sy + 5), height=3.0, center=True)
    text("Листов", (x_max - 5, sy + 12), height=1.6, center=True)
    text(str(metadata.get("sheets_total", "1")), (x_max - 5, sy + 5), height=3.0, center=True)

    return w, h
