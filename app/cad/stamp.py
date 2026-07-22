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
        LAYER_NAMES["axes"]: {"color": colors.CYAN, "lineweight": LINEWEIGHT["axis"], "linetype": "CENTER"},
        LAYER_NAMES["dims"]: {"color": colors.GREEN, "lineweight": LINEWEIGHT["thin"]},
        LAYER_NAMES["openings"]: {"color": colors.BLUE, "lineweight": LINEWEIGHT["medium"]},
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


def pick_sheet_format(bounds_w: float, bounds_h: float) -> tuple[str, str, float, float]:
    """Choose landscape sheet so content (+ margins + stamp) fits."""
    need_w = bounds_w + 70
    need_h = bounds_h + 90
    for name in ("A4", "A3", "A2", "A1"):
        lw, lh = _LANDSCAPE[name]
        if need_w <= lw - 25 and need_h <= lh - 10:
            return name, "landscape", lw, lh
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
    _ = orientation  # portrait reserved for later
    w, h = _LANDSCAPE.get(format_name, _LANDSCAPE["A3"])

    ensure_layers(doc)
    ensure_gost_style(doc)
    frame = LAYER_NAMES["frame"]
    text_layer = LAYER_NAMES["stamp_text"]

    # Outer sheet edge (optional thin)
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
    msp.add_line((sx, sy + 15), (x_max, sy + 15), dxfattribs={"layer": frame})
    msp.add_line((sx + 50, sy), (sx + 50, sy + 55), dxfattribs={"layer": frame})
    msp.add_line((x_max - 35, sy), (x_max - 35, sy + 15), dxfattribs={"layer": frame})
    msp.add_line((x_max - 35, sy + 15), (x_max - 35, sy + 55), dxfattribs={"layer": frame})
    msp.add_line((sx, sy + 40), (sx + 50, sy + 40), dxfattribs={"layer": frame})
    msp.add_line((sx, sy + 30), (sx + 50, sy + 30), dxfattribs={"layer": frame})

    def text(value: str, pos: tuple[float, float], height: float = 2.5, center: bool = False) -> None:
        entity = msp.add_text(
            value,
            height=height,
            dxfattribs={"layer": text_layer, "style": "GOST_ST"},
        )
        if center:
            entity.set_placement(pos, align=TextEntityAlignment.MIDDLE_CENTER)
        else:
            entity.set_placement(pos, align=TextEntityAlignment.LEFT)

    text(str(metadata.get("code", "")), (sx + 117.5, sy + 35), height=5.0, center=True)
    text(str(metadata.get("object_name", ""))[:48], (sx + 55, sy + 8), height=2.5)
    text(str(metadata.get("sheet_name", ""))[:40], (sx + 55, sy + 22), height=2.5)
    text(str(metadata.get("stage", "Р")), (x_max - 17.5, sy + 7.5), height=3.5, center=True)
    text(str(metadata.get("sheet_num", "1")), (x_max - 17.5, sy + 35), height=3.5, center=True)
    text(f"Разраб. {metadata.get('drawn_by', '')}", (sx + 2, sy + 45), height=2.5)
    text(f"Пров. {metadata.get('checked_by', '')}", (sx + 2, sy + 33), height=2.5)
    text(str(metadata.get("company", ""))[:28], (sx + 2, sy + 2), height=2.0)

    return w, h
