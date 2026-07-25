"""GOST / СПДС constants for CAD generation."""

# ГОСТ 2.303-68 — lineweights in mm → ezdxf uses 100ths of mm
LINEWEIGHT = {
    "thick": 70,   # ~0.7 mm capital walls
    "medium": 40,  # 0.4 mm frame
    "thin": 25,    # 0.25 mm dimensions / hatch
    "axis": 25,
}

# ГОСТ 2.307-2011
DIM_OFFSET_MM = 10.0
DIM_TICK_MM = 3.0  # 2–4 mm ticks at 45°

# ГОСТ Р 21.101-2020 — portrait reference sizes (mm); landscape via stamp._LANDSCAPE
SHEET_SIZES = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "A1": (594, 841),
}

# ASCII layer names — R2000 DXF cannot store Cyrillic without \\U+ escapes
# (breaks WebGL layer toggles). Semantic names follow ТЗ order.
LAYER_NAMES = {
    "frame": "00_FRAME",
    "stamp_text": "00_STAMP_TEXT",
    "walls": "01_WALLS",
    "partitions": "02_PARTITIONS",
    "axes": "03_AXES",
    "dims": "04_DIMS_GOST",
    "openings": "05_OPENINGS",
}

LAYER_LABELS_RU = {
    "00_FRAME": "00_Оформление",
    "00_STAMP_TEXT": "00_Текст_Штампа",
    "01_WALLS": "01_Стены",
    "02_PARTITIONS": "02_Перегородки",
    "03_AXES": "03_Оси",
    "04_DIMS_GOST": "04_Размеры",
    "05_OPENINGS": "05_Окна_Двери",
}
