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

LAYER_NAMES = {
    "frame": "00_Оформление",
    "stamp_text": "00_Текст_Штампа",
    "walls": "01_Стены_Капитальные",
    "partitions": "02_Перегородки",
    "axes": "03_Оси",
    "dims": "04_Размеры_ГОСТ",
    "openings": "05_Окна_Двери",
}
