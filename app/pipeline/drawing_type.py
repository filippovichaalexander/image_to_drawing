"""User-declared drawing type → pipeline policy (YOLO openings, profile hints)."""

from __future__ import annotations

import re
from typing import Any

# Stable ids used in forms / metadata
DRAWING_TYPES: dict[str, dict[str, Any]] = {
    "floor_plan": {
        "label_ru": "План здания / этажа (building plan)",
        "use_yolo_openings": True,
        "use_yolo_stairs": True,
        "prefer_profile": "default",
        "note": "Типичный план: стены + окна/двери",
    },
    "cottage_plan": {
        "label_ru": "План коттеджа / этажа (cottage / storey plan)",
        "use_yolo_openings": True,
        "use_yolo_stairs": True,
        "prefer_profile": "default",
        "note": "Жилой план: проёмы и лестницы важны",
    },
    "industrial_plan": {
        "label_ru": "Промплан / цех (industrial plan, rarely windows)",
        "use_yolo_openings": False,
        "use_yolo_stairs": False,
        "prefer_profile": "noisy",
        "note": "Колонны/оборудование часто путаются с окнами — YOLO проёмы выкл.",
    },
    "facade": {
        "label_ru": "Фасад (facade)",
        "use_yolo_openings": False,
        "use_yolo_stairs": False,
        "prefer_profile": "thin_lines",
        "note": "Фасад: floorplan-YOLO для окон плана не подходит",
    },
    "building_cut": {
        "label_ru": "Разрез здания (building cut / section)",
        "use_yolo_openings": False,
        "use_yolo_stairs": True,
        "prefer_profile": "high_detail",
        "note": "Разрез: точная линия важнее семантики проёмов плана",
    },
    "wall_cut": {
        "label_ru": "Разрез стены / узел (wall cut / detail)",
        "use_yolo_openings": False,
        "use_yolo_stairs": False,
        "prefer_profile": "high_detail",
        "note": "Узел/фрагмент стены: только точная трассировка",
    },
    "stair_fragment": {
        "label_ru": "Фрагмент лестницы (staircase fragment)",
        "use_yolo_openings": False,
        "use_yolo_stairs": True,
        "prefer_profile": "high_detail",
        "note": "Фокус на лестнице; оконные классы отключены",
    },
    "other": {
        "label_ru": "Другое / не уверен",
        "use_yolo_openings": False,
        "use_yolo_stairs": False,
        "prefer_profile": "default",
        "note": "Безопасный режим без YOLO-проёмов",
    },
}

_KEYWORD_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"facade|фасад|elevation|front\s*view", re.I), "facade"),
    (re.compile(r"industrial|цех|пром|завод|workshop|shop\s*floor", re.I), "industrial_plan"),
    (re.compile(r"stair|лестниц", re.I), "stair_fragment"),
    (re.compile(r"wall\s*cut|разрез\s*стен|узел|detail\s*node|фрагмент\s*стен", re.I), "wall_cut"),
    (re.compile(r"building\s*cut|разрез\s*здани|section(?!\s*al)|поперечн", re.I), "building_cut"),
    (re.compile(r"cottage|коттедж|storey|этаж|2nd|второй|жилой", re.I), "cottage_plan"),
    (re.compile(r"floor\s*plan|план\s*этаж|building\s*plan|план\s*здани", re.I), "floor_plan"),
]


def resolve_drawing_type(
    drawing_type: str | None = None,
    drawing_note: str | None = None,
) -> dict[str, Any]:
    """Merge select + free-text note into a concrete pipeline policy."""
    raw_type = (drawing_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    note = (drawing_note or "").strip()

    chosen = raw_type if raw_type in DRAWING_TYPES else ""
    matched_from_note = None
    if note:
        for pat, key in _KEYWORD_MAP:
            if pat.search(note):
                matched_from_note = key
                break

    # Explicit select wins unless user left "other"/empty and note is specific
    if chosen and chosen != "other":
        key = chosen
        source = "user_select"
    elif matched_from_note:
        key = matched_from_note
        source = "user_note"
    elif chosen == "other":
        key = "other"
        source = "user_select"
    else:
        key = "floor_plan"
        source = "default"

    base = dict(DRAWING_TYPES[key])
    return {
        "drawing_type": key,
        "drawing_type_label": base["label_ru"],
        "drawing_note": note,
        "drawing_type_source": source,
        "use_yolo_openings": bool(base["use_yolo_openings"]),
        "use_yolo_stairs": bool(base["use_yolo_stairs"]),
        "prefer_profile": str(base["prefer_profile"]),
        "policy_note": str(base["note"]),
    }
