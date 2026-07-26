"""Stage 2: YOLOv8-seg floorplan instance segmentation."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fallback when checkpoint has custom ids but no readable names.
FLOORPLAN_CLASS_MAP = {
    0: "wall",
    1: "door",
    2: "window",
    3: "stair",
    4: "axis",
}

_WALL_ALIASES = {"wall", "walls", "partition", "partitions", "стена", "стены"}
_DOOR_ALIASES = {"door", "doors", "дверь", "двери"}
_WINDOW_ALIASES = {"window", "windows", "окно", "окна"}
_STAIR_ALIASES = {"stair", "stairs", "staircase", "лестница"}
_AXIS_ALIASES = {"axis", "axes", "ось", "оси"}

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_seg_weights() -> Path:
    """Prefer domain model, then bootstrap walls_seg, then ultralytics nano-seg."""
    candidates = [
        _REPO_ROOT / "models" / "floorplan-seg.pt",
        _REPO_ROOT / "models" / "walls_seg.pt",
        Path("models/floorplan-seg.pt"),
        Path("models/walls_seg.pt"),
        Path("yolov8n-seg.pt"),
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path.resolve()
    return (_REPO_ROOT / "models" / "walls_seg.pt").resolve()


@lru_cache(maxsize=1)
def get_floorplan_model():
    """Model 1 — floor-plan parser (CVC-FP + R2V + CubiCasa walls target)."""
    return load_model()


@lru_cache(maxsize=1)
def load_model():
    from ultralytics import YOLO  # type: ignore

    weights = resolve_seg_weights()
    if not weights.is_file():
        logger.warning("Seg weights missing at %s — downloading yolov8n-seg.pt", weights)
        return YOLO("yolov8n-seg.pt")
    logger.info("Loading YOLO seg weights: %s", weights)
    return YOLO(str(weights))


def _normalize_class(name: str) -> str | None:
    key = (name or "").strip().lower()
    if key in _WALL_ALIASES or "wall" in key:
        return "wall"
    if key in _DOOR_ALIASES or "door" in key:
        return "door"
    if key in _WINDOW_ALIASES or "window" in key:
        return "window"
    if key in _STAIR_ALIASES or "stair" in key:
        return "stair"
    if key in _AXIS_ALIASES or "axis" in key:
        return "axis"
    return None


def _class_name_for(model, cls_id: int, prefer_named: bool) -> str:
    names = getattr(model, "names", None) or {}
    raw = names.get(cls_id, names.get(str(cls_id), "")) if isinstance(names, dict) else ""
    mapped = _normalize_class(str(raw))
    if mapped:
        return mapped
    if prefer_named:
        return FLOORPLAN_CLASS_MAP.get(cls_id, "wall")
    # COCO / unknown taxonomy: treat every mask as wall region
    return "wall"


def segment_floorplan(
    image_path: Path,
    orig_h: int,
    orig_w: int,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict[str, Any]:
    """
    Run YOLO-seg; return polygons/boxes in **original image pixel** coordinates.
    """
    model = load_model()
    weights = resolve_seg_weights()
    prefer_named = weights.is_file() and (
        "floorplan" in weights.name.lower() or "archivision" in weights.name.lower()
    )
    # Also prefer names if model already exposes wall/door labels
    names = getattr(model, "names", None) or {}
    name_blob = " ".join(str(v).lower() for v in (names.values() if isinstance(names, dict) else []))
    if any(k in name_blob for k in ("wall", "door", "window", "стена")):
        prefer_named = True

    results = model.predict(
        str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=0.5,
        verbose=False,
    )

    output: dict[str, Any] = {
        "walls": [],
        "doors": [],
        "windows": [],
        "stairs": [],
        "axes": [],
        "weights": str(weights),
        "named_classes": prefer_named,
        "model_names": names if isinstance(names, dict) else {},
    }

    for result in results:
        if result.masks is None:
            continue
        yolo_h, yolo_w = result.orig_shape
        fx = orig_w / max(float(yolo_w), 1.0)
        fy = orig_h / max(float(yolo_h), 1.0)

        boxes = result.boxes
        for i, mask in enumerate(result.masks.xy):
            cls_id = 0
            if boxes is not None and i < len(boxes.cls):
                cls_id = int(boxes.cls[i].item())
            cls_name = _class_name_for(model, cls_id, prefer_named)

            pts = [[float(x) * fx, float(y) * fy] for x, y in mask]
            if len(pts) < 3:
                continue

            if cls_name == "wall":
                output["walls"].append(pts)
            elif cls_name == "door" and boxes is not None and i < len(boxes.xyxy):
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                output["doors"].append(
                    {
                        "p1": [x1 * fx, y1 * fy],
                        "p2": [x2 * fx, y2 * fy],
                        "center": [(x1 + x2) / 2 * fx, (y1 + y2) / 2 * fy],
                        "kind": "door",
                    }
                )
            elif cls_name == "window" and boxes is not None and i < len(boxes.xyxy):
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                output["windows"].append(
                    {
                        "p1": [x1 * fx, y1 * fy],
                        "p2": [x2 * fx, y2 * fy],
                        "kind": "window",
                    }
                )
            elif cls_name == "stair":
                output["stairs"].append(pts)
            elif cls_name == "axis":
                output["axes"].append(pts)
            else:
                output["walls"].append(pts)

    return output
