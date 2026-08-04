"""Stage 2: YOLOv8-seg floorplan instance segmentation."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Fallback when checkpoint has custom ids but no readable names.
# Must match training taxonomy: wall, door, window, stair, partition.
FLOORPLAN_CLASS_MAP = {
    0: "wall",
    1: "door",
    2: "window",
    3: "stair",
    4: "partition",
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
    if key in _DOOR_ALIASES or "door" in key:
        return "door"
    if key in _WINDOW_ALIASES or "window" in key:
        return "window"
    if key in _STAIR_ALIASES or "stair" in key:
        return "stair"
    if key in _AXIS_ALIASES or "axis" in key:
        return "axis"
    if key in _WALL_ALIASES or "wall" in key or "partition" in key:
        return "wall"
    return None


def _bbox_to_opening_span(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[list[float], list[float], float]:
    """Turn an axis-aligned detection box into an opening centerline + thickness.

    YOLO returns opposite corners; drawing that diagonal / fat AABB misplaces
    windows on the plan. Prefer the long-axis mid-line (how openings sit in walls).
    """
    xa, xb = sorted((float(x1), float(x2)))
    ya, yb = sorted((float(y1), float(y2)))
    w = max(xb - xa, 1.0)
    h = max(yb - ya, 1.0)
    cx = 0.5 * (xa + xb)
    cy = 0.5 * (ya + yb)
    if w >= h:
        return [xa, cy], [xb, cy], h
    return [cx, ya], [cx, yb], w


def _plausible_opening_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    orig_w: int,
    orig_h: int,
    kind: str,
) -> bool:
    """Reject huge / near-square blobs that are usually mislabeled walls."""
    w = abs(float(x2) - float(x1))
    h = abs(float(y2) - float(y1))
    if w < 2 or h < 2:
        return False
    area = w * h
    img_area = max(float(orig_w) * float(orig_h), 1.0)
    # Windows/doors are local features — not half the sheet
    max_frac = 0.08 if kind == "window" else 0.12
    if area / img_area > max_frac:
        return False
    aspect = max(w, h) / max(min(w, h), 1.0)
    # Opening symbols are usually elongated along the wall
    if kind == "window" and aspect < 1.35 and area / img_area > 0.01:
        return False
    return True


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
    conf: float = 0.40,
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
        iou=0.45,
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
                score = float(boxes.conf[i].item()) if boxes.conf is not None and i < len(boxes.conf) else 1.0
                if score < max(conf, 0.35):
                    continue
                if not _plausible_opening_box(x1, y1, x2, y2, yolo_w, yolo_h, "door"):
                    continue
                p1, p2, thickness = _bbox_to_opening_span(x1, y1, x2, y2)
                output["doors"].append(
                    {
                        "p1": [p1[0] * fx, p1[1] * fy],
                        "p2": [p2[0] * fx, p2[1] * fy],
                        "center": [
                            0.5 * (p1[0] + p2[0]) * fx,
                            0.5 * (p1[1] + p2[1]) * fy,
                        ],
                        "width_px": float(thickness) * 0.5 * (fx + fy),
                        "kind": "door",
                        "conf": score,
                    }
                )
            elif cls_name == "window" and boxes is not None and i < len(boxes.xyxy):
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                score = float(boxes.conf[i].item()) if boxes.conf is not None and i < len(boxes.conf) else 1.0
                # Windows need higher confidence — industrial plans create many FPs
                if score < max(conf, 0.45):
                    continue
                if not _plausible_opening_box(x1, y1, x2, y2, yolo_w, yolo_h, "window"):
                    continue
                p1, p2, thickness = _bbox_to_opening_span(x1, y1, x2, y2)
                output["windows"].append(
                    {
                        "p1": [p1[0] * fx, p1[1] * fy],
                        "p2": [p2[0] * fx, p2[1] * fy],
                        "width_px": float(thickness) * 0.5 * (fx + fy),
                        "kind": "window",
                        "conf": score,
                    }
                )
            elif cls_name == "stair":
                output["stairs"].append(pts)
            elif cls_name == "axis":
                output["axes"].append(pts)
            else:
                output["walls"].append(pts)

    return output
