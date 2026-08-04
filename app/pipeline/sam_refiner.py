"""Optional SAM mask refinement for YOLO polygons (no-op without weights)."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_sam_model(model_type: str = "vit_b") -> Any | None:
    """Load SAM predictor (cached). Returns None if weights/deps missing."""
    models_dir = _REPO_ROOT / "models"
    weights_path = models_dir / f"sam_{model_type}.pth"
    if not weights_path.exists():
        for name in (f"sam_{model_type}.pth", "sam_vit_b_01ec64.pth", "sam_vit_b.pth"):
            alt = models_dir / name
            if alt.exists():
                weights_path = alt
                break
    if not weights_path.exists():
        logger.warning(
            "SAM weights not found under %s — run scripts/download_sam.py",
            models_dir,
        )
        return None
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        logger.warning("segment-anything not installed; SAM refine disabled")
        return None
    try:
        logger.info("Loading SAM from %s", weights_path)
        sam = sam_model_registry[model_type](checkpoint=str(weights_path))
        return SamPredictor(sam)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load SAM: %s", exc)
        return None


def polygon_to_mask(
    polygon: list[list[float]],
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    if shape is None:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        shape = (int(max(ys)) + 10, int(max(xs)) + 10)
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


def mask_to_polygon(mask: np.ndarray, epsilon_factor: float = 0.002) -> list[list[float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_factor * peri, True)
    return approx.reshape(-1, 2).astype(float).tolist()


def refine_yolo_masks_with_sam(
    image_path: Path,
    yolo_masks: list[list[list[float]]],
    boxes: list[list[float]] | None = None,
) -> list[list[list[float]]]:
    """
    Refine YOLO polygons with SAM box prompts.
    Returns original masks if SAM is unavailable.
    """
    predictor = load_sam_model()
    if predictor is None or not yolo_masks:
        return yolo_masks

    image = cv2.imread(str(image_path))
    if image is None:
        return yolo_masks

    try:
        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    except Exception as exc:  # noqa: BLE001
        logger.warning("SAM set_image failed: %s", exc)
        return yolo_masks

    refined: list[list[list[float]]] = []
    for i, poly in enumerate(yolo_masks):
        if len(poly) < 3:
            refined.append(poly)
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if boxes and i < len(boxes) and boxes[i] is not None:
            input_box = np.array(boxes[i], dtype=np.float32).reshape(1, 4)
        else:
            input_box = np.array(
                [min(xs), min(ys), max(xs), max(ys)],
                dtype=np.float32,
            ).reshape(1, 4)
        try:
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("SAM predict failed for mask %s: %s", i, exc)
            refined.append(poly)
            continue
        if len(masks) == 0:
            refined.append(poly)
            continue
        best = masks[int(np.argmax(scores))].astype(np.uint8) * 255
        refined_poly = mask_to_polygon(best)
        refined.append(refined_poly if len(refined_poly) >= 3 else poly)
    return refined
