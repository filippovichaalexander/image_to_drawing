"""Stage 1: prepare rasters for YOLO / OCR (keep text for OCR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_INFERENCE_SIDE = 1600
MAX_OCR_SIDE = 2000


def _imwrite(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Failed to encode {path}")
    path.write_bytes(buf.tobytes())


def _imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать {path}")
    return img


def prepare_for_inference(src: Path, dst_dir: Path) -> dict[str, Any]:
    """
    Return paths/scales for YOLO and OCR from an original (non-binarized) sheet.

    - OCR copy: max side ≤ MAX_OCR_SIDE (text stays readable)
    - YOLO copy: max side ≤ MAX_INFERENCE_SIDE
    Scale factors map processed coords → original pixels (multiply).
    """
    img = _imread(src)
    h, w = img.shape[:2]
    dst_dir.mkdir(parents=True, exist_ok=True)

    if max(h, w) > MAX_OCR_SIDE:
        scale = MAX_OCR_SIDE / max(h, w)
        ocr_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ocr_path = dst_dir / f"{src.stem}_ocr.png"
        _imwrite(ocr_path, ocr_img)
        ocr_scale = 1.0 / scale
    else:
        ocr_path = src
        ocr_scale = 1.0

    if max(h, w) > MAX_INFERENCE_SIDE:
        scale = MAX_INFERENCE_SIDE / max(h, w)
        yolo_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        yolo_path = dst_dir / f"{src.stem}_yolo.png"
        _imwrite(yolo_path, yolo_img)
        yolo_scale = 1.0 / scale
    else:
        yolo_path = src
        yolo_scale = 1.0

    return {
        "yolo_path": yolo_path,
        "ocr_path": ocr_path,
        "yolo_scale": yolo_scale,
        "ocr_scale": ocr_scale,
        "orig_h": h,
        "orig_w": w,
        "source": src,
    }
