"""Stage 1: scan de-noising / binarization (OpenCV lightweight path)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def denoise_image(src: Path, dst: Path) -> Path:
    """
    Produce a high-contrast binary image suitable for vectorization.

    MVP uses classical OpenCV (fast, CPU-only). ControlNet/SAM remain
    behind AI_MODE=full in a later commit — not public cloud APIs.
    """
    data = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {src.name}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mild denoise without erasing thin ink lines
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    # Adaptive threshold handles uneven scan lighting / gray paper
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        10,
    )
    # Morphological close to reconnect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", binary)
    if not ok:
        raise RuntimeError("Ошибка кодирования PNG")
    buf.tofile(str(dst))
    return dst
