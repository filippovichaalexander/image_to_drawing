"""Stage 3: dimension / axis OCR (RapidOCR primary; Paddle optional fallback)."""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DIM_RE = re.compile(r"(?<!\d)(\d{3,5})(?!\d)")
_AXIS_RE = re.compile(r"^[А-Яа-яA-Za-z0-9]$")
_MIN_MM = 400
_MAX_MM = 120_000


def _ensure_paddle_home() -> Path:
    candidates = [
        Path(os.environ["PADDLEOCR_HOME"]) if os.environ.get("PADDLEOCR_HOME") else None,
        Path("/app/data/.paddleocr"),
        Path.cwd() / "data" / ".paddleocr",
        Path("/tmp/paddleocr"),
    ]
    for root in candidates:
        if root is None:
            continue
        try:
            root.mkdir(parents=True, exist_ok=True)
            home = root.parent
            os.environ["HOME"] = str(home)
            (home / ".paddleocr").mkdir(parents=True, exist_ok=True)
            return root
        except OSError:
            continue
    return Path("/tmp/paddleocr")


@lru_cache(maxsize=1)
def _load_rapid():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        return RapidOCR()
    except Exception as exc:  # noqa: BLE001
        logger.warning("RapidOCR unavailable (%s)", exc)
        return None


@lru_cache(maxsize=1)
def _load_paddle():
    # Paddle can abort on some images in this Docker stack — keep optional.
    if os.environ.get("DISABLE_PADDLEOCR", "").strip().lower() in {"1", "true", "yes"}:
        return None
    try:
        _ensure_paddle_home()
        from paddleocr import PaddleOCR  # type: ignore

        return PaddleOCR(use_angle_cls=True, lang="ru", show_log=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PaddleOCR unavailable (%s)", exc)
        return None


@lru_cache(maxsize=1)
def load_ocr():
    """Prefer RapidOCR (stable for digit dims); Paddle if Rapid missing."""
    rapid = _load_rapid()
    if rapid is not None:
        return ("rapid", rapid)
    paddle = _load_paddle()
    if paddle is not None:
        return ("paddle", paddle)
    return None


def _iter_paddle(result) -> list[tuple[Any, str, float]]:
    rows: list[tuple[Any, str, float]] = []
    for block in result or []:
        for line in block or []:
            if not line or len(line) < 2:
                continue
            box, (text, conf) = line[0], line[1]
            rows.append((box, str(text), float(conf)))
    return rows


def _iter_rapid(result) -> list[tuple[Any, str, float]]:
    rows: list[tuple[Any, str, float]] = []
    for item in result or []:
        if not item or len(item) < 3:
            continue
        box, text, score = item[0], str(item[1]), float(item[2])
        rows.append((box, text, score))
    return rows


def _parse_items(
    items: list[tuple[Any, str, float]],
    ocr_to_orig: float,
    engine: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dimensions: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    for box, text, conf in items:
        if conf < 0.45:
            continue
        raw_text = (text or "").strip()
        cleaned = raw_text.replace(" ", "").replace(",", "").replace(".", "")
        for match in _DIM_RE.finditer(cleaned):
            value = int(match.group(1))
            if value < _MIN_MM or value > _MAX_MM:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            dimensions.append(
                {
                    "value_mm": value,
                    "cx": sum(xs) / len(xs) * ocr_to_orig,
                    "cy": sum(ys) / len(ys) * ocr_to_orig,
                    "conf": conf,
                    "text": raw_text,
                    "engine": engine,
                }
            )
        compact = raw_text.replace(" ", "")
        if _AXIS_RE.match(compact) and len(compact) == 1:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            axes.append(
                {
                    "name": compact.upper(),
                    "cx": sum(xs) / len(xs) * ocr_to_orig,
                    "cy": sum(ys) / len(ys) * ocr_to_orig,
                    "conf": conf,
                    "engine": engine,
                }
            )
    dimensions.sort(key=lambda d: (-d["value_mm"], -d["conf"]))
    return dimensions, axes


def _run_engine(kind: str, eng: Any, image_path: Path) -> list[tuple[Any, str, float]]:
    if kind == "paddle":
        raw = eng.ocr(str(image_path), cls=True)
        return _iter_paddle(raw)
    raw, _ = eng(str(image_path))
    return _iter_rapid(raw)


def extract_dimensions_and_axes(
    image_path: Path,
    ocr_to_orig: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    OCR on original/OCR-prep image (never on binarized clean).

    RapidOCR first (reliable for green/CAD digits). If empty, try PaddleOCR.
    """
    # 1) RapidOCR
    rapid = _load_rapid()
    if rapid is not None:
        try:
            items = _run_engine("rapid", rapid, image_path)
            dims, axes = _parse_items(items, ocr_to_orig, "rapid")
            if dims:
                return dims, axes
            logger.info("RapidOCR found no dims on %s; trying Paddle", image_path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RapidOCR failed on %s: %s", image_path.name, exc)

    # 2) Paddle fallback
    paddle = _load_paddle()
    if paddle is not None:
        try:
            items = _run_engine("paddle", paddle, image_path)
            return _parse_items(items, ocr_to_orig, "paddle")
        except Exception as exc:  # noqa: BLE001
            logger.warning("PaddleOCR failed on %s: %s", image_path.name, exc)

    return [], []
