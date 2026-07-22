"""Stage 3: photo enrichment — OpenCV perspective / orthophoto heuristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _read_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(4)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def perspective_rectify(src: Path, dst: Path) -> dict[str, Any]:
    """Detect largest quadrilateral and warp to a frontal view (ТЗ option A)."""
    img = _read_bgr(src)
    if img is None:
        raise ValueError(f"Cannot read photo: {src.name}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    quad = None
    for cnt in contours[:20]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * w * h:
            quad = approx
            break

    meta: dict[str, Any] = {"source": src.name, "warped": False}
    if quad is not None:
        corners = _order_corners(quad)
        width_a = np.linalg.norm(corners[2] - corners[3])
        width_b = np.linalg.norm(corners[1] - corners[0])
        height_a = np.linalg.norm(corners[1] - corners[2])
        height_b = np.linalg.norm(corners[0] - corners[3])
        max_w = max(int(max(width_a, width_b)), 64)
        max_h = max(int(max(height_a, height_b)), 64)
        dst_pts = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(corners, dst_pts)
        out = cv2.warpPerspective(img, matrix, (max_w, max_h))
        meta["warped"] = True
        meta["size"] = [max_w, max_h]
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        out = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        meta["note"] = "no_quad_clahe_fallback"

    dst.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("Failed to encode rectified photo")
    buf.tofile(str(dst))
    meta["output"] = dst.name
    return meta


def enrich_from_photos(
    geometry: dict[str, Any],
    photo_paths: list[Path],
    ai_mode: str = "stub",
    results_dir: Path | None = None,
) -> dict[str, Any]:
    geometry = dict(geometry)
    rect_dir = (results_dir or Path("/tmp/itd_photos")) / "photos_rectified"
    rect_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for idx, path in enumerate(photo_paths):
        try:
            reports.append(perspective_rectify(path, rect_dir / f"photo_{idx:02d}_rect.png"))
        except Exception as exc:  # noqa: BLE001
            reports.append({"source": path.name, "error": str(exc)})

    openings = list(geometry.get("openings") or [])
    if any(r.get("warped") for r in reports):
        openings.append(
            {
                "type": "facade_enrichment",
                "from": "photogrammetry",
                "note": "Perspective-rectified photos available for detail pass",
            }
        )

    geometry["openings"] = openings
    geometry["photos"] = {
        "count": len(photo_paths),
        "mode": ai_mode,
        "reports": reports,
        "note": "OpenCV Perspective Transform; Meshroom — post-MVP",
    }
    return geometry
