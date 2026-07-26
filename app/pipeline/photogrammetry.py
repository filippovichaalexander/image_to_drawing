"""Stage 3: photo enrichment — OpenCV perspective + light opening cues."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

MAX_PHOTO_SIDE = 1280


def _read_bgr(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _downscale_bgr(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= MAX_PHOTO_SIDE:
        return img, 1.0
    factor = MAX_PHOTO_SIDE / long_side
    small = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return small, 1.0 / factor


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

    work, _ = _downscale_bgr(img)
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
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
        out = cv2.warpPerspective(work, matrix, (max_w, max_h))
        meta["warped"] = True
        meta["size"] = [max_w, max_h]
    else:
        lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
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


def _detect_facade_openings(rectified_path: Path, max_count: int = 16) -> list[dict[str, Any]]:
    """
    Cheap window-grid cues on a rectified facade (ТЗ option B lite).
    Dark rectangular regions → candidate openings; mapped later in mm space.
    """
    img = _read_bgr(rectified_path)
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Windows are often darker than wall
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Plan footprint span used as horizontal reference for placement
    cands: list[dict[str, Any]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < 0.002 * w * h or area > 0.12 * w * h:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.35 or aspect > 3.5:
            continue
        cands.append(
            {
                "type": "window",
                "nx": (x + bw / 2) / w,  # 0..1 along facade
                "ny": 1.0 - (y + bh / 2) / h,
                "nw": bw / w,
                "nh": bh / h,
                "from": "photo_rect",
            }
        )
        if len(cands) >= max_count:
            break
    return cands


def enrich_from_photos(
    geometry: dict[str, Any],
    photo_paths: list[Path],
    ai_mode: str = "stub",
    results_dir: Path | None = None,
    plan_path: Path | None = None,
) -> dict[str, Any]:
    """
    Photo branch of the layered stack (separate from YOLO floor-plan parsing).

    1) OpenCV perspective rectify + facade window cues
    2) OpenCV ORB photo↔plan aligner validates openings (Model 2 default)
    """
    geometry = dict(geometry)
    rect_dir = (results_dir or Path("/tmp/itd_photos")) / "photos_rectified"
    rect_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    facade_cues: list[dict[str, Any]] = []
    for idx, path in enumerate(photo_paths):
        try:
            out = rect_dir / f"photo_{idx:02d}_rect.png"
            report = perspective_rectify(path, out)
            reports.append(report)
            if report.get("warped") or report.get("output"):
                facade_cues.extend(_detect_facade_openings(out))
        except Exception as exc:  # noqa: BLE001
            reports.append({"source": path.name, "error": str(exc)})

    openings = list(geometry.get("openings") or [])
    walls = geometry.get("walls") or []
    xs = [p[0] for wall in walls for p in wall] or [0.0, 12000.0]
    ys = [p[1] for wall in walls for p in wall] or [0.0, 8000.0]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    wall_y = min_y
    depth = max(span_x * 0.02, 200.0)

    added = 0
    for cue in facade_cues[:12]:
        cx = min_x + float(cue["nx"]) * span_x
        half_w = max(float(cue["nw"]) * span_x * 0.5, 400.0)
        openings.append(
            {
                "type": cue.get("type", "window"),
                "kind": "window",
                "p1": [cx - half_w, wall_y - depth],
                "p2": [cx + half_w, wall_y + depth],
                "from": "photogrammetry",
                "facade_ny": cue.get("ny"),
            }
        )
        added += 1

    if added == 0 and any(r.get("warped") for r in reports):
        openings.append(
            {
                "type": "facade_enrichment",
                "from": "photogrammetry",
                "note": "Perspective-rectified photos available; no window cues found",
            }
        )

    geometry["openings"] = openings
    geometry["photos"] = {
        "count": len(photo_paths),
        "mode": ai_mode,
        "reports": reports,
        "openings_added": added,
        "note": "OpenCV perspective + facade cues; ORB aligner validates openings",
    }

    # Model 2 — ORB photo↔plan (C3Po ignored unless USE_C3_ALIGN=1)
    from app.pipeline.fusion import fuse_geometry_with_photos

    return fuse_geometry_with_photos(
        geometry,
        photo_paths,
        plan_path=plan_path,
        ai_mode=ai_mode,
    )
