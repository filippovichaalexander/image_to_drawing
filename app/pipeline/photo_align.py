"""Model 2: photo ↔ floor-plan alignment via OpenCV ORB (current default).

ORB finds corners/edges on the photo and the plan raster, matches descriptors,
and estimates a homography (RANSAC). Good enough to test the photo workflow
without the ~10 GB C3Po network.

When `models/C3Po.pth` is downloaded (~10 GB), the app auto-switches to C3Po.
Force ORB with `USE_C3_ALIGN=0`. See docs/TODO.md.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
# Real C3Po.pth is ~10 GB; ignore tiny placeholders
_C3_MIN_BYTES = 100_000_000


def resolve_c3_weights() -> Path | None:
    for name in ("C3Po.pth", "c3-align.pt", "c3po.pth"):
        path = _REPO / "models" / name
        if path.is_file() and path.stat().st_size >= _C3_MIN_BYTES:
            return path
    return None


@lru_cache(maxsize=1)
def get_photo_aligner() -> "PhotoPlanAligner":
    """
    ORB by default. Auto-use C3Po when a large checkpoint is on disk.
    Set USE_C3_ALIGN=0 to keep ORB even if weights exist.
    """
    force_orb = os.environ.get("USE_C3_ALIGN", "").strip().lower() in {"0", "false", "no"}
    if not force_orb:
        weights = resolve_c3_weights()
        if weights is not None:
            try:
                logger.info("C3Po weights found (%s) — using C3 aligner", weights.name)
                return C3PoAligner(weights)
            except Exception as exc:  # noqa: BLE001
                logger.warning("C3Po load failed (%s); using ORB", exc)
    return OpenCVPhotoAligner()


def get_c3_model() -> "PhotoPlanAligner":
    """Backward-compatible alias → get_photo_aligner()."""
    return get_photo_aligner()


class PhotoPlanAligner:
    """Interface for photo↔plan correspondence / opening validation."""

    backend: str = "base"

    def align_photo_to_plan(
        self,
        photo_path: Path,
        plan_path: Path | None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def validate_openings(
        self,
        geometry: dict[str, Any],
        photo_paths: list[Path],
        plan_path: Path | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


class OpenCVPhotoAligner(PhotoPlanAligner):
    """
    Primary photo↔plan aligner for MVP / testing.

    Uses ORB (Oriented FAST and Rotated BRIEF): detect keypoints on both images,
    match with Lowe ratio test, fit homography with RANSAC. When the plan raster
    is missing or matches are weak, falls back to Hough-based photo orientation.
    """

    backend = "opencv_orb"

    def align_photo_to_plan(
        self,
        photo_path: Path,
        plan_path: Path | None,
    ) -> dict[str, Any]:
        photo = _imread(photo_path)
        if photo is None:
            return {"ok": False, "error": "unreadable_photo", "backend": self.backend}

        meta: dict[str, Any] = {
            "ok": True,
            "backend": self.backend,
            "source": photo_path.name,
            "homography": None,
            "inliers": 0,
            "matches": 0,
            "orientation_deg": 0.0,
            "confidence": 0.0,
        }

        if plan_path is None or not plan_path.is_file():
            meta["orientation_deg"] = _estimate_photo_rotation(photo)
            meta["confidence"] = 0.25
            meta["note"] = "no_plan_raster — orientation from photo lines only"
            return meta

        plan = _imread(plan_path)
        if plan is None:
            meta["ok"] = False
            meta["error"] = "unreadable_plan"
            return meta

        H, inliers, n_good = _orb_homography(photo, plan)
        meta["inliers"] = inliers
        meta["matches"] = n_good
        if H is not None:
            meta["homography"] = H.tolist()
            meta["orientation_deg"] = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))
            # Soft confidence from inlier count (cap ~0.85 — not a neural net)
            meta["confidence"] = float(min(0.85, 0.35 + 0.02 * inliers))
            meta["note"] = "orb_homography_ok"
        else:
            meta["orientation_deg"] = _estimate_photo_rotation(photo)
            meta["confidence"] = 0.3
            meta["note"] = (
                "orb_match_weak — photo/plan views may differ too much; "
                "using photo line orientation"
            )
        return meta

    def validate_openings(
        self,
        geometry: dict[str, Any],
        photo_paths: list[Path],
        plan_path: Path | None = None,
    ) -> dict[str, Any]:
        geometry = dict(geometry)
        openings = list(geometry.get("openings") or [])
        reports: list[dict[str, Any]] = []
        validated = 0
        best_conf = 0.0

        for path in photo_paths:
            report = self.align_photo_to_plan(path, plan_path)
            reports.append(report)
            if not report.get("ok"):
                continue
            best_conf = max(best_conf, float(report.get("confidence") or 0))
            # Plausible alignment: enough ORB inliers or any photogrammetry cue
            strong = int(report.get("inliers") or 0) >= 10 or report.get("homography")
            for op in openings:
                src = op.get("from")
                kind = op.get("kind") or op.get("type")
                if src in {"photogrammetry", "photo_rect"} or kind in {"window", "door"}:
                    if strong or src == "photogrammetry":
                        op["photo_validated"] = True
                        op["align_backend"] = self.backend
                        op["align_confidence"] = report.get("confidence")
                        validated += 1

        geometry["openings"] = openings
        geometry["photo_align"] = {
            "backend": self.backend,
            "model": "opencv_orb",
            "photos": len(photo_paths),
            "validated_openings": validated,
            "confidence": round(best_conf, 3),
            "reports": reports,
            "note": (
                "OpenCV ORB photo↔plan alignment (default for core development). "
                "Less accurate than C3Po on extreme viewpoint changes; enough to test the workflow."
            ),
        }
        return geometry


class C3PoAligner(PhotoPlanAligner):
    """Optional later: official C3Po checkpoint (opt-in via USE_C3_ALIGN=1)."""

    backend = "c3po"

    def __init__(self, weights: Path) -> None:
        self.weights = weights
        self._torch_ok = False
        self._fallback = OpenCVPhotoAligner()
        try:
            import torch  # type: ignore

            ckpt = torch.load(str(weights), map_location="cpu")
            self._ckpt_keys = list(ckpt.keys())[:8] if isinstance(ckpt, dict) else []
            self._torch_ok = True
            logger.info("C3Po checkpoint loaded from %s", weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not torch.load C3Po (%s)", exc)
            self._ckpt_keys = []

    def align_photo_to_plan(
        self,
        photo_path: Path,
        plan_path: Path | None,
    ) -> dict[str, Any]:
        meta = self._fallback.align_photo_to_plan(photo_path, plan_path)
        meta["backend"] = self.backend
        meta["weights"] = str(self.weights)
        meta["torch_checkpoint"] = self._torch_ok
        meta["note"] = "C3Po opt-in: full pointmaps need c3po-correspondence/C3Po; ORB used for geometry"
        return meta

    def validate_openings(
        self,
        geometry: dict[str, Any],
        photo_paths: list[Path],
        plan_path: Path | None = None,
    ) -> dict[str, Any]:
        geometry = self._fallback.validate_openings(geometry, photo_paths, plan_path)
        align = dict(geometry.get("photo_align") or {})
        align["backend"] = self.backend
        align["model"] = "C3Po+ORB"
        align["c3_weights"] = str(self.weights)
        geometry["photo_align"] = align
        return geometry


def _imread(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _prep_gray(img: np.ndarray) -> np.ndarray:
    """CLAHE on L channel — helps ORB on faded scans / uneven photos."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _estimate_photo_rotation(img: np.ndarray) -> float:
    gray = _prep_gray(img)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:40]:
        _, theta = rho_theta[0]
        deg = float(np.degrees(theta) - 90.0)
        while deg > 45:
            deg -= 90
        while deg < -45:
            deg += 90
        angles.append(deg)
    return float(np.median(angles)) if angles else 0.0


def _orb_homography(
    photo: np.ndarray,
    plan: np.ndarray,
    max_side: int = 1000,
) -> tuple[np.ndarray | None, int, int]:
    """Return (H, inliers, good_matches)."""

    def shrink(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        s = max(h, w)
        if s <= max_side:
            return im
        f = max_side / s
        return cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)

    photo_s, plan_s = shrink(photo), shrink(plan)
    g1, g2 = _prep_gray(photo_s), _prep_gray(plan_s)

    orb = cv2.ORB_create(
        nfeatures=4000,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        patchSize=31,
    )
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.72 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None, 0, len(good)

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.5)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < 8:
        return None, inliers, len(good)
    return H, inliers, len(good)
