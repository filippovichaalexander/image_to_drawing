"""Post-YOLO / post-photo fusion (layered approach stage)."""

from __future__ import annotations

from typing import Any

from app.pipeline.postprocess import postprocess_walls


def merge_overlapping_walls(
    walls: list[list[list[float]]],
    partitions: list[list[list[float]]] | None = None,
) -> tuple[list[list[list[float]]], list[list[list[float]]], dict[str, Any]]:
    """Delegate to contour post-process (spike/gap/closedness)."""
    return postprocess_walls(walls, partitions)


def fuse_geometry_with_photos(
    geometry: dict[str, Any],
    photo_paths: list,
    plan_path=None,
    ai_mode: str = "stub",
) -> dict[str, Any]:
    """
    Photo branch — only when photos exist.
    Default aligner: OpenCV ORB (C3Po optional later via USE_C3_ALIGN=1).
    """
    if not photo_paths:
        return geometry

    from app.pipeline.photo_align import get_photo_aligner

    aligner = get_photo_aligner()
    geometry = aligner.validate_openings(geometry, list(photo_paths), plan_path=plan_path)
    geometry["fusion"] = {
        "photo_align_backend": (geometry.get("photo_align") or {}).get("backend"),
        "ai_mode": ai_mode,
        "layered": True,
        "models": {
            "floorplan": "YOLOv8-seg (Model 1)",
            "photo_align": "OpenCV ORB (Model 2 default; C3Po optional later)",
        },
    }
    return geometry
