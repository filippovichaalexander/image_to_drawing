"""Stage 3: photo enrichment (OpenCV perspective; Meshroom later)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def enrich_from_photos(
    geometry: dict[str, Any],
    photo_paths: list[Path],
    ai_mode: str = "stub",
) -> dict[str, Any]:
    """
    MVP: record photo count; optional perspective warp lands in a later pass.
    Meshroom is intentionally deferred (heavy, breaks 15–20s SLA).
    """
    geometry = dict(geometry)
    geometry["photos"] = {
        "count": len(photo_paths),
        "mode": ai_mode,
        "note": "Perspective Transform / Meshroom — post-MVP",
    }
    return geometry
