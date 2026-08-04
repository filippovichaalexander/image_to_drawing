#!/usr/bin/env python3
"""Download SAM weights for optional mask refinement."""

from __future__ import annotations

import urllib.request
from pathlib import Path

SAM_WEIGHTS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def download_sam(model_type: str = "vit_b", models_dir: Path | None = None) -> Path:
    models_dir = models_dir or Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    url = SAM_WEIGHTS.get(model_type)
    if not url:
        raise ValueError(f"Unknown model type: {model_type}")
    dest = models_dir / f"sam_{model_type}.pth"
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"SAM weights already exist: {dest}")
        return dest
    print(f"Downloading SAM {model_type} from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded: {dest}")
    return dest


if __name__ == "__main__":
    download_sam("vit_b")
