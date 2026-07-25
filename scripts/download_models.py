#!/usr/bin/env python3
"""Download bootstrap segmentation weights into models/walls_seg.pt."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    target = models / "walls_seg.pt"

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed; pip install -r requirements-full.txt", file=sys.stderr)
        return 1

    # Official nano-seg as bootstrap until a domain-tuned walls_seg exists
    print("Downloading yolov8n-seg.pt (bootstrap for walls_seg.pt)…")
    model = YOLO("yolov8n-seg.pt")
    # Ultralytics caches under cwd / weights; resolve actual file
    src = Path("yolov8n-seg.pt")
    if not src.exists():
        # common cache locations
        candidates = list(Path(".").glob("**/yolov8n-seg.pt")) + list(
            Path.home().joinpath(".cache").glob("**/yolov8n-seg.pt")
        )
        if not candidates:
            # model.ckpt_path if available
            ckpt = getattr(model, "ckpt_path", None) or getattr(getattr(model, "model", None), "pt_path", None)
            if ckpt and Path(str(ckpt)).exists():
                src = Path(str(ckpt))
            else:
                print("Could not locate downloaded yolov8n-seg.pt", file=sys.stderr)
                return 1
        else:
            src = candidates[0]

    shutil.copy2(src, target)
    print(f"Wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
