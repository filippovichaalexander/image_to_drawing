#!/usr/bin/env python3
"""Download bootstrap segmentation weights into models/walls_seg.pt."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _find_yolo_weight() -> Path | None:
    candidates = [
        Path("yolov8n-seg.pt"),
        Path("/app/yolov8n-seg.pt"),
        *Path(".").glob("**/yolov8n-seg.pt"),
        *Path.home().joinpath(".cache").glob("**/yolov8n-seg.pt"),
    ]
    try:
        from ultralytics.utils import WEIGHTS_DIR  # type: ignore

        candidates.append(Path(WEIGHTS_DIR) / "yolov8n-seg.pt")
    except Exception:  # noqa: BLE001
        pass
    for path in candidates:
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return None


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    target = models / "walls_seg.pt"

    if target.is_file() and target.stat().st_size > 1_000_000:
        print(f"Already present: {target} ({target.stat().st_size} bytes)")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed; pip install -r requirements-full.txt", file=sys.stderr)
        return 1

    print("Downloading yolov8n-seg.pt (bootstrap → models/walls_seg.pt)…")
    model = YOLO("yolov8n-seg.pt")

    src = _find_yolo_weight()
    if src is None:
        ckpt = getattr(model, "ckpt_path", None)
        if ckpt and Path(str(ckpt)).is_file():
            src = Path(str(ckpt))
    if src is None:
        # Force predict once — ensures weight materializes on disk
        import numpy as np

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        model.predict(blank, verbose=False)
        src = _find_yolo_weight()

    if src is None:
        print("Could not locate downloaded yolov8n-seg.pt", file=sys.stderr)
        return 1

    shutil.copy2(src, target)
    print(f"Wrote {target} ({target.stat().st_size} bytes) from {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
