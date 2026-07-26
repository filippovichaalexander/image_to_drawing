#!/usr/bin/env python3
"""Download segmentation weights into models/.

Priority:
1. models/floorplan-seg.pt already present (your fine-tuned / dropped-in weights)
2. Hugging Face floor-plan checkpoints (open weights, not CVC-FP/WAFFLE APIs)
3. Ultralytics yolov8n-seg.pt bootstrap → models/walls_seg.pt

CVC-FP and WAFFLE are *datasets* for training — there is no public inference API
that serves those corpora. Place a trained .pt in models/floorplan-seg.pt instead.
"""

from __future__ import annotations

import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Open Hugging Face checkpoints (direct resolve URLs). First success wins.
HF_FLOORPLAN_CANDIDATES = [
    # Architectural seg with wall/door/window classes (preferred)
    (
        "TanTan2025/archivision-yolo",
        "best.pt",
        "https://huggingface.co/TanTan2025/archivision-yolo/resolve/main/best.pt",
    ),
    # Alternate upload (may still be COCO-named — used only if Archivision fails)
    (
        "TyurinYuriRost/super-cool-model",
        "floor_plan_yolo.pt",
        "https://huggingface.co/TyurinYuriRost/super-cool-model/resolve/main/floor_plan_yolo.pt",
    ),
]


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


def _download(url: str, dest: Path, timeout: int = 180) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "image_to_drawing/0.3"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"  failed: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    if tmp.stat().st_size < 1_000_000:
        print(f"  failed: file too small ({tmp.stat().st_size} bytes)", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    print(f"  wrote {dest} ({dest.stat().st_size} bytes)")
    return True


def _try_hf_floorplan(models: Path) -> Path | None:
    floorplan = models / "floorplan-seg.pt"
    if floorplan.is_file() and floorplan.stat().st_size > 1_000_000:
        return floorplan
    for repo, name, url in HF_FLOORPLAN_CANDIDATES:
        print(f"Trying Hugging Face {repo} ({name})…")
        if _download(url, floorplan):
            return floorplan
    return None


def _bootstrap_coco(models: Path, walls: Path) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed; pip install -r requirements-full.txt", file=sys.stderr)
        return 1

    print("Downloading yolov8n-seg.pt (COCO bootstrap → models/walls_seg.pt)…")
    model = YOLO("yolov8n-seg.pt")
    src = _find_yolo_weight()
    if src is None:
        ckpt = getattr(model, "ckpt_path", None)
        if ckpt and Path(str(ckpt)).is_file():
            src = Path(str(ckpt))
    if src is None:
        import numpy as np

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        model.predict(blank, verbose=False)
        src = _find_yolo_weight()
    if src is None:
        print("Could not locate downloaded yolov8n-seg.pt", file=sys.stderr)
        return 1
    shutil.copy2(src, walls)
    print(f"Wrote {walls} ({walls.stat().st_size} bytes) from {src}")
    print(
        "NOTE: COCO nano-seg is not a floorplan model. "
        "Prefer models/floorplan-seg.pt (auto-download from HF or drop your .pt there)."
    )
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)
    walls = models / "walls_seg.pt"
    floorplan = models / "floorplan-seg.pt"

    if floorplan.is_file() and floorplan.stat().st_size > 1_000_000:
        print(f"Domain model present: {floorplan} ({floorplan.stat().st_size} bytes)")
        if not walls.is_file() or walls.stat().st_size < 1_000_000:
            shutil.copy2(floorplan, walls)
            print(f"Synced → {walls}")
        return 0

    hf = _try_hf_floorplan(models)
    if hf is not None:
        shutil.copy2(hf, walls)
        print(f"Synced floorplan weights → {walls}")
        print("Place your own fine-tuned .pt at models/floorplan-seg.pt to override.")
        return 0

    if walls.is_file() and walls.stat().st_size > 1_000_000:
        print(f"Bootstrap present: {walls} ({walls.stat().st_size} bytes)")
        print("HF floorplan download failed; keeping walls_seg.pt. Retry later or drop floorplan-seg.pt manually.")
        return 0

    return _bootstrap_coco(models, walls)


if __name__ == "__main__":
    raise SystemExit(main())
