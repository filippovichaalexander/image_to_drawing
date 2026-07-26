#!/usr/bin/env python3
"""Download Model 1 (floorplan YOLO) and Model 2 (C3Po) weights into models/.

  python scripts/download_models.py           # floorplan-seg (Model 1)
  python scripts/download_models.py --c3      # C3Po.pth (Model 2, SEPARATE)
  python scripts/download_models.py --all

C3 is photo↔plan alignment only — never train YOLO on the C3 dataset.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HF_FLOORPLAN_CANDIDATES = [
    (
        "TanTan2025/archivision-yolo",
        "best.pt",
        "https://huggingface.co/TanTan2025/archivision-yolo/resolve/main/best.pt",
    ),
    (
        "TyurinYuriRost/super-cool-model",
        "floor_plan_yolo.pt",
        "https://huggingface.co/TyurinYuriRost/super-cool-model/resolve/main/floor_plan_yolo.pt",
    ),
]

C3PO_GDRIVE_ID = "1RhfDKI8Ztwjb-1R0XQpLMqs0Xs4Ax52B"
C3PO_PAGE = "https://github.com/c3po-correspondence/C3Po"


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
    return 0


def download_floorplan(models: Path) -> int:
    walls = models / "walls_seg.pt"
    floorplan = models / "floorplan-seg.pt"

    if floorplan.is_file() and floorplan.stat().st_size > 1_000_000:
        print(f"[Model 1] Domain model present: {floorplan}")
        if not walls.is_file() or walls.stat().st_size < 1_000_000:
            shutil.copy2(floorplan, walls)
            print(f"Synced → {walls}")
        return 0

    hf = _try_hf_floorplan(models)
    if hf is not None:
        shutil.copy2(hf, walls)
        print(f"[Model 1] Synced floorplan weights → {walls}")
        return 0

    if walls.is_file() and walls.stat().st_size > 1_000_000:
        print(f"[Model 1] Bootstrap present: {walls}")
        return 0

    return _bootstrap_coco(models, walls)


def download_c3(models: Path, confirm_large: bool = False) -> int:
    """Model 2 — C3Po photo↔plan alignment (SEPARATE from YOLO).

    Official C3Po.pth is ~10 GB. Refuse unless --confirm-large is set.
    App works with OpenCV ORB fallback without these weights.
    """
    target = models / "C3Po.pth"
    if target.is_file() and target.stat().st_size > 1_000_000:
        print(f"[Model 2 / C3] Already present: {target} ({target.stat().st_size} bytes)")
        return 0

    (models / "C3_DOWNLOAD.txt").write_text(
        f"C3 / C3Po — Model 2 ONLY (photo↔plan). Do NOT train YOLO on C3.\n\n"
        f"Project: {C3PO_PAGE}\n"
        f"Dataset: https://huggingface.co/datasets/kwhuang/C3 (~90k pairs)\n"
        f"Weights (~10 GB): https://drive.google.com/file/d/{C3PO_GDRIVE_ID}/view\n"
        f"Save as: {target}\n\n"
        f"Download:\n"
        f"  python scripts/download_models.py --c3 --confirm-large\n"
        f"Or: gdown {C3PO_GDRIVE_ID} -O {target}\n\n"
        f"Until then the app uses OpenCV ORB photo↔plan (default).\n"
        f"After download, C3Po is picked automatically (see docs/TODO.md).\n"
        f"Force ORB: USE_C3_ALIGN=0\n",
        encoding="utf-8",
    )

    if not confirm_large:
        print(
            "[Model 2 / C3] Skipping auto-download — C3Po.pth is ~10 GB.\n"
            "  Metadata written to models/C3_DOWNLOAD.txt\n"
            "  To fetch: python scripts/download_models.py --c3 --confirm-large\n"
            "  Runtime: OpenCV ORB aligner works without C3Po.pth."
        )
        return 0

    print("[Model 2 / C3] Downloading C3Po.pth (~10 GB, photo alignment ONLY)…")
    print(f"  Source: {C3PO_PAGE}")

    def _via_gdown() -> bool:
        import gdown  # type: ignore

        url = f"https://drive.google.com/uc?id={C3PO_GDRIVE_ID}"
        gdown.download(url, str(target), quiet=False)
        return target.is_file() and target.stat().st_size > 1_000_000_000  # >1GB sanity

    try:
        if _via_gdown():
            shutil.copy2(target, models / "c3-align.pt")
            print(f"[Model 2 / C3] Wrote {target}")
            return 0
    except ImportError:
        print("gdown not installed — trying pip install gdown …")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "-q"], check=True)
            if _via_gdown():
                shutil.copy2(target, models / "c3-align.pt")
                print(f"[Model 2 / C3] Wrote {target}")
                return 0
        except Exception as exc:  # noqa: BLE001
            print(f"gdown failed: {exc}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"C3Po download failed: {exc}", file=sys.stderr)

    print(
        f"[Model 2 / C3] Auto-download failed. Place C3Po.pth at {target}\n"
        f"  Drive: https://drive.google.com/file/d/{C3PO_GDRIVE_ID}/view",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorplan", action="store_true", help="Model 1 YOLO floorplan-seg")
    parser.add_argument("--c3", action="store_true", help="Model 2 C3Po (photo alignment)")
    parser.add_argument(
        "--confirm-large",
        action="store_true",
        help="Required with --c3 to download ~10 GB C3Po.pth",
    )
    parser.add_argument("--all", action="store_true", help="Model 1 + C3 metadata (not 10GB)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    models = root / "models"
    models.mkdir(parents=True, exist_ok=True)

    if not args.floorplan and not args.c3 and not args.all:
        args.floorplan = True

    rc = 0
    if args.all or args.floorplan:
        rc |= download_floorplan(models)
    if args.all or args.c3:
        # --all only writes C3 instructions unless --confirm-large
        rc |= download_c3(models, confirm_large=bool(args.confirm_large and args.c3))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
