#!/usr/bin/env python3
"""Fine-tune YOLOv8-seg on CVC-FP + R2V (datasets/combined).

Produces models/floorplan-seg.pt for the app (AI_MODE=full).

Usage (inside Docker recommended — has ultralytics + torch):
  python scripts/prepare_yolo_dataset.py --clean
  python scripts/train_floorplan_seg.py
  python scripts/train_floorplan_seg.py --epochs 50 --imgsz 640 --batch 4
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "datasets" / "combined" / "data.yaml"
MODELS = ROOT / "models"
RUNS = ROOT / "runs" / "floorplan_seg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument(
        "--project",
        type=Path,
        default=RUNS.parent,
        help="Ultralytics project directory",
    )
    parser.add_argument("--run-name", default=RUNS.name, help="Ultralytics run name")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu | 0 | 0,1")
    parser.add_argument("--model", default="yolov8n-seg.pt", help="base checkpoint")
    parser.add_argument(
        "--from-checkpoint",
        type=Path,
        default=None,
        help="Resume fine-tune from a same-taxonomy .pt (not Archivision)",
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=MODELS,
        help="Where to copy best.pt as floorplan-seg.pt",
    )
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume last run under runs/floorplan_seg",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        print(
            f"Missing {args.data}. Run: python scripts/prepare_yolo_dataset.py",
            file=sys.stderr,
        )
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics required. Use Docker or: pip install ultralytics", file=sys.stderr)
        return 1

    MODELS.mkdir(parents=True, exist_ok=True)
    install_dir = Path(args.install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    base = args.model
    if args.from_checkpoint:
        ckpt = Path(args.from_checkpoint)
        if not ckpt.is_file():
            print(f"Checkpoint missing: {ckpt}", file=sys.stderr)
            return 1
        print(f"Continuing from {ckpt} (must match class order in data.yaml)")
        base = str(ckpt)
    else:
        # Do NOT auto-load Archivision floorplan-seg.pt — class ids differ
        # (window/door/column/wall/beam vs wall/door/window/stair/partition).
        print(f"Training from base weights: {base}")

    model = YOLO(base)
    print(
        f"Training YOLO-seg: data={args.data} epochs={args.epochs} "
        f"imgsz={args.imgsz} batch={args.batch} device={args.device}"
    )
    results = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        project=str(Path(args.project).resolve()),
        name=str(args.run_name),
        exist_ok=True,
        resume=bool(args.resume),
        plots=False,
        verbose=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.run_name))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        candidates = list(Path(args.project).glob("**/weights/best.pt"))
        if candidates:
            best = max(candidates, key=lambda p: p.stat().st_mtime)
    if not best.is_file():
        print(f"Training finished but best.pt not found under {args.project}", file=sys.stderr)
        return 1

    dest = install_dir / "floorplan-seg.pt"
    shutil.copy2(best, dest)
    walls = install_dir / "walls_seg.pt"
    try:
        shutil.copy2(best, walls)
    except OSError:
        walls = None
    print(f"Installed {dest}" + (f" (and {walls})" if walls else ""))
    print("Restart the web container / process so it reloads YOLO weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
