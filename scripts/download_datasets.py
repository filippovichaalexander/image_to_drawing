#!/usr/bin/env python3
"""Download / scaffold datasets for the layered training strategy.

Model 1 (floor-plan YOLO): CVC-FP + R2V (+ CubiCasa walls only)
Model 2 (photo alignment):  C3 — SEPARATE, never mixed into YOLO training

Usage:
  python scripts/download_datasets.py              # scaffold + fetch what we can
  python scripts/download_datasets.py --cvc-fp --r2v
  python scripts/download_datasets.py --c3-meta     # C3 index via HuggingFace (small)
  python scripts/download_datasets.py --c3-full     # FULL C3 visual (~huge) — opt-in

See datasets/README.md and ML-instructions-approaches.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"

# Official / community entry points (URLs change — README has fallbacks)
CVC_FP_PAGE = "http://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/"
CVC_FP_ZIP = "http://dag.cvc.uab.es/DATASETS/CVC-FP.zip"
R2V_GIT = "https://github.com/art-programmer/FloorplanTransformation.git"
CUBICASA_PAGE = "https://github.com/CubiCasa/CubiCasa5k"
CUBICASA_ZIP = "https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1"
C3_HF = "https://huggingface.co/datasets/kwhuang/C3"
C3_PROJECT = "https://c3po-correspondence.github.io/"
C3PO_GIT = "https://github.com/c3po-correspondence/C3Po.git"


def _write_readme() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    text = f"""# Datasets (layered approach)

| Priority | Dataset | Role | Path |
|----------|---------|------|------|
| MUST | **CVC-FP** | Model 1 — walls/doors/windows | `datasets/cvc_fp/` |
| MUST | **R2V** | Model 1 — diversity (~800) | `datasets/r2v/` (via FloorplanTransformation) |
| NICE | **CubiCasa5K** | Model 1 — wall/window only | `datasets/cubicasa5k/` |
| **SEPARATE** | **C3** (~90k pairs) | **Model 2 only** — photo↔plan | `datasets/c3/` |
| SKIP | FloorPlanCAD | too many classes | — |

## Model 2 — C3 (do not train YOLO on this)

- Project: {C3_PROJECT}
- HuggingFace: {C3_HF}
- Code + `C3Po.pth`: {C3PO_GIT}
- App weights path: `models/C3Po.pth` (`python scripts/download_models.py --c3`)

C3 is cross-view correspondence data. Mixing it into segmentation confuses the wall parser.

## Commands

```bash
python scripts/download_datasets.py --cvc-fp --r2v --c3-meta
python scripts/prepare_yolo_dataset.py --clean   # CVC-FP SVG + R2V synthetic → YOLO-seg
python scripts/train_floorplan_seg.py            # → models/floorplan-seg.pt
python scripts/download_models.py --c3           # C3Po.pth → models/ (optional Model 2)
```
"""
    (DATASETS / "README.md").write_text(text, encoding="utf-8")
    print(f"Wrote {DATASETS / 'README.md'}")


def fetch_cvc_fp() -> int:
    """Download + extract official CVC-FP (~27 MB) into datasets/cvc_fp/ImagesGT."""
    import shutil
    import zipfile

    dest = DATASETS / "cvc_fp"
    dest.mkdir(parents=True, exist_ok=True)
    images = dest / "ImagesGT"
    meta = {
        "name": "CVC-FP",
        "role": "model1_floorplan_seg",
        "page": CVC_FP_PAGE,
        "zip": CVC_FP_ZIP,
        "status": "downloaded" if images.is_dir() and any(images.glob("*.png")) else "pending",
        "note": "PNG + SVG GT (walls/doors/windows). Convert via scripts/prepare_yolo_dataset.py",
    }
    if images.is_dir() and len(list(images.glob("*.png"))) > 50:
        meta["status"] = "present"
        (dest / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"CVC-FP already present → {images} ({len(list(images.glob('*.png')))} PNGs)")
        return 0

    zip_path = dest / "CVC-FP.zip"
    print(f"Downloading CVC-FP from {CVC_FP_ZIP} …")
    try:
        urllib.request.urlretrieve(CVC_FP_ZIP, zip_path)
    except Exception as exc:  # noqa: BLE001
        meta["status"] = "download_failed"
        meta["error"] = str(exc)
        (dest / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"CVC-FP download failed ({exc}). Manual: {CVC_FP_ZIP}", file=sys.stderr)
        return 1

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
    except zipfile.BadZipFile as exc:
        print(f"Bad zip: {exc}", file=sys.stderr)
        return 1
    finally:
        zip_path.unlink(missing_ok=True)

    # Normalize layout if zip nested oddly
    if not images.is_dir():
        nested = next(dest.rglob("ImagesGT"), None)
        if nested and nested.is_dir() and nested != images:
            shutil.move(str(nested), str(images))

    n = len(list(images.glob("*.png"))) if images.is_dir() else 0
    meta["status"] = "downloaded" if n else "empty"
    meta["png_count"] = n
    (dest / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"CVC-FP → {dest} ({n} PNGs)")
    print("Next: python scripts/prepare_yolo_dataset.py && python scripts/train_floorplan_seg.py")
    return 0 if n else 1


def fetch_r2v() -> int:
    dest = DATASETS / "r2v"
    dest.mkdir(parents=True, exist_ok=True)
    repo = dest / "FloorplanTransformation"
    if (repo / ".git").is_dir():
        print(f"R2V repo already present: {repo}")
        return 0
    print(f"Cloning R2V / FloorplanTransformation → {repo}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", R2V_GIT, str(repo)],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        (dest / "SOURCE.json").write_text(
            json.dumps({"git": R2V_GIT, "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        print(f"Clone failed ({exc}). Clone manually: git clone {R2V_GIT}", file=sys.stderr)
        return 1
    (dest / "SOURCE.json").write_text(
        json.dumps(
            {
                "git": R2V_GIT,
                "role": "model1_floorplan_seg",
                "note": (
                    "LIFULL rasters are not redistributable. "
                    "scripts/prepare_yolo_dataset.py synthesizes RGB + YOLO labels "
                    "from data/floorplan_representation/*.txt"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def fetch_cubicasa_meta() -> int:
    dest = DATASETS / "cubicasa5k"
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": "CubiCasa5K",
        "role": "model1_walls_windows_only",
        "git": CUBICASA_PAGE,
        "note": "Use only wall/door/window classes — skip furniture.",
    }
    (dest / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"CubiCasa5K scaffold → {dest}")
    print(f"  {CUBICASA_PAGE}")
    return 0


def fetch_c3_meta() -> int:
    """Small C3 pointer + optional HF dataset card (NOT full 90k visuals)."""
    dest = DATASETS / "c3"
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": "C3",
        "role": "model2_photo_plan_alignment_ONLY",
        "pairs": 90_000,
        "huggingface": C3_HF,
        "project": C3_PROJECT,
        "code": C3PO_GIT,
        "weights": "models/C3Po.pth via scripts/download_models.py --c3",
        "warning": "Do NOT mix into YOLOv8 floor-plan training.",
    }
    (dest / "SOURCE.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    # Try to pull the dataset card README for offline reference
    readme = dest / "HF_DATASET_CARD.md"
    try:
        url = "https://huggingface.co/datasets/kwhuang/C3/raw/main/README.md"
        with urllib.request.urlopen(url, timeout=60) as resp:
            readme.write_bytes(resp.read())
        print(f"C3 meta + HF card → {dest}")
    except Exception as exc:  # noqa: BLE001
        print(f"C3 SOURCE.json written; HF card skip ({exc})")
    print(f"  Full visual download: python scripts/download_datasets.py --c3-full")
    print(f"  Weights: python scripts/download_models.py --c3")
    return 0


def fetch_c3_full() -> int:
    """Opt-in full C3 via huggingface_hub (large)."""
    dest = DATASETS / "c3"
    dest.mkdir(parents=True, exist_ok=True)
    fetch_c3_meta()
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError:
        print(
            "huggingface_hub not installed. pip install huggingface_hub\n"
            f"Or browse {C3_HF} and download geometric/ + visual/ manually.",
            file=sys.stderr,
        )
        return 1
    print("Downloading C3 snapshot (this can be tens of GB)…")
    snapshot_download(
        repo_id="kwhuang/C3",
        repo_type="dataset",
        local_dir=str(dest / "hf_snapshot"),
    )
    print(f"C3 full snapshot → {dest / 'hf_snapshot'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cvc-fp", action="store_true", help="Scaffold CVC-FP (Model 1)")
    parser.add_argument("--r2v", action="store_true", help="Clone R2V FloorplanTransformation")
    parser.add_argument("--cubicasa", action="store_true", help="Scaffold CubiCasa5K")
    parser.add_argument(
        "--c3-meta",
        action="store_true",
        help="C3 SEPARATE model-2 metadata (recommended)",
    )
    parser.add_argument(
        "--c3-full",
        action="store_true",
        help="Download full C3 from HuggingFace (huge, Model 2 only)",
    )
    parser.add_argument("--all-small", action="store_true", help="cvc-fp + r2v + cubicasa + c3-meta")
    args = parser.parse_args()

    _write_readme()
    if not any([args.cvc_fp, args.r2v, args.cubicasa, args.c3_meta, args.c3_full, args.all_small]):
        args.all_small = True

    rc = 0
    if args.all_small or args.cvc_fp:
        rc |= fetch_cvc_fp()
    if args.all_small or args.r2v:
        rc |= fetch_r2v()
    if args.all_small or args.cubicasa:
        rc |= fetch_cubicasa_meta()
    if args.all_small or args.c3_meta:
        rc |= fetch_c3_meta()
    if args.c3_full:
        rc |= fetch_c3_full()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
