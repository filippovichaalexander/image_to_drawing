# Datasets (layered approach)

| Priority | Dataset | Role | Path |
|----------|---------|------|------|
| MUST | **CVC-FP** | Model 1 — walls/doors/windows | `datasets/cvc_fp/` |
| MUST | **R2V** | Model 1 — diversity (~800) | `datasets/r2v/` (via FloorplanTransformation) |
| NICE | **CubiCasa5K** | Model 1 — wall/window only | `datasets/cubicasa5k/` |
| **SEPARATE** | **C3** (~90k pairs) | **Model 2 only** — photo↔plan | `datasets/c3/` |
| SKIP | FloorPlanCAD | too many classes | — |

## Model 2 — C3 (do not train YOLO on this)

- Project: https://c3po-correspondence.github.io/
- HuggingFace: https://huggingface.co/datasets/kwhuang/C3
- Code + `C3Po.pth`: https://github.com/c3po-correspondence/C3Po.git
- App weights path: `models/C3Po.pth` (`python scripts/download_models.py --c3`)

C3 is cross-view correspondence data. Mixing it into segmentation confuses the wall parser.

## Commands

```bash
python scripts/download_datasets.py --cvc-fp --r2v --c3-meta
python scripts/download_models.py --c3   # C3Po.pth → models/
```
