# Architecture notes (TZ collisions & decisions)

Status labels:
- **Resolved now** — decision shipped in the current app; no action needed.
- **Future** — still open or deferred; see [`TODO.md`](./TODO.md).

---

## Коллизии ТЗ

### 1. Lightweight Docker vs heavy AI stack — **Resolved now**

| ТЗ asked for | What we ship |
|--------------|--------------|
| ControlNet + SAM + Meshroom + full PyTorch always on | Optional profile |

**Decision (done):** `AI_MODE=stub|full` + `INSTALL_FULL_AI` for Docker.
- `stub` — OpenCV + RapidOCR, light image.
- `full` — YOLOv8-seg + PaddleOCR/RapidOCR; ControlNet/SAM/Meshroom **not** in the image.
- Model 2 photo align = **ORB now**; C3Po (~10 GB) only if you opt in later ([`TODO.md`](./TODO.md)).

**Still future:** thinner ONNX-only image; GPU worker for heavy jobs.

---

### 2. 15–20 s SLA (3 drawings + 2 photos) — **Partially resolved / Future**

| Now | Future |
|-----|--------|
| Downscale ≤1600/1280, OpenCV denoise, lazy YOLO/OCR imports, PDF page cap | Process-pool for multi-sheet; ROI-OCR; avoid loading C3Po on CPU in the hot path |

ControlNet-on-CPU is **out of scope** (was never shipped). Meeting hard 15–20 s on CPU with `AI_MODE=full` is still not guaranteed — treat as a **Future** perf goal, not a current regression.

---

### 3. DWG vs DXF — **Resolved now**

ТЗ said DWG. Product delivers **DXF only** (`ezdxf`). LibreDWG/ODA removed.
DXF opens in AutoCAD / LibreCAD. No further work unless a customer mandates native DWG.

---

### 4. «No Node» UI vs WebGL viewer — **Resolved now** (was never a real conflict)

FastAPI + Jinja2; viewer JS from CDN. No Node build. Keep as-is.

---

### 5. Meshroom API for photos — **Resolved now (MVP) / Future (optional)**

| Now | Future |
|-----|--------|
| OpenCV perspective + facade cues + **ORB** photo↔plan | Meshroom / full SfM only if ORB+C3Po still insufficient ([`TODO.md`](./TODO.md) — low priority vs C3Po) |

Meshroom stays **out** of the default pipeline.

---

### 6. Public cloud AI APIs — **Resolved now**

Stack is local CV (OpenCV / YOLO / OCR). No OpenAI Vision (etc.) in the main path.
Cloud APIs remain an optional experiment only — **not planned** for ГОСТ core.

---

## Оптимизации

### Already applied (now)

- Stub-first / DXF-only Docker (no LibreDWG)
- Downscale-then-scale (CV ≤1600 / photos ≤1280)
- PDF: first N pages @ 150 DPI
- Lazy imports for YOLO/OCR
- R2000 DXF for broad CAD compatibility
- Layered Model 1 (floorplan YOLO) vs Model 2 (ORB; C3Po later)
- OCR scale from dimension chains; no silent 12 000 mm
- Contour post-process + `closedness` report

### Future (see [`TODO.md`](./TODO.md))

- **High impact:** C3Po photo↔plan (`download_models.py --c3 --confirm-large`)
- Fine-tune `floorplan-seg.pt`: `prepare_yolo_dataset.py` + `train_floorplan_seg.py` (CVC-FP + R2V synthetic)
- Full C3Po forward pass (not checkpoint + ORB hybrid)
- ROI-OCR around dimension chains
- ONNX nano denoise / wall-seg for lighter images
- Parallel sheets (process pool) when ≥2 CPUs

---

## Реализовано (v0.3+)

- WebGL viewer with layer toggles
- OpenCV perspective + facade window cues + ORB validation
- DXF via ezdxf; optional ZIP package
- `AI_MODE=full` → YOLO+OCR with stub fallback
- Pipeline: `preprocess` → `segmentation` → `ocr` → `scale` → `postprocess` → (`photogrammetry` + `fusion` if photos)
- Weights: `models/floorplan-seg.pt` (CVC-FP+R2V fine-tune preferred; else HF Archivision)
- Datasets: `scripts/download_datasets.py` + `prepare_yolo_dataset.py` → `datasets/combined/`
- PDF via pypdfium2
- Stub ortho walls + hatch suppression
- Control length UI; `needs_scale` + `scale_warning`
- Cross-validation (module voting, axes, elevations)
- GOST Form 3 stamp (+ portrait fallback)

---

## Паттерны лёгкости / скорости (current policy)

1. **Stub-first / DXF-only** — no LibreDWG in Docker.
2. **Downscale-then-scale** — CV on ≤1600/1280 px.
3. **PDF cap** — first N pages @ 150 DPI.
4. **Lazy imports** — YOLO/OCR only when needed.
5. **R2000 DXF** — wide CAD client support.
6. **ORB before C3Po** — ship photo workflow without a 10 GB dependency.
