# Future development TODO

Prioritized follow-ups that improve quality once the ORB-based MVP is solid.

## High impact — C3 / C3Po photo↔plan (Model 2)

OpenCV ORB is the **current default** and is enough to build and test the core app.
When photo-to-plan accuracy becomes the bottleneck (strong viewpoint / modality gap),
enable the specialized **C3Po** network. This is what the layered-approach instructions
describe and is expected to give the largest gain on the photo branch.

### Steps

1. **Run the download** (stable network, ~10 GB free disk):

   ```bash
   python scripts/download_models.py --c3 --confirm-large
   ```

2. **Wait for the download** — `models/C3Po.pth` (~10 GB). Do not interrupt; partial files are useless.

3. **Enable automatically** — once `models/C3Po.pth` is present and large enough, the app
   picks C3Po instead of ORB (no code change). To force ORB again: `USE_C3_ALIGN=0`.

4. **Optional full stack** — for true C3Po pointmaps (not ORB wrapped with a checkpoint),
   vendor [c3po-correspondence/C3Po](https://github.com/c3po-correspondence/C3Po) + DUSt3R
   and wire a real forward pass in `app/pipeline/photo_align.py` (`C3PoAligner`).

### Why it helps

| | ORB (now) | C3Po (later) |
|--|-----------|--------------|
| Speed / size | Fast, no download | Slow cold start, ~10 GB |
| Same-ish views | Often OK | Stronger |
| Photo vs abstract plan | Weak | Trained for this |
| Extreme angles | Often fails | Much better |

**Do not** train YOLO floor-plan seg on the C3 dataset — C3 is Model 2 only.

Related: `ML-instructions-approaches.txt`, `datasets/c3/`, `models/C3_DOWNLOAD.txt`.

---

## Other quality TODOs

- [x] Download CVC-FP + convert with R2V synthetic → YOLO-seg (`scripts/prepare_yolo_dataset.py`)
- [x] Fine-tune / install `models/floorplan-seg.pt` via `scripts/train_floorplan_seg.py` (CVC-FP+R2V; longer resume may still improve)
- [ ] Wire full C3Po inference (not checkpoint+ORB hybrid) after weights land
- [ ] ROI-OCR around dimension chains only
- [ ] ONNX nano denoise / wall-seg for lighter Docker images
- [ ] Parallel sheet processing (process pool) when ≥2 CPUs

### Model 1 training (walls / doors / windows)

```bash
python scripts/download_datasets.py --cvc-fp --r2v
python scripts/prepare_yolo_dataset.py --clean
# Prefer GPU host or Colab; inside Docker (CPU):
docker exec -e MPLCONFIGDIR=/tmp/mpl -e HOME=/tmp image_to_drawing_web \
  python /app/scripts/train_floorplan_seg.py --epochs 40 --batch 4 --device cpu
docker compose restart web
```

R2V does **not** ship LIFULL rasters; the prepare script renders vector annotations to synthetic plans for diversity. CubiCasa5K remains optional (walls/doors/windows only).
