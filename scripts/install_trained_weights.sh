#!/usr/bin/env bash
# Copy trained CVC-FP+R2V YOLO weights into models/ and restart web.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-$ROOT/data/results/floorplan_train/floorplan-seg.pt}"
if [[ ! -f "$SRC" ]]; then
  # Prefer ultralytics best.pt
  CAND="$ROOT/data/results/floorplan_train/cvc_r2v/weights/best.pt"
  if [[ -f "$CAND" ]]; then
    SRC="$CAND"
  else
    echo "No weights at $SRC or $CAND — finish training first." >&2
    exit 1
  fi
fi
mkdir -p "$ROOT/models"
cp -f "$SRC" "$ROOT/models/floorplan-seg.pt"
cp -f "$SRC" "$ROOT/models/walls_seg.pt"
echo "Installed $SRC → models/floorplan-seg.pt"
if docker ps --format '{{.Names}}' | grep -q '^image_to_drawing_web$'; then
  docker restart image_to_drawing_web
  echo "Restarted image_to_drawing_web (clears YOLO LRU cache)."
fi
