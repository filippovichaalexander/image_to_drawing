#!/usr/bin/env bash
# Install a YOLO-seg checkpoint into models/ and reload the web app.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-}"
DEST_DIR="${ROOT}/models"
MIN_BYTES=$((1024 * 1024))

if [[ -z "${SRC}" ]]; then
  echo "Usage: $0 <path-to-best.pt>"
  echo "Example: $0 ~/Downloads/best.pt"
  exit 1
fi

if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: file not found: ${SRC}"
  exit 1
fi

SIZE="$(wc -c < "${SRC}" | tr -d ' ')"
if [[ "${SIZE}" -lt "${MIN_BYTES}" ]]; then
  echo "ERROR: ${SRC} is only ${SIZE} bytes (< 1 MiB) — looks corrupted"
  exit 1
fi

mkdir -p "${DEST_DIR}/backup"
STAMP="$(date +%Y%m%d-%H%M%S)"
if [[ -f "${DEST_DIR}/floorplan-seg.pt" ]]; then
  cp -f "${DEST_DIR}/floorplan-seg.pt" "${DEST_DIR}/backup/floorplan-seg.${STAMP}.pt"
fi

cp -f "${SRC}" "${DEST_DIR}/floorplan-seg.pt"
cp -f "${SRC}" "${DEST_DIR}/walls_seg.pt"
cp -f "${SRC}" "${DEST_DIR}/floorplan-seg.colab.pt"

echo "Installed ${DEST_DIR}/floorplan-seg.pt ($(du -h "${DEST_DIR}/floorplan-seg.pt" | cut -f1))"

cd "${ROOT}"
if docker compose ps --status running 2>/dev/null | grep -q image_to_drawing_web; then
  docker compose restart web
  echo "Restarted docker compose service: web"
elif docker ps --format '{{.Names}}' | grep -qx image_to_drawing_web; then
  docker restart image_to_drawing_web
  echo "Restarted container: image_to_drawing_web"
else
  echo "WARN: web container not running — start with: docker compose up -d"
fi

echo
echo "Smoke test (inside container):"
echo "  docker exec image_to_drawing_web yolo predict model=/app/models/floorplan-seg.pt source=/app/data/uploads/<your>.jpg"
echo "Or open http://127.0.0.1:8000 and upload a plan."
