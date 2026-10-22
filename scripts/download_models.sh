#!/usr/bin/env bash
# Pretrained YOLO weights for the video ball detector (mtah/video.py, mtah/video_ft.py).
# Official ultralytics release assets (AGPL-3.0). Stored in models/, which is gitignored.
# Usage: bash scripts/download_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
for w in yolo11s yolo11m; do
  [ -s "models/$w.pt" ] || curl -sSL --retry 3 -o "models/$w.pt" \
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/$w.pt"
  echo "models/$w.pt $(du -h "models/$w.pt" | cut -f1)"
done
