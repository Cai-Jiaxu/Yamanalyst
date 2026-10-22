#!/usr/bin/env bash
# Package what the Colab notebook needs into colab_upload/MTAH/ (gitignored):
#   mtah_code.zip  (games.py, the video modules, models/yolo11s.pt) + ball_detector.ipynb
# Upload both into MyDrive/MTAH/ next to data/tactical/. Rebuild after any change to those files.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -s models/yolo11s.pt ] || bash scripts/download_models.sh
rm -rf colab_upload && mkdir -p colab_upload/MTAH
zip -q -r colab_upload/MTAH/mtah_code.zip games.py mtah/__init__.py mtah/data.py mtah/video.py \
  mtah/video_ft.py models/yolo11s.pt
cp colab/ball_detector.ipynb colab_upload/MTAH/
ls -la colab_upload/MTAH
