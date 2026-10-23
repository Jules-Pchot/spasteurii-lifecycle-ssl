#!/bin/zsh
# Start Label Studio (installed in its own venv, separate from the napari one) with access to the tiles
# prepared by Make_Labeling_Tasks.py. Open http://localhost:8080 once it is running; stop with Ctrl+C.
# Run with: zsh start_label_studio.sh

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="/Volumes/Extreme SSD/YOLO_Labeling"
exec /Users/juwul/.venvs/label-studio/bin/label-studio start --port 8080
