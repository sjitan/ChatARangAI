#!/usr/bin/env bash
set -e
source .venv/bin/activate
IDX=$(python - <<'PY'
import cv2
for i in (1,0,2,3):
    cap=cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    if cap.isOpened():
        print(i); cap.release(); break
PY
)
[ -z "$IDX" ] && echo "NO_CAMERA" && exit 1
./presets.sh "${SUBJECT:-DEMO}" "${CONDITION:-interrupt}" "$IDX"
