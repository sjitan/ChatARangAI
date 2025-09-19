#!/usr/bin/env bash
set -e
: "${GDRIVE_DIR:?set GDRIVE_DIR to your Drive folder path}"
mkdir -p "$GDRIVE_DIR/plots" "$GDRIVE_DIR/reports"
rsync -aq --delete logs/plots/ "$GDRIVE_DIR/plots/"
rsync -aq --delete logs/reports/ "$GDRIVE_DIR/reports/"
