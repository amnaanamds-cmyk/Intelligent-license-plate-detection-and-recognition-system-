#!/usr/bin/env bash
# One-click start for Linux / macOS: sets up a virtual environment on first
# run, installs the dependencies, then starts the service and web app.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "First run: creating virtual environment and installing packages (takes a few minutes)..."
  $PY -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi
if [ ! -f weights/plate_yolo11.pt ]; then
  echo
  echo "NOTE: weights/plate_yolo11.pt not found. The web app will start, but recognition"
  echo "      needs a trained model: python scripts/train.py --data configs/data.yaml"
  echo
fi
exec .venv/bin/python scripts/serve.py --config configs/system.yaml "$@"
