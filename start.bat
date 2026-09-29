@echo off
REM One-click start for Windows: sets up a virtual environment on first run,
REM installs the dependencies, then starts the service and web app.
cd /d "%~dp0"
if not exist .venv (
  echo First run: creating virtual environment and installing packages ^(takes a few minutes^)...
  python -m venv .venv || (echo Python 3.10 or 3.11 is required: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt || (echo Installation failed & pause & exit /b 1)
)
if not exist weights\plate_yolo11.pt (
  echo.
  echo NOTE: weights\plate_yolo11.pt not found. The web app will start, but recognition
  echo       needs a trained model: python scripts\train.py --data configs\data.yaml
  echo.
)
.venv\Scripts\python scripts\serve.py --config configs\system.yaml %*
pause
