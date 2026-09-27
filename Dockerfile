# CPU image for the LPR service. For NVIDIA GPUs, start from an
# nvidia/cuda runtime image and install paddlepaddle-gpu and CUDA PyTorch instead.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # PaddleOCR / Ultralytics model caches, kept in a volume
    PADDLE_PDX_CACHE_HOME=/models/paddlex \
    YOLO_CONFIG_DIR=/models/ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# The desktop GUI (PyQt5) is not needed in the server image.
RUN grep -v -i "pyqt5" requirements.txt > req-server.txt \
    && pip install --upgrade pip \
    && pip install -r req-server.txt \
    && pip uninstall -y opencv-python && pip install opencv-python-headless

COPY lpr/ lpr/
COPY service/ service/
COPY scripts/ scripts/
COPY configs/ configs/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["python", "scripts/serve.py", "--config", "configs/system.yaml"]
