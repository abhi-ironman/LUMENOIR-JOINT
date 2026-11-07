# LUMENOIR inference backend for hosted CPU deployment.
# The React frontend is deployed separately from /frontend.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install the API/runtime stack first from PyPI.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Railway runs this service on CPU. Use the official CPU-only PyTorch wheel
# index so the image does not pull CUDA/cuDNN packages it cannot use.
RUN pip install --no-cache-dir torch==2.14.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY backend ./backend

# Keep model weights out of Git. Fetch the published Zero-DCE++ checkpoint
# during the image build and verify its known SHA-256 before enabling inference.
ARG ZERO_DCE_CHECKPOINT_URL=https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth
ARG ZERO_DCE_CHECKPOINT_SHA256=ca8855b90df9a80fa4195a831f33d3476b1964f787eb70602797c773067f3b84

RUN mkdir -p /app/backend/weights && \
    python -c 'import hashlib, pathlib, sys, urllib.request; p=pathlib.Path("/app/backend/weights/zero_dce_plus_plus.pth"); urllib.request.urlretrieve(sys.argv[1], p); actual=hashlib.sha256(p.read_bytes()).hexdigest(); expected=sys.argv[2]; print("Zero-DCE++ checkpoint SHA-256:", actual); assert actual == expected, f"Checkpoint SHA-256 mismatch: {actual}"' \
    "$ZERO_DCE_CHECKPOINT_URL" "$ZERO_DCE_CHECKPOINT_SHA256"

# This public demo starts with the published pretrained model on CPU.
# Replace the checkpoint only with a separately verified, licensed artifact.
ENV LUMENOIR_MODEL_ID=zero_dce_plus_plus \
    LUMENOIR_MODEL_WEIGHTS=/app/backend/weights/zero_dce_plus_plus.pth \
    LUMENOIR_DEVICE=cpu \
    LUMENOIR_INFERENCE_DEVICE=cpu \
    LUMENOIR_PRECISION=float32 \
    LUMENOIR_INFERENCE_PRECISION=float32 \
    LUMENOIR_LOAD_MODEL_ON_STARTUP=true \
    LUMENOIR_ZERODCE_SCALE_FACTOR=12 \
    LUMENOIR_CLOSED_LOOP_ENABLED=false

EXPOSE 8000

# Railway supplies PORT; local Docker runs fall back to 8000.
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
