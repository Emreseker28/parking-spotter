FROM python:3.11-slim

# ffmpeg binary is the capture method (HEVC snapshot grabs). libgl/libglib are
# needed by opencv-python-headless at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + non-secret config. NOTE: .env and models/*.onnx are gitignored and
# .dockerignore'd — they are provided at runtime (env_file + volume in compose),
# never baked into the image.
COPY parkingbot/ ./parkingbot/
COPY config.yaml ./config.yaml

# Cap CPU threads on the modest i3 (also set in config; this is a backstop).
ENV OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "parkingbot"]
