# Dockerfile for gemma-tpu training on Cloud TPU
#
# Build:
#   docker build -t gemma-tpu .
#
# Run on TPU VM:
#   docker run --privileged --network host \
#     -v /dev/shm:/dev/shm \
#     -e HF_TOKEN=$HF_TOKEN \
#     gemma-tpu python training/train.py --config configs/default.yaml

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY training/requirements.txt /app/training/requirements.txt

# Install JAX for TPU
RUN pip install --no-cache-dir \
    'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install other dependencies
RUN pip install --no-cache-dir -r training/requirements.txt

# Install Tunix, Qwix, Flax from source
RUN pip install --no-cache-dir \
    git+https://github.com/google/tunix \
    git+https://github.com/google/qwix && \
    pip uninstall -y flax && \
    pip install --no-cache-dir git+https://github.com/google/flax

# Copy application code
COPY . /app

# Set environment variables for TPU
ENV PJRT_DEVICE=TPU
ENV PYTHONUNBUFFERED=1
ENV TMPDIR=/dev/shm

# Default command
CMD ["python", "training/train.py", "--config", "configs/default.yaml"]
