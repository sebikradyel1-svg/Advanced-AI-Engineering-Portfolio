# ==============================================================================
# DOCKERFILE - KL Guard RLHF System
# Project: Reinforcement Learning from Human Feedback with KL Divergence Control
# Author: Sebastian Kradyel
# ==============================================================================
# INSTRUCTIONS:
# 1. Rename this file to "Dockerfile" (no extension)
# 2. Place in your project root folder
# 3. Build: docker build -t kl-rlhf:latest .
# 4. Run: docker run -it kl-rlhf:latest
# 5. With GPU: docker run --gpus all -it kl-rlhf:latest
# ==============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: BUILDER
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# System dependencies for PyTorch compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages (takes 5-10 min for PyTorch + Transformers)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 2: RUNTIME
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

LABEL maintainer="Sebastian Kradyel"
LABEL description="RLHF with PPO and KL Divergence Control"
LABEL version="1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY . .

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers /app/outputs

# Security: Non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD python -c "import torch; from transformers import AutoTokenizer; print('ok')" || exit 1

# Run RLHF training script
CMD ["python", "KL_Guard_RLHF.py"]
