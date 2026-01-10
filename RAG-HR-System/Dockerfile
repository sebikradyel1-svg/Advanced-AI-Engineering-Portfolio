# ==============================================================================
# DOCKERFILE - HR Internal Knowledge RAG System
# Project: RAG-based HR Policy Question Answering
# Author: Sebastian Kradyel
# ==============================================================================
# INSTRUCTIONS:
# 1. Rename this file to "Dockerfile" (no extension)
# 2. Place in your project root folder
# 3. Build: docker build -t hr-rag:latest .
# 4. Run: docker run -it hr-rag:latest
# ==============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: BUILDER - Install and compile dependencies
# -----------------------------------------------------------------------------
# Multi-stage build: First stage compiles everything, second stage runs it
# Result: Much smaller final image (no compilers, cache, etc.)

FROM python:3.10-slim AS builder

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed for compilation
# build-essential, gcc = C compilers for native Python extensions
# --no-install-recommends = Skip optional packages (smaller image)
# rm -rf /var/lib/apt/lists/* = Clean apt cache (saves ~100MB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caching optimization)
# If requirements.txt unchanged, Docker skips reinstalling packages
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 2: RUNTIME - Minimal image for running the application
# -----------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Image metadata
LABEL maintainer="Sebastian Kradyel"
LABEL description="HR Internal Knowledge RAG System - LangChain + FAISS + Transformers"
LABEL version="1.0"

# Environment variables
# PYTHONDONTWRITEBYTECODE=1 = Don't create .pyc cache files
# PYTHONUNBUFFERED=1 = Print output immediately (better for logs)
# HF_HOME = Hugging Face model cache location
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Runtime-only system dependencies
# libgomp1 = OpenMP library (required by FAISS for parallel search)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/vector_store /app/.cache

# Security: Create and switch to non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Document the port (doesn't actually expose it)
EXPOSE 8000

# Health check - verify critical imports work
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; import langchain; print('healthy')" || exit 1

# Default command to run
CMD ["python", "hr_rag_system_updated.py"]
