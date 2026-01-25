# HR RAG Knowledge Assistant - Docker Configuration
# Optimized for Groq API deployment (2GB RAM)

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/.cache \
    HF_HOME=/app/.cache

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ONLY embeddings model (not LLM - using Groq API!)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && \
    echo "✅ Embeddings model cached"

# Create cache and logs directories with proper permissions
RUN mkdir -p /app/.cache /app/logs && chmod 777 /app/.cache /app/logs

# Copy application code and faiss_index
COPY app.py .
COPY faiss_index/ ./faiss_index/

# Expose port (Render assigns PORT dynamically)
EXPOSE 7860

# Health check - check if Gradio is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/ > /dev/null 2>&1 || exit 1

# Run the application
CMD ["python", "app.py"]