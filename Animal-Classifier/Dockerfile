FROM python:3.11-slim

WORKDIR /app

# Install system dependencies pentru TensorFlow
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY universal_classifier.py .
COPY data/ ./data/

# Expose port pentru Streamlit
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "universal_classifier.py", "--server.port=8501", "--server.address=0.0.0.0"]