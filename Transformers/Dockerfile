FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY transformer_stock_prediction.py .

# Create output directory
RUN mkdir -p /app/output

# Run training
CMD ["python", "transformer_stock_prediction.py"]