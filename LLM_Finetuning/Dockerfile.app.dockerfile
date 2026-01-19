FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app și model
COPY gradio_app_legal_llm.py .
COPY romanian_legal_model/ ./romanian_legal_model/

# Expose Gradio port
EXPOSE 7860

# Run Gradio app
CMD ["python", "gradio_app_legal_llm.py", "--model_path", "./romanian_legal_model"]