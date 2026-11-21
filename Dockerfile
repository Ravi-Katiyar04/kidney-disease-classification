# Base image with Python 3.11 (slim for smaller image footprint)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=8080

# Set working directory
WORKDIR /app

# Install minimal system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition and source tree first to leverage Docker layer caching
COPY requirements.txt setup.py README.md ./
COPY src ./src

# Upgrade pip and install all Python dependencies (including cnnClassifier package)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-fetch and cache model from Hugging Face during image build (fails gracefully if offline)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='katiyar2002/kidney-disease-classification', filename='model.h5')" || true

# Copy all remaining application code
COPY . .

# Expose web application port
EXPOSE 8080

# Health check to monitor application responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Launch Flask application
CMD ["python", "app.py"]

