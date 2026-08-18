# ─── Clinical Polyp Segmentation Web App Dockerfile ─────────────────────────
# Optimized for Google Cloud Run (Single Container, Port 8080)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8080 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Install minimal OS dependencies for OpenCV, libgomp, and image decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install dependencies with CPU-optimized PyTorch wheels for fast, lightweight Cloud Run builds
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and app
COPY src/ ./src/
COPY app/ ./app/

# Copy model checkpoint and evaluation outputs
COPY outputs/checkpoints/baseline_kvasir_unet_resnet34_best.pth ./outputs/checkpoints/
COPY outputs/eval_in_distribution/metrics_report.json ./outputs/eval_in_distribution/
COPY outputs/eval_in_distribution/per_sample_metrics.csv ./outputs/eval_in_distribution/
COPY outputs/eval_ood_cvc_clinicdb/metrics_report.json ./outputs/eval_ood_cvc_clinicdb/
COPY outputs/eval_ood_cvc_clinicdb/per_sample_metrics.csv ./outputs/eval_ood_cvc_clinicdb/

# Copy curated demo datasets
COPY Data/ ./Data/

# Create non-root user for Cloud Run security hardening
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/_stcore/health || exit 1

# Launch Streamlit server
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
