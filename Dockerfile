# Multi-stage build to optimize final image size
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ================================================================
# Final stage - Production API Server
# ================================================================
FROM python:3.12-slim as api

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH and environment
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Create cache and data directories
RUN mkdir -p /app/.cache/huggingface /app/data/06_models

# Copy project files
COPY pyproject.toml .
COPY src/ ./src
COPY conf/ ./conf

# Pre-download zero-shot model (fallback)
RUN python -c "from transformers import pipeline; \
    print('Pre-loading zero-shot model...'); \
    pipeline('zero-shot-classification', model='MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'); \
    print('✓ Zero-shot model cached')" || echo "⚠️  Could not pre-cache zero-shot model"

# Health check for API
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health')" || exit 1

# Expose API port
EXPOSE 8001

# Default: Run FastAPI
CMD ["python", "-m", "uvicorn", "email_classification.api:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

# ================================================================
# Kedro Training Stage - For model training via Kedro
# ================================================================
FROM python:3.12-slim as training

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH and environment
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface

# Create directories for data
RUN mkdir -p /app/data/{01_raw,02_intermediate,03_primary,05_model_input,06_models,07_model_output,08_reporting}

# Copy project files
COPY . .

# Run Kedro training pipeline
ENTRYPOINT ["kedro", "run", "--pipeline", "training"]