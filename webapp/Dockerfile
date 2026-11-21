# ── Stage 1: Build ──────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY webapp/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ────────────────────────────────────
FROM python:3.12-slim

# Create non-root user for security
RUN groupadd -r cropapp && useradd -r -g cropapp -m cropapp

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy model files
COPY models/ /app/models/
COPY preprocessing_pipeline.joblib /app/

# Copy application code
COPY webapp/backend/ /app/backend/
COPY webapp/frontend/ /app/frontend/
COPY webapp/.env /app/.env

# Update .env paths for Docker layout
RUN sed -i 's|MODEL_PATH=.*|MODEL_PATH=/app/models/random_forest.joblib|' /app/.env \
 && sed -i 's|PIPELINE_PATH=.*|PIPELINE_PATH=/app/preprocessing_pipeline.joblib|' /app/.env

# Set ownership
RUN chown -R cropapp:cropapp /app

# Switch to non-root user
USER cropapp

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
