FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by faiss-cpu and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source (models mounted at runtime, not baked in)
COPY src/ src/
COPY pyproject.toml .

RUN pip install --no-cache-dir -e .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "triage_iq.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
