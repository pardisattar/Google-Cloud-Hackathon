# =============================================================================
# Fashion Finder — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder) — install uv + resolve/install all Python deps
#   Stage 2 (runtime) — lean image with only what's needed to run
#
# The image serves BOTH the FastAPI backend (port 8000) AND the static
# frontend (via FastAPI's StaticFiles mount). No separate web server needed.
#
# Build:
#   docker build -t fashion-finder .
#
# Run locally (with your .env file):
#   docker run --env-file .env -p 8000:8000 \
#     -v $(pwd)/data:/app/data \
#     fashion-finder
# =============================================================================

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifests first (better layer caching)
COPY pyproject.toml ./

# Install all production dependencies into /app/.venv
# --no-dev     → skip pytest / ipykernel / httpx
# --compile    → pre-compile .pyc files (faster cold start)
RUN uv venv .venv && \
    uv pip install \
        --python .venv/bin/python \
        --no-cache \
        "torch>=2.3.0" \
        "torchvision>=0.18.0" \
        "transformers>=4.40.0" \
        "rembg[cpu]>=2.0.57" \
        "Pillow>=10.0.0" \
        "scikit-learn>=1.4.0" \
        "scikit-image>=0.22.0" \
        "scipy>=1.13.0" \
        "numpy>=1.26.0" \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.29.0" \
        "python-multipart>=0.0.9" \
        "pydantic-settings>=2.2.0" \
        "pinecone>=3.0.0" \
        "pandas>=2.2.0" \
        "tqdm>=4.66.0" \
        "python-dotenv>=1.0.0"


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Install only the OS-level libs that rembg / PyTorch CPU need
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application source
COPY config/    ./config/
COPY src/        ./src/
COPY frontend/   ./frontend/
COPY scripts/    ./scripts/

# rembg downloads U2-Net weights on first use (~170 MB) into this dir.
# Pre-setting it to a writable location inside the container avoids
# permission issues when the container runs as a non-root user.
ENV U2NET_HOME=/app/.u2net

# Create writable dirs for model caches
RUN mkdir -p /app/.u2net /app/data/processed /app/data/raw && \
    chmod -R 777 /app/.u2net

# HuggingFace transformers cache (FashionCLIP weights)
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache

# Expose the API port
EXPOSE 8000

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start uvicorn — workers=1 keeps a single model copy in memory
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]
