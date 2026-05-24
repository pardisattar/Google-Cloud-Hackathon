"""
Fashion Finder API — FastAPI application entry point.

Start with:
  uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

The lifespan context manager warms up both ML models and Pinecone on startup
so the first real user request does not pay the cold-start cost.
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# ---------------------------------------------------------------------------
# DLL path fix for onnxruntime-gpu
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    venv_base = os.path.join(os.getcwd(), ".venv", "Lib", "site-packages", "nvidia")
    cuda_paths = [
        os.path.join(venv_base, "cuda_runtime", "bin"),
        os.path.join(venv_base, "cublas", "bin"),
        os.path.join(venv_base, "cudnn", "bin"),
        os.path.join(venv_base, "cuda_nvrtc", "bin"),
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            os.add_dll_directory(path)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-up models and Pinecone connection on startup."""

    # --- Pinecone ---
    try:
        from src.vector_store.pinecone_client import init_index
        init_index()
        print("✓ Pinecone connected")
    except Exception:
        logger.exception("Failed to connect to Pinecone on startup.")

    # --- FashionCLIP (embed_image triggers module-level model load) ---
    try:
        from src.models.embedder import embed_image
        dummy = Image.new("RGB", (1, 1), color=(255, 255, 255))
        embed_image(dummy)
        print("✓ Models loaded (FashionCLIP)")
    except Exception:
        logger.exception("Failed to load FashionCLIP on startup.")

    # --- rembg (segment_garment triggers U2-Net model download/load) ---
    try:
        from src.models.segmentor import segment_garment
        dummy = Image.new("RGB", (1, 1), color=(255, 255, 255))
        segment_garment(dummy)
        print("✓ Models loaded (rembg / U2-Net)")
    except Exception:
        logger.exception("Failed to load rembg on startup.")

    yield  # application runs here

    # Shutdown — nothing to clean up
    logger.info("Fashion Finder API shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fashion Finder API",
    description="Clothing similarity search using FashionCLIP + Pinecone",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — open for hackathon demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from src.api.routes.health import router as health_router  # noqa: E402
from src.api.routes.images import router as images_router  # noqa: E402
from src.api.routes.search import router as search_router  # noqa: E402

app.include_router(search_router, prefix="")
app.include_router(health_router, prefix="")
app.include_router(images_router, prefix="")
