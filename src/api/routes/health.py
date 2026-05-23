"""
GET /health — liveness and readiness check for the Fashion Finder API.

Returns:
  - Pinecone connectivity status and vector count
  - Whether ML models (FashionCLIP) are loaded in memory
"""
import logging

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.vector_store.pinecone_client import init_index, _get_index

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """
    Check service health.

    Attempts a live Pinecone connection and inspects model loading state.
    Always returns HTTP 200 — callers must inspect the response body.
    """
    # ------------------------------------------------------------------
    # 1. Pinecone connectivity
    # ------------------------------------------------------------------
    pinecone_connected = False
    index_vector_count = 0
    try:
        init_index()
        index = _get_index()
        stats = index.describe_index_stats()
        pinecone_connected = True
        # Pinecone v3 SDK returns an object; fall back to dict-style access
        # if a future SDK version changes the shape.
        if hasattr(stats, "total_vector_count"):
            index_vector_count = int(stats.total_vector_count or 0)
        else:
            index_vector_count = int(stats.get("total_vector_count", 0))
    except Exception:
        logger.exception("Health check: Pinecone connection failed.")

    # ------------------------------------------------------------------
    # 2. Models loaded check
    # ------------------------------------------------------------------
    models_loaded = False
    try:
        from src.models import embedder  # noqa: F401 — triggers module import
        # The module-level _model is set at import time; if it survived
        # import without raising, the model is loaded.
        models_loaded = embedder._model is not None
    except Exception:
        logger.exception("Health check: model load check failed.")

    return HealthResponse(
        status="ok",
        pinecone_connected=pinecone_connected,
        index_vector_count=index_vector_count,
        models_loaded=models_loaded,
    )
