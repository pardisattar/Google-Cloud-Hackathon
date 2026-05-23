"""
Pinecone vector store client.

Provides:
  init_index()     — creates the index if it does not already exist
  upsert_batch()   — upserts a list of vector dicts to Pinecone
  query()          — queries the index and returns top-k matches
"""
import logging
import time
from typing import Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec

from config.settings import settings

logger = logging.getLogger(__name__)

_INDEX_DIM = 583
_INDEX_METRIC = "cosine"
_CLOUD = "aws"
_REGION = "us-east-1"
_UPSERT_CHUNK = 100  # Pinecone free tier limit per upsert call

# ---------------------------------------------------------------------------
# Module-level Pinecone client (lazy singleton)
# ---------------------------------------------------------------------------
_pc: Optional[Pinecone] = None
_index = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    return _pc


def _get_index():
    global _index
    if _index is None:
        _index = _get_client().Index(settings.PINECONE_INDEX_NAME)
    return _index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_index() -> None:
    """
    Create the Pinecone index if it does not already exist.
    Blocks until the index is ready (polls describe_index).
    """
    pc = _get_client()
    existing = [idx.name for idx in pc.list_indexes()]

    if settings.PINECONE_INDEX_NAME in existing:
        logger.info("Index '%s' already exists — skipping creation.",
                    settings.PINECONE_INDEX_NAME)
        return

    logger.info(
        "Creating index '%s' (dim=%d, metric=%s) …",
        settings.PINECONE_INDEX_NAME, _INDEX_DIM, _INDEX_METRIC,
    )
    pc.create_index(
        name=settings.PINECONE_INDEX_NAME,
        dimension=_INDEX_DIM,
        metric=_INDEX_METRIC,
        spec=ServerlessSpec(cloud=_CLOUD, region=_REGION),
    )

    # Poll until ready
    for _ in range(60):  # up to 60 s
        desc = pc.describe_index(settings.PINECONE_INDEX_NAME)
        if desc.status.get("ready", False):
            break
        time.sleep(1)
    logger.info("Index '%s' is ready.", settings.PINECONE_INDEX_NAME)


def upsert_batch(vectors: List[Dict]) -> None:
    """
    Upsert a list of vector dicts to Pinecone.

    Each dict must have:
      {
        "id":       str,
        "values":   list[float],   # must be plain Python list, not np.ndarray
        "metadata": dict,
      }

    Splits into chunks of 100 to respect the Pinecone free-tier limit.
    """
    index = _get_index()
    for i in range(0, len(vectors), _UPSERT_CHUNK):
        chunk = vectors[i : i + _UPSERT_CHUNK]
        index.upsert(vectors=chunk)
    logger.debug("Upserted %d vectors.", len(vectors))


def query(
    vector: List[float],
    top_k: int = 10,
    filter: Optional[Dict] = None,
) -> List[Dict]:
    """
    Query the index and return top-k matches.

    Parameters
    ----------
    vector  : query vector as plain Python list of floats
    top_k   : number of results to return
    filter  : optional Pinecone metadata filter dict

    Returns
    -------
    List of match dicts with keys: id, score, metadata.
    """
    index = _get_index()
    kwargs: Dict = {"vector": vector, "top_k": top_k, "include_metadata": True}
    if filter:
        kwargs["filter"] = filter
    response = index.query(**kwargs)
    return [
        {"id": m.id, "score": m.score, "metadata": m.metadata}
        for m in response.matches
    ]
