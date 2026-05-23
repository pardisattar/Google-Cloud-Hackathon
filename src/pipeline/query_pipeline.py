"""
Query pipeline — orchestrates the full user-image-to-search-results flow.

Steps for every user query:
  1. Segment garment (rembg background removal)
  2. Extract features: CLIP, Gabor texture, KMeans color, shape (zeros)
  3. Resolve category / formality / season from optional filter params
  4. Build the 583-dimensional combined vector
  5. Construct Pinecone metadata filter
  6. Query Pinecone
  7. Parse and return SearchResult list
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from src.features.feature_builder import (
    CATEGORY_DECOMPOSITION,
    FORMALITY_SCORE,
    build_combined_vector,
    encode_season_cyclic,
    infer_season,
)
from src.models.color_extractor import extract_dominant_color_lab
from src.models.embedder import embed_image
from src.models.segmentor import segment_and_flatten
from src.models.texture_extractor import extract_texture
from src.vector_store.pinecone_client import query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FilterParams:
    """
    Optional filters provided by the user via the search form.
    All fields optional — None means no filter is applied for that field.
    """
    category_name: Optional[str] = None   # e.g. "long sleeve top"
    season: Optional[str] = None          # "spring" | "summer" | "autumn" | "winter"
    min_formality: Optional[float] = None  # 0.0 to 1.0
    max_formality: Optional[float] = None  # 0.0 to 1.0
    viewpoint: Optional[int] = None        # 1=frontal, 2=side, 3=back
    top_k: int = 10


@dataclass
class SearchResult:
    item_id: str
    score: float
    category_name: str
    formality: float
    season: str
    color_lab: list   # [L, a, b]
    image_path: str
    metadata: dict    # full raw metadata from Pinecone


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _build_pinecone_filter(filters: FilterParams) -> Optional[dict]:
    """
    Translate FilterParams into a Pinecone metadata filter dict.

    Single conditions are returned directly.
    Multiple conditions are wrapped in {"$and": [...]}.
    Returns None if no filter fields are set.
    """
    conditions = []

    if filters.category_name is not None:
        conditions.append({"category_name": {"$eq": filters.category_name}})

    if filters.viewpoint is not None:
        conditions.append({"viewpoint": {"$eq": filters.viewpoint}})

    if filters.min_formality is not None:
        conditions.append({"formality": {"$gte": filters.min_formality}})

    if filters.max_formality is not None:
        conditions.append({"formality": {"$lte": filters.max_formality}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ---------------------------------------------------------------------------
# Main query function
# ---------------------------------------------------------------------------

def run_query(
    pil_image: Image.Image,
    filters: FilterParams,
) -> list[SearchResult]:
    """
    Run the full query pipeline from a raw PIL image to ranked search results.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Raw image uploaded by the user (any size / mode).
    filters : FilterParams
        Optional filter parameters from the search form.

    Returns
    -------
    list[SearchResult]
        Ranked list of matching catalog items, length = filters.top_k.
    """
    # ------------------------------------------------------------------
    # Step 1 — Segmentation
    # ------------------------------------------------------------------
    logger.info("Segmenting garment …")
    segmented = segment_and_flatten(pil_image)

    # ------------------------------------------------------------------
    # Step 2 — Feature extraction
    # ------------------------------------------------------------------
    logger.info("Extracting features …")
    clip_vec: np.ndarray = embed_image(segmented)          # (512,) L2-normed
    texture_vec: np.ndarray = extract_texture(segmented)   # (36,)
    color_lab: np.ndarray = extract_dominant_color_lab(segmented)  # (3,)
    shape_vec: np.ndarray = np.zeros(20, dtype=np.float32)         # no landmarks

    # ------------------------------------------------------------------
    # Step 3 — Category / formality / season from filters
    # ------------------------------------------------------------------
    if filters.category_name is not None:
        cat_name = filters.category_name
        cat_vec = np.array(
            CATEGORY_DECOMPOSITION.get(cat_name, [0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        formality = float(FORMALITY_SCORE.get(cat_name, 0.5))
        season = infer_season(cat_name)
    else:
        cat_vec = np.zeros(5, dtype=np.float32)
        formality = 0.5   # neutral default
        season = "summer"  # neutral default

    # User-specified season overrides the category-inferred one
    if filters.season is not None:
        season = filters.season

    season_vec = encode_season_cyclic(season)                         # (2,)
    proportions = np.array([0.5, 0.5, 1.0, 0.25], dtype=np.float32)  # neutral

    # ------------------------------------------------------------------
    # Step 4 — Build combined 583d vector
    # ------------------------------------------------------------------
    logger.info("Building combined vector …")
    final_vec = build_combined_vector(
        clip_vec, texture_vec, shape_vec, cat_vec,
        color_lab, formality, season_vec, proportions,
    )

    # ------------------------------------------------------------------
    # Step 5 — Build Pinecone metadata filter
    # ------------------------------------------------------------------
    pinecone_filter = _build_pinecone_filter(filters)

    # ------------------------------------------------------------------
    # Step 6 — Query Pinecone
    # ------------------------------------------------------------------
    logger.info("Querying Pinecone (top_k=%d) …", filters.top_k)
    raw_results = query(
        vector=final_vec.tolist(),
        top_k=filters.top_k,
        filter=pinecone_filter,
    )

    # ------------------------------------------------------------------
    # Step 7 — Build SearchResult list
    # ------------------------------------------------------------------
    results: list[SearchResult] = []
    for match in raw_results:
        meta = match.get("metadata", {})
        result = SearchResult(
            item_id=match.get("id", ""),
            score=float(match.get("score", 0.0)),
            category_name=meta.get("category_name", ""),
            formality=float(meta.get("formality", 0.0)),
            season=meta.get("season", ""),
            color_lab=[
                float(meta.get("color_l", 0.0)),
                float(meta.get("color_a", 0.0)),
                float(meta.get("color_b", 0.0)),
            ],
            image_path=meta.get("image_path", ""),
            metadata=meta,
        )
        results.append(result)

    logger.info("Query complete — %d results returned.", len(results))
    return results
