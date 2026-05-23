"""
POST /search — accept an uploaded garment image, run the query pipeline,
and return the top-k most similar catalog items.

Request (multipart/form-data):
  file    : image file (required)
  filters : JSON string matching FilterForm schema (optional)

Response: SearchResponse JSON
"""
import io
import logging
import time
import traceback

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.api.schemas import FilterForm, SearchResponse, SearchResultItem
from src.pipeline.query_pipeline import FilterParams, run_query

logger = logging.getLogger(__name__)

router = APIRouter()


def _filter_form_to_params(form: FilterForm) -> FilterParams:
    """Convert FilterForm (API schema) → FilterParams (pipeline dataclass)."""
    return FilterParams(
        category_name=form.category_name,
        season=form.season,
        min_formality=form.min_formality,
        max_formality=form.max_formality,
        viewpoint=form.viewpoint,
        top_k=form.top_k,
    )


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def search(
    file: UploadFile = File(..., description="Garment image to search with"),
    filters: str = Form(
        default="{}",
        description="JSON string of FilterForm fields (optional)",
    ),
) -> SearchResponse:
    """
    Upload a clothing photo and retrieve the most visually similar catalog items.

    - Background is automatically removed with rembg before feature extraction.
    - Filters can narrow results by category, season, formality, or viewpoint.
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Decode uploaded image
    # ------------------------------------------------------------------
    try:
        raw_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Uploaded file is not a valid image: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # 2. Parse filter JSON string → FilterParams
    # ------------------------------------------------------------------
    try:
        if not filters or filters.strip() in ("", "{}"):
            filter_form = FilterForm()
        else:
            filter_form = FilterForm.model_validate_json(filters)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid filters JSON: {exc}",
        ) from exc

    filter_params = _filter_form_to_params(filter_form)

    # ------------------------------------------------------------------
    # 3. Run query pipeline
    # ------------------------------------------------------------------
    try:
        search_results = run_query(pil_image, filter_params)
    except ConnectionError as exc:
        raise HTTPException(
            status_code=503,
            detail="Vector database unavailable",
        ) from exc
    except Exception as exc:
        logger.error("Search pipeline error:\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # 4. Build response
    # ------------------------------------------------------------------
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    result_items = [
        SearchResultItem(
            item_id=r.item_id,
            score=r.score,
            category_name=r.category_name,
            formality=r.formality,
            season=r.season,
            color_lab=r.color_lab,
            image_path=r.image_path,
        )
        for r in search_results
    ]

    return SearchResponse(
        results=result_items,
        total=len(result_items),
        query_time_ms=round(elapsed_ms, 2),
    )
