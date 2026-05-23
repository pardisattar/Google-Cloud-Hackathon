"""
Pydantic request / response schemas for the Fashion Finder API.

Schemas:
  FilterForm       — optional search filters sent alongside the uploaded image
  SearchResultItem — a single matched catalog item
  SearchResponse   — full POST /search response envelope
  HealthResponse   — GET /health response
"""
from pydantic import BaseModel, Field


class FilterForm(BaseModel):
    """
    Optional filter parameters sent alongside the uploaded image.
    All fields have defaults so the form can be submitted partially or empty.
    """

    category_name: str | None = Field(
        default=None,
        description="One of the 13 DeepFashion2 category names",
    )
    season: str | None = Field(
        default=None,
        description="spring | summer | autumn | winter",
    )
    min_formality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum formality score (0=very casual, 1=very formal)",
    )
    max_formality: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum formality score",
    )
    viewpoint: int | None = Field(
        default=None,
        description="1=frontal, 2=side, 3=back",
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return",
    )


class SearchResultItem(BaseModel):
    item_id: str
    score: float
    category_name: str
    formality: float
    season: str
    color_lab: list[float]  # [L, a, b]
    image_path: str


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    total: int
    query_time_ms: float


class HealthResponse(BaseModel):
    status: str              # "ok"
    pinecone_connected: bool
    index_vector_count: int
    models_loaded: bool
