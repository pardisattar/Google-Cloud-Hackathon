"""
Ingestion pipeline orchestrator.

Two-pass design
---------------
PASS 1  Collect all landmark lists from the subset → fit (or load) PCA.
PASS 2  For each annotation file:
          • load image as PIL RGB
          • parse all items (item1, item2 if present)
          • skip items with viewpoint == 0 (not worn)
          • crop image to bounding box
          • run all feature extractors
          • build combined 583-d vector
          • buffer until BATCH_SIZE → upsert to Pinecone

Item ID format: "{split}_{base_filename}_{item_key}"  e.g. "train_000001_item1"
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from config.settings import settings
from src.features.feature_builder import (
    CATEGORY_DECOMPOSITION,
    FORMALITY_SCORE,
    build_combined_vector,
    build_metadata,
    encode_season_cyclic,
    extract_proportions,
    infer_season,
)
from src.models.color_extractor import extract_dominant_color_lab
from src.models.embedder import embed_batch
from src.models.shape_extractor import extract_shape, fit_pca, load_pca
from src.models.texture_extractor import extract_texture
from src.vector_store.pinecone_client import init_index, upsert_batch

logger = logging.getLogger(__name__)

_DEFAULT_CAT_VEC = np.zeros(5, dtype=np.float32)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotation(json_path: Path) -> List[Dict]:
    """
    Parse a DeepFashion2 annotation file.

    Returns a list of item dicts, one per item (item1 and/or item2).
    Each dict has the item fields plus an injected "item_key" and "source".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source = data.get("source", "shop")
    pair_id = data.get("pair_id", 0)

    items = []
    for key in ("item1", "item2"):
        if key not in data:
            continue
        item = data[key]
        item = dict(item)          # shallow copy so we can add keys safely
        item["item_key"] = key
        item["source"] = source
        item["pair_id"] = pair_id
        items.append(item)

    return items


# ---------------------------------------------------------------------------
# Single-item processing
# ---------------------------------------------------------------------------

def _clamp_bbox(bbox: List[int], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Clamp bounding box coordinates to image dimensions."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))
    return x1, y1, x2, y2


def process_item(
    item: Dict,
    pil_image: Image.Image,
    pca_model: PCA,
    split: str,
    base_filename: str,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Extract all features for a single item and return (vector, metadata).
    Returns None if the item should be skipped or fails.
    """
    viewpoint = item.get("viewpoint", 1)
    if viewpoint == 0:
        return None   # not worn — skip

    category_name = item.get("category_name", "").lower().strip()
    category_id = item.get("category_id", 0)
    scale = item.get("scale", 1)
    occlusion = item.get("occlusion", 1)
    landmarks_raw = item.get("landmarks", [])
    bbox = item.get("bounding_box", [0, 0, *pil_image.size])
    item_key = item.get("item_key", "item1")
    source = item.get("source", "shop")
    pair_id = item.get("pair_id", 0)

    img_w, img_h = pil_image.size
    x1, y1, x2, y2 = _clamp_bbox(bbox, img_w, img_h)
    cropped = pil_image.crop((x1, y1, x2, y2))

    # --- Feature extraction ---
    texture_vec = extract_texture(cropped)
    shape_vec = extract_shape(landmarks_raw, pca_model)
    color_lab = extract_dominant_color_lab(cropped)
    # FashionCLIP is batched in run_ingestion; store cropped image reference here
    # We return the cropped image so embed_batch can be called outside
    cat_vec = np.array(
        CATEGORY_DECOMPOSITION.get(category_name, [0.0, 0, 0.0, 0, 0]),
        dtype=np.float32,
    )
    formality = FORMALITY_SCORE.get(category_name, 0.3)
    season = infer_season(category_name)
    season_vec = encode_season_cyclic(season)
    proportions = extract_proportions(bbox, img_w, img_h)

    return {
        "cropped_image": cropped,
        "texture_vec": texture_vec,
        "shape_vec": shape_vec,
        "color_lab": color_lab,
        "cat_vec": cat_vec,
        "formality": formality,
        "season_vec": season_vec,
        "proportions": proportions,
        "metadata_kwargs": dict(
            split=split,
            base_filename=base_filename,
            item_key=item_key,
            source=source,
            category_name=category_name,
            category_id=category_id,
            pair_id=pair_id,
            formality=formality,
            season=season,
            viewpoint=viewpoint,
            color_lab=color_lab,
            scale=scale,
            occlusion=occlusion,
            image_path=str(pil_image.filename) if hasattr(pil_image, "filename") else "",
        ),
    }


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def run_ingestion(
    data_split: str = "train",
    subset: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict:
    """
    Orchestrate the full dataset ingestion pipeline.

    Parameters
    ----------
    data_split  : "train" or "validation"
    subset      : max number of annotation files to process (None = all)
    batch_size  : number of items per Pinecone upsert

    Returns
    -------
    dict with keys: total_processed, total_upserted, total_errors, elapsed_seconds
    """
    subset = subset or settings.SUBSET_SIZE
    batch_size = batch_size or settings.BATCH_SIZE

    split_dir = settings.data_path / data_split
    image_dir = split_dir / "image"
    annos_dir = split_dir / "annos"

    # Collect annotation files sorted for reproducibility
    annot_files = sorted(annos_dir.glob("*.json"))
    if subset:
        annot_files = annot_files[:subset]

    logger.info(
        "Starting ingestion: split=%s, files=%d, batch_size=%d",
        data_split, len(annot_files), batch_size,
    )

    # Ensure Pinecone index exists
    init_index()

    # -----------------------------------------------------------------------
    # PASS 1 — PCA fitting (skip if pkl already exists)
    # -----------------------------------------------------------------------
    pca_model: Optional[PCA] = None

    if settings.shape_pca_path.exists():
        logger.info("Loading existing PCA from %s", settings.shape_pca_path)
        pca_model = load_pca(settings.shape_pca_path)
    else:
        logger.info("PASS 1: collecting landmarks for PCA fitting …")
        all_landmarks: List[List[int]] = []
        for annot_path in tqdm(annot_files, desc="Pass 1 (landmarks)"):
            try:
                items = parse_annotation(annot_path)
                for item in items:
                    if item.get("viewpoint", 1) == 0:
                        continue
                    lm = item.get("landmarks", [])
                    if lm:
                        all_landmarks.append(lm)
            except Exception as exc:
                logger.warning("Pass 1 — skipping %s: %s", annot_path.name, exc)

        if len(all_landmarks) < 1000:
            logger.warning(
                "Only %d landmark arrays available (recommended ≥ 1000). "
                "PCA quality may be limited.",
                len(all_landmarks),
            )

        pca_model = fit_pca(all_landmarks, settings.shape_pca_path)

    # -----------------------------------------------------------------------
    # PASS 2 — Feature extraction + upsert
    # -----------------------------------------------------------------------
    logger.info("PASS 2: extracting features and upserting …")

    total_processed = 0
    total_upserted = 0
    total_errors = 0

    # Accumulation buffers for batched CLIP embedding
    _pending_items: List[Dict] = []   # intermediate dicts from process_item
    _pending_images: List[Image.Image] = []

    def _flush_batch() -> None:
        nonlocal total_upserted
        if not _pending_items:
            return

        # Batch CLIP embedding
        clip_vecs = embed_batch(_pending_images)  # (N, 512)

        vectors_to_upsert = []
        for i, item_data in enumerate(_pending_items):
            try:
                combined = build_combined_vector(
                    clip_vec=clip_vecs[i],
                    texture_vec=item_data["texture_vec"],
                    shape_vec=item_data["shape_vec"],
                    cat_vec=item_data["cat_vec"],
                    color_lab=item_data["color_lab"],
                    formality=item_data["formality"],
                    season_vec=item_data["season_vec"],
                    proportions=item_data["proportions"],
                )
                meta = build_metadata(**item_data["metadata_kwargs"])
                vec_id = meta["item_id"]
                vectors_to_upsert.append({
                    "id": vec_id,
                    "values": combined.tolist(),
                    "metadata": meta,
                })
            except Exception as exc:
                logger.error("build_combined_vector failed: %s", exc)

        upsert_batch(vectors_to_upsert)
        total_upserted += len(vectors_to_upsert)
        _pending_items.clear()
        _pending_images.clear()

    t0 = time.time()

    with tqdm(total=len(annot_files), desc="Ingesting") as pbar:
        for annot_path in annot_files:
            base = annot_path.stem                       # e.g. "000001"
            image_path = image_dir / f"{base}.jpg"

            if not image_path.exists():
                logger.warning("Image not found: %s — skipping.", image_path)
                pbar.update(1)
                continue

            try:
                pil_image = Image.open(image_path).convert("RGB")
                # Attach filename attribute so process_item can record it
                pil_image.filename = str(image_path)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.error("Cannot open image %s: %s", image_path, exc)
                total_errors += 1
                pbar.update(1)
                continue

            try:
                items = parse_annotation(annot_path)
            except Exception as exc:
                logger.error("Cannot parse annotation %s: %s", annot_path, exc)
                total_errors += 1
                pbar.update(1)
                continue

            n_items_this_file = 0
            for item in items:
                try:
                    result = process_item(
                        item, pil_image, pca_model, data_split, base
                    )
                    if result is None:
                        continue   # viewpoint == 0 skip

                    _pending_items.append(result)
                    _pending_images.append(result.pop("cropped_image"))
                    total_processed += 1
                    n_items_this_file += 1

                    if len(_pending_items) >= batch_size:
                        _flush_batch()

                except Exception as exc:
                    logger.error(
                        "Error processing %s / %s: %s",
                        annot_path.name, item.get("item_key", "?"), exc,
                    )
                    total_errors += 1

            pbar.update(1)

    # Flush any remaining items
    _flush_batch()

    elapsed = time.time() - t0
    summary = {
        "total_processed": total_processed,
        "total_upserted": total_upserted,
        "total_errors": total_errors,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info("Ingestion complete: %s", summary)
    return summary
