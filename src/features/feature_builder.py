"""
Feature builder — combines all descriptors into the final 583-dimensional vector.

Vector layout (583d total):
  [512] FashionCLIP    weight 0.70  → l2-normalized
  [ 36] Texture Gabor  weight 0.10  → l2-normalized
  [ 20] Shape PCA      weight 0.08  → l2-normalized
  [  5] Category       weight 0.05  → l2-normalized
  [  3] Color LAB      weight 0.04  → divided by [100, 128, 128]
  [  1] Formality      weight 0.02  → scalar in [0, 1]
  [  2] Season cyclic  weight 0.005 → sin/cos in [-1, 1]
  [  4] Proportions    weight 0.005 → floats in [0, 1]
 ────────────────────────────────────────────────────────
 Total: 512+36+20+5+3+1+2+4 = 583d

Viewpoint is stored in metadata only (weight 0.00).
"""
import math
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Category lookup tables
# ---------------------------------------------------------------------------

# Axes: [body_part_normalized, is_outer, sleeve_len_normalized, is_dress, is_layering]
CATEGORY_DECOMPOSITION: Dict[str, List[float]] = {
    "short sleeve top":       [0.0, 0, 0.33, 0, 0],
    "long sleeve top":        [0.0, 0, 1.0,  0, 0],
    "short sleeve outwear":   [0.0, 1, 0.33, 0, 1],
    "long sleeve outwear":    [0.0, 1, 1.0,  0, 1],
    "vest":                   [0.0, 0, 0.0,  0, 0],
    "sling":                  [0.0, 0, 0.0,  0, 0],
    "shorts":                 [0.5, 0, 0.33, 0, 0],
    "trousers":               [0.5, 0, 1.0,  0, 0],
    "skirt":                  [0.5, 0, 1.0,  0, 0],
    "short sleeve dress":     [1.0, 0, 0.33, 1, 0],
    "long sleeve dress":      [1.0, 0, 1.0,  1, 0],
    "vest dress":             [1.0, 0, 0.0,  1, 0],
    "sling dress":            [1.0, 0, 0.0,  1, 0],
}

FORMALITY_SCORE: Dict[str, float] = {
    "sling": 0.0,
    "shorts": 0.15,
    "sling dress": 0.2,
    "vest": 0.25,
    "short sleeve top": 0.3,
    "short sleeve dress": 0.4,
    "skirt": 0.5,
    "long sleeve top": 0.55,
    "long sleeve dress": 0.65,
    "vest dress": 0.7,
    "trousers": 0.75,
    "short sleeve outwear": 0.8,
    "long sleeve outwear": 0.9,
}

# Season inference sets
_SUMMER_CATEGORIES = {"sling", "shorts", "sling dress", "short sleeve top", "short sleeve dress"}
_WINTER_CATEGORIES = {"long sleeve outwear", "trousers", "long sleeve top"}
_AUTUMN_CATEGORIES = {"long sleeve dress", "vest dress", "skirt", "vest"}
_SPRING_CATEGORIES = {"short sleeve outwear"}

SEASON_ANGLES: Dict[str, int] = {
    "spring": 0,
    "summer": 90,
    "autumn": 180,
    "winter": 270,
}

_COLOR_NORM = np.array([100.0, 128.0, 128.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def infer_season(category_name: str) -> str:
    """Infer season string from category name. Defaults to 'summer'."""
    cat = category_name.lower()
    if cat in _SUMMER_CATEGORIES:
        return "summer"
    if cat in _WINTER_CATEGORIES:
        return "winter"
    if cat in _AUTUMN_CATEGORIES:
        return "autumn"
    if cat in _SPRING_CATEGORIES:
        return "spring"
    return "summer"


def encode_season_cyclic(season: str) -> np.ndarray:
    """Encode season as 2d cyclic vector (sin, cos) in [-1, 1]."""
    angle_deg = SEASON_ANGLES.get(season.lower(), 90)
    angle_rad = math.radians(angle_deg)
    return np.array([math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32)


def encode_viewpoint_cyclic(vp_id: int) -> np.ndarray:
    """
    Encode viewpoint ID as 2d cyclic vector.
    IDs: 0=no wear (skipped upstream), 1=frontal, 2=side, 3=back.
    Maps to angles: 1→0°, 2→90°, 3→180°.
    """
    angle_map = {0: 0, 1: 0, 2: 90, 3: 180}
    angle_deg = angle_map.get(vp_id, 0)
    angle_rad = math.radians(angle_deg)
    return np.array([math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32)


def extract_proportions(
    bbox: List[int],
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Compute normalized bounding-box proportion features.

    Parameters
    ----------
    bbox    : [x1, y1, x2, y2]
    img_w   : image width in pixels
    img_h   : image height in pixels

    Returns
    -------
    np.ndarray of shape (4,): [rel_w, rel_h, aspect, coverage]
    """
    x1, y1, x2, y2 = bbox
    rel_w = (x2 - x1) / (img_w + 1e-8)
    rel_h = (y2 - y1) / (img_h + 1e-8)
    aspect = rel_w / (rel_h + 1e-8)
    coverage = rel_w * rel_h
    return np.array([rel_w, rel_h, aspect, coverage], dtype=np.float32)


# ---------------------------------------------------------------------------
# Core combination function
# ---------------------------------------------------------------------------

def build_combined_vector(
    clip_vec: np.ndarray,       # (512,)
    texture_vec: np.ndarray,    # (36,)
    shape_vec: np.ndarray,      # (20,)
    cat_vec: np.ndarray,        # (5,)
    color_lab: np.ndarray,      # (3,)
    formality: float,           # scalar
    season_vec: np.ndarray,     # (2,)
    proportions: np.ndarray,    # (4,)
) -> np.ndarray:
    """
    Concatenate and weight all descriptors into a single 583d vector.

    L2-normalized components: clip, texture, shape, category.
    Scaled components: color (÷[100,128,128]), formality, season, proportions.
    """
    def _l2(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-8)

    combined = np.concatenate([
        _l2(clip_vec)                                  * 0.70,   # 512d
        _l2(texture_vec)                               * 0.10,   #  36d
        _l2(shape_vec)                                 * 0.08,   #  20d
        _l2(cat_vec)                                   * 0.05,   #   5d
        (color_lab / _COLOR_NORM)                      * 0.04,   #   3d
        np.array([formality], dtype=np.float32)        * 0.02,   #   1d
        season_vec                                     * 0.005,  #   2d
        proportions                                    * 0.005,  #   4d
    ])
    # Sanity check (dev-time only)
    assert combined.shape == (583,), f"Expected 583d, got {combined.shape}"
    return combined.astype(np.float32)


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_metadata(
    split: str,
    base_filename: str,
    item_key: str,
    source: str,
    category_name: str,
    category_id: int,
    pair_id: int,
    formality: float,
    season: str,
    viewpoint: int,
    color_lab: np.ndarray,
    scale: int,
    occlusion: int,
    image_path: str,
) -> dict:
    """Assemble the Pinecone metadata dict for one item."""
    return {
        "item_id": f"{split}_{base_filename}_{item_key}",
        "source": source,
        "category_name": category_name,
        "category_id": category_id,
        "pair_id": pair_id,
        "formality": float(formality),
        "season": season,
        "viewpoint": viewpoint,
        "color_l": float(color_lab[0]),
        "color_a": float(color_lab[1]),
        "color_b": float(color_lab[2]),
        "scale": scale,
        "occlusion": occlusion,
        "image_path": image_path,
    }
