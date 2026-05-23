"""
Landmark shape descriptor via Procrustes normalization + PCA.

Pipeline
--------
1. Parse the flat 294-int landmark list into (98, 3) — [x, y, visibility].
2. Filter to only visible points (visibility > 0).
3. Procrustes normalization: center → scale to unit size.
4. Flatten and zero-pad to fixed length 196 (98 × 2).
5. PCA(n_components=20) projects to a 20-d shape vector.

fit_pca()    — collects all normalized landmark arrays, fits PCA, saves to disk.
extract_shape() — transforms a single landmark list to 20d via saved PCA.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

_PAD_LENGTH = 196  # 98 landmarks × 2 coords
_N_COMPONENTS = 20


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_and_normalize(landmarks_raw: List[int]) -> Optional[np.ndarray]:
    """
    Parse a flat landmark list and return a normalized, zero-padded (196,) array.
    Returns None if no visible points are found.
    """
    if not landmarks_raw or len(landmarks_raw) % 3 != 0:
        return None

    pts = np.array(landmarks_raw, dtype=np.float64).reshape(-1, 3)  # (N, 3)
    visible = pts[pts[:, 2] > 0, :2]  # only x, y for visible pts

    if len(visible) == 0:
        return None

    # --- Procrustes normalization ---
    centered = visible - visible.mean(axis=0)
    scale = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
    if scale < 1e-8:
        return None
    normalized = centered / scale

    # Zero-pad to fixed length 196
    flat = normalized.flatten()  # length = num_visible × 2
    padded = np.zeros(_PAD_LENGTH, dtype=np.float32)
    length = min(len(flat), _PAD_LENGTH)
    padded[:length] = flat[:length]
    return padded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_pca(
    all_landmark_lists: List[List[int]],
    save_path: Path,
    min_samples: int = 1000,
) -> PCA:
    """
    Fit PCA on a collection of raw landmark lists and persist to disk.

    Parameters
    ----------
    all_landmark_lists : list of flat 294-int lists
    save_path : where to save the fitted PCA (pickle)
    min_samples : warn if fewer samples than this are available

    Returns
    -------
    Fitted sklearn PCA object.
    """
    arrays = []
    for lm in all_landmark_lists:
        arr = _parse_and_normalize(lm)
        if arr is not None:
            arrays.append(arr)

    n = len(arrays)
    if n < min_samples:
        logger.warning(
            "Only %d valid landmark arrays collected for PCA fitting "
            "(recommended ≥ %d). Results may be poor.",
            n, min_samples,
        )
    if n == 0:
        raise ValueError("No valid landmark arrays found — cannot fit PCA.")

    X = np.stack(arrays)  # (N, 196)
    pca = PCA(n_components=_N_COMPONENTS, random_state=42)
    pca.fit(X)
    logger.info("PCA fitted on %d samples, explaining %.1f%% variance.", n,
                pca.explained_variance_ratio_.sum() * 100)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pca, f)
    logger.info("PCA model saved to %s.", save_path)
    return pca


def load_pca(path: Path) -> PCA:
    """Load a previously fitted PCA model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_shape(landmarks_raw: List[int], pca_model: PCA) -> np.ndarray:
    """
    Transform a single flat landmark list to a 20-d shape vector.

    Returns np.zeros(20) for empty or invalid landmark data.
    """
    arr = _parse_and_normalize(landmarks_raw)
    if arr is None:
        return np.zeros(_N_COMPONENTS, dtype=np.float32)

    projected = pca_model.transform(arr.reshape(1, -1))  # (1, 20)
    return projected.flatten().astype(np.float32)
