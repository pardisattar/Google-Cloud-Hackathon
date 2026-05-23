"""
Dominant color extraction in CIELAB color space.

Uses KMeans(k=3) on a downsampled 64×64 image to find the dominant cluster,
then converts that cluster's RGB centroid to LAB.
"""
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import KMeans


def extract_dominant_color_lab(pil_image: Image.Image) -> np.ndarray:
    """
    Extract the dominant color from *pil_image* as a CIELAB triplet.

    Steps
    -----
    1. Resize to 64×64 for speed.
    2. Flatten to (4096, 3) in [0, 1] float32.
    3. KMeans(k=3, n_init=5) to find three colour clusters.
    4. The dominant cluster = argmax of pixel counts per cluster.
    5. Convert dominant centroid (RGB [0,1]) → LAB.

    Returns
    -------
    np.ndarray of shape (3,)  — [L*, a*, b*]
    """
    img = pil_image.convert("RGB").resize((64, 64), Image.BILINEAR)
    pixels = np.asarray(img, dtype=np.float32) / 255.0  # (64, 64, 3)
    pixels_flat = pixels.reshape(-1, 3)  # (4096, 3)

    km = KMeans(n_clusters=3, n_init=5, random_state=42)
    labels = km.fit_predict(pixels_flat)

    # Most dominant cluster by pixel count
    dominant_idx = int(np.argmax(np.bincount(labels)))
    centroid_rgb = km.cluster_centers_[dominant_idx]  # shape (3,) in [0, 1]

    # rgb2lab expects shape (H, W, 3) with values in [0, 1]
    centroid_rgb_hwc = centroid_rgb.reshape(1, 1, 3)
    lab = rgb2lab(centroid_rgb_hwc)  # shape (1, 1, 3)

    return lab.flatten().astype(np.float32)  # (3,): [L*, a*, b*]
