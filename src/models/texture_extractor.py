"""
Gabor filter bank texture descriptor.

Computes a 36-dimensional texture descriptor using a bank of Gabor filters
applied to a grayscale, 128×128 version of the input image.

Layout: 3 frequencies × 6 orientations × 2 statistics (mean, std) = 36d
"""
import numpy as np
from PIL import Image
from skimage.filters import gabor

# Filter bank parameters
_FREQUENCIES = [0.1, 0.25, 0.4]
_THETAS_DEG = [0, 30, 60, 90, 120, 150]
_THETAS_RAD = [t * np.pi / 180.0 for t in _THETAS_DEG]


def extract_texture(pil_image: Image.Image) -> np.ndarray:
    """
    Compute a 36-dimensional Gabor texture descriptor.

    Steps
    -----
    1. Convert to grayscale and resize to 128×128.
    2. Apply Gabor filter for each (frequency, theta) combination.
    3. Compute magnitude = sqrt(real² + imag²).
    4. Append [magnitude.mean(), magnitude.std()] per filter pair.

    Returns
    -------
    np.ndarray of shape (36,)
    """
    gray = pil_image.convert("L").resize((128, 128), Image.BILINEAR)
    img_array = np.asarray(gray, dtype=np.float64) / 255.0

    descriptor = []
    for freq in _FREQUENCIES:
        for theta in _THETAS_RAD:
            real, imag = gabor(img_array, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            descriptor.append(magnitude.mean())
            descriptor.append(magnitude.std())

    return np.array(descriptor, dtype=np.float32)  # (36,)
