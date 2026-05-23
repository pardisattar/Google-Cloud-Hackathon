"""
Garment segmentor — wraps rembg (U2-Net) to remove image backgrounds.

Provides:
  segment_garment()       — removes background, returns RGBA PIL Image
  segment_and_flatten()   — removes background and composites onto white RGB
                            (ready for FashionCLIP and other extractors)

Note: rembg downloads its U2-Net model weights (~170 MB) on the first call.
      This is expected and not an error. Subsequent calls use the cached model.
"""
import io

from PIL import Image
from rembg import remove


def segment_garment(pil_image: Image.Image) -> Image.Image:
    """
    Remove the background from a PIL Image using rembg (U2-Net).

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Input image in any mode (RGB, RGBA, L, …).

    Returns
    -------
    PIL.Image.Image
        RGBA image with background pixels set to transparent.
    """
    # Step 1: Ensure RGB input
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Step 2: PIL → PNG bytes
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    input_bytes = buf.getvalue()

    # Step 3: Remove background (returns PNG bytes)
    output_bytes = remove(input_bytes)

    # Step 4: PNG bytes → PIL Image (RGBA)
    result = Image.open(io.BytesIO(output_bytes)).convert("RGBA")

    return result


def segment_and_flatten(pil_image: Image.Image) -> Image.Image:
    """
    Segment the garment and composite it onto a white background.

    Downstream feature extractors (FashionCLIP, Gabor, KMeans color) expect
    a clean RGB image — not an RGBA image with transparency.  This function
    removes the background then composites the transparent RGBA result onto a
    solid white canvas, returning a plain RGB image.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Input image in any mode.

    Returns
    -------
    PIL.Image.Image
        RGB image with garment on a white background.
    """
    # Step 1: Segment → RGBA with background removed
    rgba = segment_garment(pil_image)

    # Step 2: White RGB canvas of the same size
    background = Image.new("RGB", rgba.size, (255, 255, 255))

    # Step 3: Paste garment onto white background using alpha channel as mask
    background.paste(rgba, mask=rgba.split()[3])  # channel index 3 = alpha

    return background
