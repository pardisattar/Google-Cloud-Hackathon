"""
FashionCLIP embedding wrapper.

Loads patrickjohncyh/fashion-clip once at import time.
Provides embed_image() and embed_batch() for single and batch inference.
"""
import logging
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "patrickjohncyh/fashion-clip"

# ---------------------------------------------------------------------------
# Module-level singleton (loaded once)
# ---------------------------------------------------------------------------
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("FashionCLIP: loading model on %s …", _device)

_processor = CLIPProcessor.from_pretrained(MODEL_ID)
_model = CLIPModel.from_pretrained(MODEL_ID).to(_device).eval()

logger.info("FashionCLIP: model ready.")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_image(pil_image: Image.Image) -> np.ndarray:
    """
    Embed a single PIL image.

    Returns
    -------
    np.ndarray of shape (512,)
    """
    return embed_batch([pil_image])[0]


def embed_batch(images: List[Image.Image]) -> np.ndarray:
    """
    Embed a list of PIL images in one forward pass.

    Returns
    -------
    np.ndarray of shape (N, 512)
    """
    inputs = _processor(images=images, return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        outputs = _model.get_image_features(pixel_values=inputs["pixel_values"])
        # Some versions of transformers return a dict-like object here
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs
        features = F.normalize(features, dim=-1)
    return features.cpu().numpy()
