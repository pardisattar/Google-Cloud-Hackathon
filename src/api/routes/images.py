"""
GET /image?path=... — serve dataset images by relative file path.

Security: only paths inside the data/ directory are allowed.
"""
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/image", tags=["images"])
def get_image(path: str) -> FileResponse:
    """
    Serve a dataset image by its relative path.

    Parameters
    ----------
    path : str
        Relative path, e.g. ``data/raw/train/image/000045.jpg``.
        Must resolve inside the ``data/`` directory.

    Returns
    -------
    FileResponse
        The image file with ``image/jpeg`` media type.
    """
    abs_path = os.path.abspath(path)
    data_dir = os.path.abspath("data")

    # Security: reject any path that escapes the data/ directory
    if not abs_path.startswith(data_dir + os.sep) and abs_path != data_dir:
        logger.warning("Blocked image path traversal attempt: %s", path)
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Detect media type from extension; default to jpeg
    ext = os.path.splitext(abs_path)[1].lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")

    return FileResponse(abs_path, media_type=media_type)
