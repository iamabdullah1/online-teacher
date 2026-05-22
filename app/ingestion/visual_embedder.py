"""Visual embedder for page images.

DEPENDENCY NOTE:
This module requires colpali-engine which has conflicting dependencies:
- colpali-engine requires transformers>=5.3.0
- FlagEmbedding (used in text_embedder) requires transformers<5.0

For production, consider using separate virtual environments:
    # Environment 1 (text embeddings)
    pip install FlagEmbedding 'transformers<5.0'

    # Environment 2 (visual embeddings)
    pip install colpali-engine 'transformers>=5.3.0' 'torch>=2.4'

This stub provides the interface for future ColPali integration.
"""
import asyncio
from pathlib import Path
from PIL import Image


async def embed_images(image_paths: list[str]) -> list[dict]:
    """Embed a list of page images using ColPali.

    Args:
        image_paths: List of absolute paths to PNG images.

    Returns:
        List of dicts with keys:
          - image_path: str (original path)
          - vectors: list[list[float]] (multi-vector, one per patch)
          - vector_count: int (number of patch vectors)
    """
    if not image_paths:
        return []
    print(
        "[visual_embedder] WARNING: Using stub implementation. "
        "Install colpali-engine in separate environment for actual embeddings."
    )
    results = []
    for image_path in image_paths:
        results.append({
            "image_path": image_path,
            "vectors": [[0.0] * 128],
            "vector_count": 1,
        })
    return results


async def embed_query_image(query_text: str) -> list[float]:
    """Embed a text query for visual search using ColPali.

    Args:
        query_text: The search query string.

    Returns:
        Single query vector as list[float] (128 dims).
    """
    print(
        "[visual_embedder] WARNING: Using stub implementation. "
        "Install colpali-engine in separate environment for actual embeddings."
    )
    return [0.0] * 128
