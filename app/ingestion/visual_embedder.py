"""
Visual embedder — calls ColPali microservice via HTTP.
ColPali runs in a separate Docker container to avoid
dependency conflicts with FlagEmbedding/transformers.
"""

import os
import asyncio
import httpx
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

COLPALI_SERVICE_URL = os.getenv(
    "COLPALI_SERVICE_URL",
    "http://localhost:8001"
)
REQUEST_TIMEOUT = 120.0  # ColPali is slow on CPU


async def embed_images(image_paths: list[str]) -> list[dict]:
    """Embed page images by calling the ColPali microservice.

    Args:
        image_paths: List of absolute paths to PNG images.

    Returns:
        List of dicts with keys:
          - image_path: str
          - vectors: list[list[float]] (multi-vector)
          - vector_count: int
    """
    if not image_paths:
        return []
    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT
        ) as client:
            response = await client.post(
                f"{COLPALI_SERVICE_URL}/embed/images",
                json={"image_paths": image_paths}
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data["results"]:
                results.append({
                    "image_path": item["image_path"],
                    "vectors": item["vectors"],
                    "vector_count": item["vector_count"]
                })
            print(f"[visual_embedder] Embedded {len(results)} images.")
            return results
    except httpx.ConnectError:
        print(
            "[visual_embedder] WARNING: ColPali service not running. "
            "Start it with: docker-compose up colpali"
        )
        return _stub_embeddings(image_paths)
    except Exception as e:
        print(f"[visual_embedder] Error: {e}")
        return _stub_embeddings(image_paths)


async def embed_query_image(query_text: str) -> list[float]:
    """Embed a text query for visual search via ColPali service.

    Args:
        query_text: The search query string.

    Returns:
        Query vector as list[float] (128 dims).
    """
    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT
        ) as client:
            response = await client.post(
                f"{COLPALI_SERVICE_URL}/embed/query",
                json={"query_text": query_text}
            )
            response.raise_for_status()
            return response.json()["vector"]
    except httpx.ConnectError:
        print(
            "[visual_embedder] WARNING: ColPali service not running."
        )
        return [0.0] * 128
    except Exception as e:
        print(f"[visual_embedder] Error: {e}")
        return [0.0] * 128


def _stub_embeddings(image_paths: list[str]) -> list[dict]:
    """Return stub embeddings when ColPali service is unavailable.

    Args:
        image_paths: List of image paths.

    Returns:
        List of stub embedding dicts with zero vectors.
    """
    return [
        {
            "image_path": path,
            "vectors": [[0.0] * 128],
            "vector_count": 1
        }
        for path in image_paths
    ]