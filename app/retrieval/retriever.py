"""Text retrieval using MiniLM dense search."""

from typing import Any

import app.ingestion.qdrant_client as qc


async def retrieve(
    query_dense: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Perform text retrieval using MiniLM dense search.

    Args:
        query_dense: Dense vector (384-dim) from MiniLM.
        limit: Max results to return.

    Returns:
        List of text results sorted by relevance.
    """
    await qc.init_collections()
    return await qc.search_text(query_dense, limit=limit)


async def retrieve_text_only(
    query_dense: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Text-only retrieval.

    Args:
        query_dense: Dense vector from MiniLM.
        limit: Max results to return.

    Returns:
        List of text results.
    """
    return await retrieve(query_dense, limit=limit)
