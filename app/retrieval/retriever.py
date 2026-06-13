"""Text retrieval using BGE-M3 dense + sparse search."""

from typing import Any

import app.ingestion.qdrant_client as qc


async def retrieve(
    query_dense: list[float],
    query_sparse: dict[int, float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Perform text retrieval using BGE-M3 hybrid search.

    Args:
        query_dense: Dense vector (1024-dim) from BGE-M3.
        query_sparse: Sparse vector dict from BGE-M3.
        limit: Max results to return.

    Returns:
        List of text results sorted by relevance.
    """
    await qc.init_collections()
    return await qc.search_text(query_dense, query_sparse, limit=limit)


async def retrieve_text_only(
    query_dense: list[float],
    query_sparse: dict[int, float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Text-only retrieval.

    Args:
        query_dense: Dense vector from BGE-M3.
        query_sparse: Sparse vector from BGE-M3.
        limit: Max results to return.

    Returns:
        List of text results.
    """
    return await retrieve(query_dense, query_sparse, limit=limit)