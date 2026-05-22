"""Hybrid retrieval combining text and visual search with Reciprocal Rank Fusion."""

from typing import Any

import app.ingestion.qdrant_client as qc


def _rrf_score(rank: int, k: int = 60) -> float:
    """Calculate reciprocal rank fusion score.

    Args:
        rank: Position in result list (1-indexed).
        k: RRF constant (default 60).

    Returns:
        RRF score as float.
    """
    return 1.0 / (k + rank)


async def retrieve(
    query: str,
    query_dense: list[float],
    query_sparse: dict[int, float],
    query_visual: list[float] | None = None,
    text_limit: int = 10,
    visual_limit: int = 5,
    fusion_limit: int = 10,
) -> list[dict[str, Any]]:
    """Perform hybrid retrieval combining text and visual results via RRF.

    Args:
        query: Original query text.
        query_dense: Dense vector (1024-dim) from BGE-M3.
        query_sparse: Sparse vector dict from BGE-M3.
        query_visual: Visual vector (128-dim) from ColPali (optional).
        text_limit: Max text results to retrieve.
        visual_limit: Max visual results to retrieve.
        fusion_limit: Final fused results to return.

    Returns:
        Combined results sorted by fused RRF score.
    """
    await qc.init_collections()

    # Text search (always performed)
    text_results = await qc.search_text(query_dense, query_sparse, limit=text_limit)

    # Build text rankings
    text_rank_map: dict[str, tuple[int, float]] = {}
    for idx, res in enumerate(text_results):
        key = f"text:{res.get('source_pdf', '')}:{res.get('page_num', 0)}"
        text_rank_map[key] = (idx + 1, res.get("score", 0.0))

    # Visual search (if visual vector provided)
    visual_results: list[dict[str, Any]] = []
    if query_visual:
        visual_results = await qc.search_visual(query_visual, limit=visual_limit)

    # Build visual rankings
    visual_rank_map: dict[str, tuple[int, float]] = {}
    for idx, res in enumerate(visual_results):
        key = f"visual:{res.get('image_path', '')}"
        visual_rank_map[key] = (idx + 1, res.get("score", 0.0))

    # Combine all results with RRF
    all_keys = set(text_rank_map.keys()) | set(visual_rank_map.keys())
    fused_scores: list[tuple[str, float]] = []

    for key in all_keys:
        rrf = 0.0
        # Text contribution
        if key in text_rank_map:
            rank, sim = text_rank_map[key]
            rrf += _rrf_score(rank) * sim
        # Visual contribution
        if key in visual_rank_map:
            rank, sim = visual_rank_map[key]
            rrf += _rrf_score(rank) * sim
        fused_scores.append((key, rrf))

    # Sort by RRF score descending
    fused_scores.sort(key=lambda x: x[1], reverse=True)

    # Build final results with source info
    results: list[dict[str, Any]] = []
    for key, score in fused_scores[:fusion_limit]:
        parts = key.split(":", 2)
        if parts[0] == "text":
            source_pdf = parts[1] if len(parts) > 1 else ""
            page_num = int(parts[2]) if len(parts) > 2 else 0
            # Find original text result
            for tr in text_results:
                if tr.get("source_pdf") == source_pdf and tr.get("page_num") == page_num:
                    results.append({
                        "type": "text",
                        "chunk": tr.get("chunk", ""),
                        "page_num": page_num,
                        "source_pdf": source_pdf,
                        "chunk_index": tr.get("chunk_index", 0),
                        "score": score,
                    })
                    break
        elif parts[0] == "visual":
            image_path = parts[1] if len(parts) > 1 else ""
            # Find original visual result
            for vr in visual_results:
                if vr.get("image_path") == image_path:
                    results.append({
                        "type": "visual",
                        "image_path": image_path,
                        "score": score,
                    })
                    break

    return results


async def retrieve_text_only(
    query_dense: list[float],
    query_sparse: dict[int, float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Text-only retrieval fallback.

    Args:
        query_dense: Dense vector from BGE-M3.
        query_sparse: Sparse vector from BGE-M3.
        limit: Max results to return.

    Returns:
        List of text results.
    """
    await qc.init_collections()
    return await qc.search_text(query_dense, query_sparse, limit=limit)