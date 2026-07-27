import asyncio
import hashlib
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "")
TEXT_COLLECTION = "text_chunks"
FIGURES_COLLECTION = "visual_index"
DENSE_DIM = 384
DENSE_VECTOR_NAME = "dense"

_client: QdrantClient | None = None


def _point_id(*parts: str) -> int:
    """Derive a stable point ID from content-identifying parts.

    Unlike Python's built-in hash(), this is deterministic across
    process restarts, so re-ingesting the same PDF overwrites its
    existing points instead of creating duplicates.

    Args:
        *parts: Strings that uniquely identify the point.

    Returns:
        A stable non-negative integer ID.
    """
    key = "|".join(parts)
    return int(hashlib.md5(key.encode()).hexdigest()[:16], 16)


def _get_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _client
    if _client is None:
        if not QDRANT_URL:
            _client = QdrantClient(":memory:")
            print("[qdrant_client] Using in-memory Qdrant (no QDRANT_URL set)")
        else:
            QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
            api_key = QDRANT_API_KEY
            if api_key and "cloud.qdrant.io" in QDRANT_URL:
                _client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=api_key,
                    port=443,
                    https=True,
                    prefer_grpc=False
                )
            else:
                _client = QdrantClient(url=QDRANT_URL)
            print(f"[qdrant_client] Connected to Qdrant at {QDRANT_URL}")
    return _client


async def _ensure_collection_dims(client, collection_name: str) -> bool:
    """Return True if collection exists with correct dim, else delete and return False."""
    loop = asyncio.get_event_loop()
    try:
        info = await loop.run_in_executor(
            None, lambda: client.get_collection(collection_name)
        )
        vectors_config = info.config.params.vectors
        # vectors_config can be a VectorParams (unnamed) or dict of name -> VectorParams
        if isinstance(vectors_config, dict):
            actual_dim = vectors_config[DENSE_VECTOR_NAME].size
        else:
            actual_dim = vectors_config.size
        if actual_dim == DENSE_DIM:
            return True
        print(
            f"[qdrant_client] Collection {collection_name} has dim={actual_dim}, "
            f"expected {DENSE_DIM}. Deleting..."
        )
    except Exception as e:
        print(f"[qdrant_client] Could not inspect {collection_name}: {e}")
    # Delete (ignore if already gone)
    try:
        await loop.run_in_executor(
            None, lambda: client.delete_collection(collection_name)
        )
    except Exception:
        pass
    return False


async def _create_text_collection(client) -> None:
    """Create text_chunks collection with payload index."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: client.create_collection(
            collection_name=TEXT_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE,
                )
            },
        ),
    )
    print(f"[qdrant_client] Created collection: {TEXT_COLLECTION}")
    await loop.run_in_executor(
        None,
        lambda: client.create_payload_index(
            collection_name=TEXT_COLLECTION,
            field_name="source_pdf",
            field_schema="keyword"
        )
    )
    print(f"[qdrant_client] Created index: {TEXT_COLLECTION}.source_pdf")


async def _create_figures_collection(client) -> None:
    """Create visual_index collection with payload index."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: client.create_collection(
            collection_name=FIGURES_COLLECTION,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=DENSE_DIM,
                    distance=Distance.COSINE
                )
            }
        )
    )
    print(f"[qdrant_client] Created collection: {FIGURES_COLLECTION}")
    await loop.run_in_executor(
        None,
        lambda: client.create_payload_index(
            collection_name=FIGURES_COLLECTION,
            field_name="source_pdf",
            field_schema="keyword"
        )
    )
    print(f"[qdrant_client] Created index: {FIGURES_COLLECTION}.source_pdf")


async def init_collections() -> None:
    """Create text_chunks and visual_index collections if they do not exist."""
    loop = asyncio.get_event_loop()
    client = _get_client()
    existing = await loop.run_in_executor(None, client.get_collections)
    existing_names = [c.name for c in existing.collections]

    if TEXT_COLLECTION in existing_names:
        valid = await _ensure_collection_dims(client, TEXT_COLLECTION)
        if not valid:
            existing_names.remove(TEXT_COLLECTION)

    if FIGURES_COLLECTION in existing_names:
        valid = await _ensure_collection_dims(client, FIGURES_COLLECTION)
        if not valid:
            existing_names.remove(FIGURES_COLLECTION)

    if TEXT_COLLECTION not in existing_names:
        await _create_text_collection(client)

    if FIGURES_COLLECTION not in existing_names:
        await _create_figures_collection(client)


async def upsert_text_chunks(chunks: list[dict]) -> int:
    """Insert text chunks with dense vectors into Qdrant.

    Args:
        chunks: List of dicts from text_embedder.embed_chunks()

    Returns:
        Number of points upserted.
    """
    if not chunks:
        return 0
    loop = asyncio.get_event_loop()
    client = _get_client()
    points = []
    for i, chunk in enumerate(chunks):
        point = PointStruct(
            id=_point_id(
                chunk.get("source_pdf", ""),
                str(chunk.get("page_num", 0)),
                str(chunk.get("chunk_index", 0)),
            ),
            vector={
                DENSE_VECTOR_NAME: chunk["dense_vector"],
            },
            payload={
                "chunk": chunk["chunk"],
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 0),
                "source_pdf": chunk.get("source_pdf", ""),
                "page_num": chunk.get("page_num", 0),
                "page_total": chunk.get("page_total", 0),
                "chapter": chunk.get("chapter", ""),
                "section_title": chunk.get("section_title", ""),
                "word_count": chunk.get("word_count", 0),
                "is_first_chunk": chunk.get("is_first_chunk", False),
                "doc_id": chunk.get("doc_id", ""),
                "ingested_at": chunk.get("ingested_at", "")
            },
        )
        points.append(point)
    await loop.run_in_executor(
        None,
        lambda: client.upsert(collection_name=TEXT_COLLECTION, points=points),
    )
    print(f"[qdrant_client] Upserted {len(points)} text chunks.")
    return len(points)


async def search_text(
    dense_vector: list[float],
    limit: int = 5,
    source_pdf: str | None = None,
) -> list[dict]:
    """Search text_chunks collection using dense vector.

    Args:
        dense_vector: Query dense vector from text_embedder
        limit: Number of results to return
        source_pdf: If set, restrict results to this PDF filename only.

    Returns:
        List of dicts with chunk, page_num, source_pdf, chunk_index, score
    """
    loop = asyncio.get_event_loop()
    client = _get_client()
    query_filter = (
        Filter(must=[FieldCondition(key="source_pdf", match=MatchValue(value=source_pdf))])
        if source_pdf
        else None
    )
    results = await loop.run_in_executor(
        None,
        lambda: client.query_points(
            collection_name=TEXT_COLLECTION,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=query_filter,
            limit=limit,
        ),
    )
    return [
        {
            "chunk": r.payload.get("chunk", ""),
            "chunk_index": r.payload.get("chunk_index", 0),
            "total_chunks": r.payload.get("total_chunks", 0),
            "source_pdf": r.payload.get("source_pdf", ""),
            "page_num": r.payload.get("page_num", 0),
            "page_total": r.payload.get("page_total", 0),
            "chapter": r.payload.get("chapter", ""),
            "section_title": r.payload.get("section_title", ""),
            "word_count": r.payload.get("word_count", 0),
            "is_first_chunk": r.payload.get("is_first_chunk", False),
            "doc_id": r.payload.get("doc_id", ""),
            "ingested_at": r.payload.get("ingested_at", ""),
            "score": r.score,
        }
        for r in results.points
    ]


async def upsert_figures(figures: list[dict]) -> int:
    """Insert figure descriptions into visual_index collection.

    Args:
        figures: List of figure dicts from visual_indexer
                 with description, keywords, figure_path etc.
                 Must also include dense_vector from MiniLM.

    Returns:
        Number of figures upserted.
    """
    if not figures:
        return 0
    loop = asyncio.get_event_loop()
    client = _get_client()
    points = []
    for fig in figures:
        point = PointStruct(
            id=_point_id(fig.get("source_pdf", ""), fig.get("figure_filename", "")),
            vector={
                DENSE_VECTOR_NAME: fig["dense_vector"]
            },
            payload={
                "description": fig.get("description", ""),
                "keywords": fig.get("keywords", []),
                "subject": fig.get("subject", ""),
                "has_diagram": fig.get("has_diagram", False),
                "has_table": fig.get("has_table", False),
                "has_formula": fig.get("has_formula", False),
                "source_pdf": fig.get("source_pdf", ""),
                "page_num": fig.get("page_num", 0),
                "figure_index": fig.get("figure_index", 0),
                "figure_filename": fig.get("figure_filename", ""),
                "cloudinary_url": fig.get("cloudinary_url", ""),
                "cloudinary_public_id": fig.get("cloudinary_public_id", ""),
                "doc_id": fig.get("doc_id", ""),
                "ingested_at": fig.get("ingested_at", "")
            }
        )
        points.append(point)
    await loop.run_in_executor(
        None,
        lambda: client.upsert(
            collection_name=FIGURES_COLLECTION,
            points=points
        )
    )
    print(f"[qdrant_client] Upserted {len(points)} figures.")
    return len(points)


async def search_figures_collection(
    dense_vector: list[float],
    source_pdf: str,
    limit: int = 3
) -> list[dict]:
    """Search visual_index collection for matching figures.

    Args:
        dense_vector: Query dense vector from MiniLM.
        source_pdf: PDF filename to filter by.
        limit: Max results.

    Returns:
        List of figure dicts with score.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    loop = asyncio.get_event_loop()
    client = _get_client()
    results = await loop.run_in_executor(
        None,
        lambda: client.query_points(
            collection_name=FIGURES_COLLECTION,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_pdf",
                        match=MatchValue(value=source_pdf)
                    )
                ]
            ),
            limit=limit
        )
    )
    return [
        {
            "figure_filename": r.payload.get("figure_filename", ""),
            "page_num": r.payload.get("page_num", 0),
            "source_pdf": r.payload.get("source_pdf", ""),
            "description": r.payload.get("description", ""),
            "keywords": r.payload.get("keywords", []),
            "has_diagram": r.payload.get("has_diagram", False),
            "subject": r.payload.get("subject", ""),
            "doc_id": r.payload.get("doc_id", ""),
            "score": r.score
        }
        for r in results.points
    ]


async def delete_pdf_data(source_pdf: str) -> dict:
    """Delete all Qdrant data associated with a PDF.

    Args:
        source_pdf: PDF filename e.g. "book.pdf"

    Returns:
        Dict with keys:
          - text_chunks_deleted: bool
          - figures_deleted: bool
          - status: str
    """
    loop = asyncio.get_event_loop()
    client = _get_client()

    pdf_filter = Filter(
        must=[
            FieldCondition(
                key="source_pdf",
                match=MatchValue(value=source_pdf)
            )
        ]
    )
    selector = FilterSelector(filter=pdf_filter)

    text_chunks_deleted = False
    figures_deleted = False

    try:
        await loop.run_in_executor(
            None,
            lambda: client.delete(
                collection_name=TEXT_COLLECTION,
                points_selector=selector,
            )
        )
        print(f"[qdrant_client] Deleted text chunks for {source_pdf}")
        text_chunks_deleted = True
    except Exception as e:
        print(f"[qdrant_client] Error deleting text chunks: {e}")

    try:
        await loop.run_in_executor(
            None,
            lambda: client.delete(
                collection_name=FIGURES_COLLECTION,
                points_selector=selector,
            )
        )
        print(f"[qdrant_client] Deleted figures for {source_pdf}")
        figures_deleted = True
    except Exception as e:
        print(f"[qdrant_client] Error deleting figures: {e}")

    return {
        "text_chunks_deleted": text_chunks_deleted,
        "figures_deleted": figures_deleted,
        "status": "success",
    }
