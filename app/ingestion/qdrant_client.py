import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
    NamedVector,
    SearchRequest,
    Filter,
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
TEXT_COLLECTION = "text_chunks"
VISUAL_COLLECTION = "visual_pages"
DENSE_DIM = 1024
VISUAL_DIM = 128
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Get or create Qdrant client instance."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        print(f"[qdrant_client] Connected to Qdrant at {QDRANT_URL}")
    return _client


async def init_collections() -> None:
    """Create text_chunks and visual_pages collections if they do not exist."""
    loop = asyncio.get_event_loop()
    client = _get_client()
    existing = await loop.run_in_executor(None, client.get_collections)
    existing_names = [c.name for c in existing.collections]

    if TEXT_COLLECTION not in existing_names:
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
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            ),
        )
        print(f"[qdrant_client] Created collection: {TEXT_COLLECTION}")

    if VISUAL_COLLECTION not in existing_names:
        await loop.run_in_executor(
            None,
            lambda: client.create_collection(
                collection_name=VISUAL_COLLECTION,
                vectors_config=VectorParams(
                    size=VISUAL_DIM,
                    distance=Distance.COSINE,
                ),
            ),
        )
        print(f"[qdrant_client] Created collection: {VISUAL_COLLECTION}")


async def upsert_text_chunks(chunks: list[dict]) -> int:
    """Insert text chunks with dense and sparse vectors into Qdrant.

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
            id=abs(hash(chunk["chunk"])) % (2**63),
            vector={
                DENSE_VECTOR_NAME: chunk["dense_vector"],
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=list(chunk["sparse_vector"].keys()),
                    values=list(chunk["sparse_vector"].values()),
                ),
            },
            payload={
                "chunk": chunk["chunk"],
                "page_num": chunk.get("page_num", 0),
                "source_pdf": chunk.get("source_pdf", ""),
                "chunk_index": chunk.get("chunk_index", i),
            },
        )
        points.append(point)
    await loop.run_in_executor(
        None,
        lambda: client.upsert(collection_name=TEXT_COLLECTION, points=points),
    )
    print(f"[qdrant_client] Upserted {len(points)} text chunks.")
    return len(points)


async def upsert_visual_pages(pages: list[dict]) -> int:
    """Insert visual page embeddings into Qdrant.

    Args:
        pages: List of dicts from visual_embedder.embed_images()

    Returns:
        Number of points upserted.
    """
    if not pages:
        return 0
    loop = asyncio.get_event_loop()
    client = _get_client()
    points = []
    for page in pages:
        for j, vector in enumerate(page["vectors"]):
            point = PointStruct(
                id=abs(hash(f"{page['image_path']}_{j}")) % (2**63),
                vector=vector,
                payload={
                    "image_path": page["image_path"],
                    "page_num": page.get("page_num", 0),
                    "source_pdf": page.get("source_pdf", ""),
                    "patch_index": j,
                },
            )
            points.append(point)
    await loop.run_in_executor(
        None,
        lambda: client.upsert(collection_name=VISUAL_COLLECTION, points=points),
    )
    print(f"[qdrant_client] Upserted {len(points)} visual vectors.")
    return len(points)


async def search_text(
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    limit: int = 5,
) -> list[dict]:
    """Search text_chunks collection using dense vector.

    Args:
        dense_vector: Query dense vector from text_embedder
        sparse_vector: Query sparse vector from text_embedder
        limit: Number of results to return

    Returns:
        List of dicts with chunk, page_num, source_pdf, chunk_index, score
    """
    loop = asyncio.get_event_loop()
    client = _get_client()
    results = await loop.run_in_executor(
        None,
        lambda: client.query_points(
            collection_name=TEXT_COLLECTION,
            query=dense_vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
        ),
    )
    return [
        {
            "chunk": r.payload.get("chunk", ""),
            "page_num": r.payload.get("page_num", 0),
            "source_pdf": r.payload.get("source_pdf", ""),
            "chunk_index": r.payload.get("chunk_index", 0),
            "score": r.score,
        }
        for r in results.points
    ]


async def search_visual(query_vector: list[float], limit: int = 5) -> list[dict]:
    """Search visual_pages collection.

    Args:
        query_vector: Query vector from visual_embedder
        limit: Number of results to return

    Returns:
        List of dicts with image_path, page_num, source_pdf, patch_index, score
    """
    loop = asyncio.get_event_loop()
    client = _get_client()
    results = await loop.run_in_executor(
        None,
        lambda: client.query_points(
            collection_name=VISUAL_COLLECTION,
            query=query_vector,
            limit=limit,
        ),
    )
    return [
        {
            "image_path": r.payload.get("image_path", ""),
            "page_num": r.payload.get("page_num", 0),
            "source_pdf": r.payload.get("source_pdf", ""),
            "patch_index": r.payload.get("patch_index", 0),
            "score": r.score,
        }
        for r in results.points
    ]
