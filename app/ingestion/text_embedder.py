import asyncio
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 32

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load MiniLM model once and reuse. Downloads on first call (~80MB)."""
    global _model
    if _model is None:
        print(f"[text_embedder] Loading {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[text_embedder] Model loaded.")
    return _model


async def embed_chunks(chunks: list[str]) -> list[dict]:
    """Embed a list of text chunks using sentence-transformers.

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of dicts with keys:
          - chunk: str (original text)
          - dense_vector: list[float] (384 dims)
    """
    if not chunks:
        return []
    loop = asyncio.get_event_loop()
    dense_vecs = await loop.run_in_executor(
        None,
        lambda: _get_model().encode(chunks, batch_size=BATCH_SIZE, show_progress_bar=False),
    )
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "chunk": chunk,
            "dense_vector": dense_vecs[i].tolist(),
        })
    print(f"[text_embedder] Embedded {len(chunks)} chunks.")
    return results


async def embed_query(query: str) -> dict:
    """Embed a single query string for search.

    Args:
        query: The search query string.

    Returns:
        Dict with keys:
          - dense_vector: list[float]
    """
    results = await embed_chunks([query])
    return {
        "dense_vector": results[0]["dense_vector"],
    }