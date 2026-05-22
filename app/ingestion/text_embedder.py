import asyncio
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel

MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32

_model: BGEM3FlagModel | None = None


def _get_model() -> BGEM3FlagModel:
    """Load BGE-M3 model once and reuse. Downloads on first call."""
    global _model
    if _model is None:
        print(f"[text_embedder] Loading BGE-M3 model...")
        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)
        print(f"[text_embedder] Model loaded.")
    return _model


async def embed_chunks(chunks: list[str]) -> list[dict]:
    """Embed a list of text chunks using BGE-M3.

    Args:
        chunks: List of text strings to embed.

    Returns:
        List of dicts with keys:
          - chunk: str (original text)
          - dense_vector: list[float] (1024 dims)
          - sparse_vector: dict[int, float] (token_id -> weight)
    """
    if not chunks:
        return []
    loop = asyncio.get_event_loop()
    output = await loop.run_in_executor(
        None,
        lambda: _get_model().encode(
            chunks,
            batch_size=BATCH_SIZE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        ),
    )
    dense_vecs = output["dense_vecs"]
    lexical_weights = output["lexical_weights"]
    results = []
    for i, chunk in enumerate(chunks):
        sparse = {int(k): float(v) for k, v in lexical_weights[i].items()}
        results.append({
            "chunk": chunk,
            "dense_vector": dense_vecs[i].tolist(),
            "sparse_vector": sparse,
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
          - sparse_vector: dict[int, float]
    """
    results = await embed_chunks([query])
    return {
        "dense_vector": results[0]["dense_vector"],
        "sparse_vector": results[0]["sparse_vector"],
    }