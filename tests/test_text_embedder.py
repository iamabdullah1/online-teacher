import pytest

pytestmark = pytest.mark.asyncio

SAMPLE_CHUNKS = [
    "Newton's second law states that force equals mass times acceleration.",
    "The centripetal force acts toward the center of circular motion.",
    "Energy is conserved in an isolated system according to thermodynamics.",
]


async def test_embed_chunks_returns_list():
    """Test that embed_chunks returns a list."""
    from app.ingestion.text_embedder import embed_chunks
    result = await embed_chunks(SAMPLE_CHUNKS)
    assert isinstance(result, list)
    assert len(result) == 3


async def test_embed_chunks_has_required_keys():
    """Test that each result has required keys."""
    from app.ingestion.text_embedder import embed_chunks
    result = await embed_chunks(SAMPLE_CHUNKS)
    for item in result:
        assert "chunk" in item
        assert "dense_vector" in item


async def test_dense_vector_dimensions():
    """Test that dense vectors have 384 dimensions."""
    from app.ingestion.text_embedder import embed_chunks
    result = await embed_chunks(SAMPLE_CHUNKS)
    assert len(result[0]["dense_vector"]) == 384


async def test_chunk_text_preserved():
    """Test that original chunk text is preserved."""
    from app.ingestion.text_embedder import embed_chunks
    result = await embed_chunks(SAMPLE_CHUNKS)
    assert result[0]["chunk"] == SAMPLE_CHUNKS[0]
    assert result[2]["chunk"] == SAMPLE_CHUNKS[2]


async def test_empty_chunks_returns_empty():
    """Test that empty input returns empty list."""
    from app.ingestion.text_embedder import embed_chunks
    result = await embed_chunks([])
    assert result == []


async def test_embed_query_returns_dict():
    """Test that embed_query returns correct dict structure."""
    from app.ingestion.text_embedder import embed_query
    result = await embed_query("What is Newton's second law?")
    assert "dense_vector" in result
    assert len(result["dense_vector"]) == 384


async def test_different_chunks_have_different_vectors():
    """Test that different chunks produce different vectors."""
    from app.ingestion.text_embedder import embed_chunks
    result = await embed_chunks(SAMPLE_CHUNKS)
    vec1 = result[0]["dense_vector"]
    vec2 = result[1]["dense_vector"]
    assert vec1 != vec2
