import pytest
import time

pytestmark = pytest.mark.asyncio

import app.ingestion.qdrant_client as qc

TEST_TEXT_COLLECTION = f"test_text_{int(time.time())}"
TEST_VISUAL_COLLECTION = f"test_visual_{int(time.time())}"

qc.TEXT_COLLECTION = TEST_TEXT_COLLECTION
qc.VISUAL_COLLECTION = TEST_VISUAL_COLLECTION


@pytest.fixture(autouse=True)
def cleanup():
    """Delete test collections after each test."""
    yield
    client = qc._get_client()
    try:
        client.delete_collection(TEST_TEXT_COLLECTION)
    except Exception:
        pass
    try:
        client.delete_collection(TEST_VISUAL_COLLECTION)
    except Exception:
        pass


SAMPLE_TEXT_CHUNK = {
    "chunk": "Newton's second law: F = ma",
    "dense_vector": [0.1] * 1024,
    "sparse_vector": {101: 0.8, 202: 0.6, 303: 0.4},
    "page_num": 1,
    "source_pdf": "physics.pdf",
    "chunk_index": 0,
    "chapter": "Chapter 1",
    "section_title": "Newton's Laws",
    "word_count": 6,
    "ingested_at": "2026-01-01T00:00:00",
    "is_first_chunk": True
}

SAMPLE_VISUAL_PAGE = {
    "image_path": "/data/processed/images/page_0001.png",
    "vectors": [[0.1] * 128, [0.2] * 128],
    "vector_count": 2,
    "page_num": 1,
    "source_pdf": "physics.pdf",
}


async def test_init_collections_creates_collections():
    """Test that init_collections creates both collections."""
    await qc.init_collections()
    client = qc._get_client()
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    assert TEST_TEXT_COLLECTION in names
    assert TEST_VISUAL_COLLECTION in names


async def test_init_collections_idempotent():
    """Test that init_collections can be called twice without error."""
    await qc.init_collections()
    await qc.init_collections()


async def test_upsert_text_chunks_returns_count():
    """Test that upsert_text_chunks returns correct count."""
    await qc.init_collections()
    count = await qc.upsert_text_chunks([SAMPLE_TEXT_CHUNK])
    assert count == 1


async def test_upsert_visual_pages_returns_count():
    """Test that upsert_visual_pages returns correct count (one per patch)."""
    await qc.init_collections()
    count = await qc.upsert_visual_pages([SAMPLE_VISUAL_PAGE])
    assert count == 2


async def test_search_text_returns_results():
    """Test that search_text returns results with expected keys."""
    await qc.init_collections()
    await qc.upsert_text_chunks([SAMPLE_TEXT_CHUNK])
    results = await qc.search_text([0.1] * 1024, {101: 0.8}, limit=1)
    assert len(results) == 1
    assert "chunk" in results[0]
    assert "score" in results[0]


async def test_search_visual_returns_results():
    """Test that search_visual returns results."""
    await qc.init_collections()
    await qc.upsert_visual_pages([SAMPLE_VISUAL_PAGE])
    results = await qc.search_visual([0.1] * 128, limit=1)
    assert len(results) >= 1
    assert "image_path" in results[0]


async def test_search_text_result_keys():
    """Test that search_text results have all required keys."""
    await qc.init_collections()
    await qc.upsert_text_chunks([SAMPLE_TEXT_CHUNK])
    results = await qc.search_text([0.1] * 1024, {101: 0.8}, limit=1)
    result = results[0]
    assert "chunk" in result
    assert "page_num" in result
    assert "source_pdf" in result
    assert "chunk_index" in result
    assert "score" in result


async def test_empty_upsert_returns_zero():
    """Test that empty upsert returns zero."""
    await qc.init_collections()
    assert await qc.upsert_text_chunks([]) == 0
    assert await qc.upsert_visual_pages([]) == 0


async def test_search_text_returns_chapter_metadata():
    """Test that search_text returns chapter and section_title."""
    await qc.init_collections()
    await qc.upsert_text_chunks([SAMPLE_TEXT_CHUNK])
    results = await qc.search_text([0.1] * 1024, {101: 0.8}, limit=1)
    assert "chapter" in results[0]
    assert "section_title" in results[0]
    assert results[0]["chapter"] == "Chapter 1"