"""Tests for visual_embedder — mocks HTTP calls to ColPali service."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from PIL import Image

pytestmark = pytest.mark.asyncio

MOCK_EMBED_IMAGES_RESPONSE = {
    "results": [
        {
            "image_path": "/data/processed/images/page_0001.png",
            "vectors": [[0.1] * 128, [0.2] * 128, [0.3] * 128],
            "vector_count": 3
        }
    ]
}

MOCK_EMBED_QUERY_RESPONSE = {
    "vector": [0.1] * 128
}


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a real test PNG image."""
    image_path = tmp_path / "test_page.png"
    img = Image.new("RGB", (400, 600), color=(255, 255, 255))
    img.save(str(image_path))
    return str(image_path)


async def test_embed_images_returns_list():
    """embed_images returns a list of dicts."""
    from app.ingestion.visual_embedder import embed_images
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_EMBED_IMAGES_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__aenter__.return_value.post = \
            AsyncMock(return_value=mock_response)
        result = await embed_images(["/fake/path.png"])
        assert isinstance(result, list)
        assert len(result) == 1


async def test_embed_images_has_required_keys():
    """Each result has image_path, vectors, vector_count."""
    from app.ingestion.visual_embedder import embed_images
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_EMBED_IMAGES_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__aenter__.return_value.post = \
            AsyncMock(return_value=mock_response)
        result = await embed_images(["/fake/path.png"])
        assert "image_path" in result[0]
        assert "vectors" in result[0]
        assert "vector_count" in result[0]


async def test_embed_images_vectors_are_128_dims():
    """Each patch vector is 128 dimensions."""
    from app.ingestion.visual_embedder import embed_images
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_EMBED_IMAGES_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__aenter__.return_value.post = \
            AsyncMock(return_value=mock_response)
        result = await embed_images(["/fake/path.png"])
        assert len(result[0]["vectors"][0]) == 128


async def test_embed_images_vector_count_matches():
    """vector_count equals len(vectors)."""
    from app.ingestion.visual_embedder import embed_images
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_EMBED_IMAGES_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__aenter__.return_value.post = \
            AsyncMock(return_value=mock_response)
        result = await embed_images(["/fake/path.png"])
        assert result[0]["vector_count"] == len(result[0]["vectors"])


async def test_embed_images_empty_returns_empty():
    """Empty input returns empty list without calling service."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([])
    assert result == []


async def test_embed_images_service_unavailable_returns_stub():
    """Returns stub vectors when ColPali service is down."""
    from app.ingestion.visual_embedder import embed_images
    import httpx
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = \
            AsyncMock(side_effect=httpx.ConnectError("refused"))
        result = await embed_images(["/fake/path.png"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["vectors"] == [[0.0] * 128]


async def test_embed_query_returns_vector():
    """embed_query_image returns a 128-dim vector."""
    from app.ingestion.visual_embedder import embed_query_image
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_EMBED_QUERY_RESPONSE
        mock_response.raise_for_status = MagicMock()
        mock_client.return_value.__aenter__.return_value.post = \
            AsyncMock(return_value=mock_response)
        result = await embed_query_image("What is the diagram?")
        assert isinstance(result, list)
        assert len(result) == 128


async def test_embed_query_service_unavailable_returns_zeros():
    """Returns zero vector when ColPali service is down."""
    from app.ingestion.visual_embedder import embed_query_image
    import httpx
    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post = \
            AsyncMock(side_effect=httpx.ConnectError("refused"))
        result = await embed_query_image("test query")
        assert result == [0.0] * 128