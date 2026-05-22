import pytest
from PIL import Image, ImageDraw

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_image_path(tmp_path):
    """Creates a real white PNG image for testing."""
    image_path = tmp_path / "test_page.png"
    img = Image.new("RGB", (400, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 50), "Figure 1.1: Test Diagram", fill=(0, 0, 0))
    draw.rectangle([100, 100, 300, 400], outline=(0, 0, 0), width=2)
    img.save(str(image_path))
    return str(image_path)


async def test_embed_images_returns_list(sample_image_path):
    """Test that embed_images returns a list."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([sample_image_path])
    assert isinstance(result, list)
    assert len(result) == 1


async def test_embed_images_has_required_keys(sample_image_path):
    """Test that result dict has required keys."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([sample_image_path])
    assert "image_path" in result[0]
    assert "vectors" in result[0]
    assert "vector_count" in result[0]


async def test_vectors_is_list_of_lists(sample_image_path):
    """Test that vectors is a list of lists."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([sample_image_path])
    assert isinstance(result[0]["vectors"], list)
    assert isinstance(result[0]["vectors"][0], list)


async def test_patch_vector_dimensions(sample_image_path):
    """Test that each patch vector has 128 dimensions (ColPali)."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([sample_image_path])
    assert len(result[0]["vectors"][0]) == 128


async def test_vector_count_matches_vectors(sample_image_path):
    """Test that vector_count matches actual vector count."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([sample_image_path])
    assert result[0]["vector_count"] == len(result[0]["vectors"])


async def test_empty_list_returns_empty():
    """Test that empty input returns empty list."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([])
    assert result == []


async def test_embed_query_image_returns_vector():
    """Test that embed_query_image returns a 128-dim vector (ColPali)."""
    from app.ingestion.visual_embedder import embed_query_image
    result = await embed_query_image("What does the diagram show?")
    assert isinstance(result, list)
    assert len(result) == 128


async def test_image_path_preserved(sample_image_path):
    """Test that original image path is preserved."""
    from app.ingestion.visual_embedder import embed_images
    result = await embed_images([sample_image_path])
    assert result[0]["image_path"] == sample_image_path