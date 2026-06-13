"""Tests for slide_generator module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from types import SimpleNamespace

pytestmark = pytest.mark.asyncio

SAMPLE_CHUNKS = [
    {
        "chunk": "Newton's second law states F = ma.",
        "page_num": 4,
        "source_pdf": "physics.pdf",
        "chapter": "Chapter 3",
        "section_title": "Newton's Laws",
        "word_count": 12
    }
]

MOCK_TOPICS_RESPONSE = '[{"topic":"Test","chapter":"Ch1"}]'


@pytest.fixture
def sample_image(tmp_path):
    from PIL import Image
    p = tmp_path / "page.png"
    Image.new("RGB", (100,100)).save(str(p))
    return str(p)


async def test_find_figures_for_slides_success(sample_image):
    """Test that _find_figures_for_slides returns matched paths."""
    slides = [{"title": "Test", "chapter": "Ch1"}]
    with patch("app.generation.slide_generator.embed_chunks", new_callable=AsyncMock) as me:
        me.return_value = [{"dense_vector": [0.1]*1024}]
        with patch("app.generation.slide_generator.search_figures_collection", new_callable=AsyncMock) as ms:
            ms.return_value = [{"figure_filename": Path(sample_image).name, "score": 0.9}]
            with patch("pathlib.Path.exists", return_value=True):
                from app.generation.slide_generator import _find_figures_for_slides
                r = await _find_figures_for_slides(slides, "physics.pdf")
                assert len(r) == 1
                assert r[0] is not None
                assert Path(sample_image).name in r[0]


async def test_find_figures_for_slides_no_results():
    """Test that _find_figures_for_slides returns None for no matches."""
    slides = [{"title": "X", "chapter": "Ch1"}]
    with patch("app.generation.slide_generator.embed_chunks", new_callable=AsyncMock) as me:
        me.return_value = [{"dense_vector": [0.1]*1024}]
        with patch("app.generation.slide_generator.search_figures_collection", new_callable=AsyncMock) as ms:
            ms.return_value = []
            from app.generation.slide_generator import _find_figures_for_slides
            r = await _find_figures_for_slides(slides, "p.pdf")
            assert r == [None]


async def test_find_figures_for_slides_below_threshold():
    """Test that _find_figures_for_slides returns None below threshold."""
    slides = [{"title": "X", "chapter": "Ch1"}]
    with patch("app.generation.slide_generator.embed_chunks", new_callable=AsyncMock) as me:
        me.return_value = [{"dense_vector": [0.1]*1024}]
        with patch("app.generation.slide_generator.search_figures_collection", new_callable=AsyncMock) as ms:
            ms.return_value = [{"figure_filename": "fig.png", "score": 0.1}]
            from app.generation.slide_generator import _find_figures_for_slides
            r = await _find_figures_for_slides(slides, "p.pdf")
            assert r == [None]
