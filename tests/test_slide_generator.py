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


async def test_find_best_figure_for_slide_success(sample_image):
    with patch("app.generation.slide_generator.search_figures", new_callable=AsyncMock) as ms:
        ms.return_value = [{"figure_filename": Path(sample_image).name, "source_pdf": "physics.pdf", "score": 0.9}]
        with patch("pathlib.Path.exists", return_value=True):
            from app.generation.slide_generator import _find_best_figure_for_slide
            r = await _find_best_figure_for_slide("Test", "Ch1", "physics.pdf")
            assert r == str(Path("data/processed/figures") / Path(sample_image).name)


async def test_find_best_figure_for_slide_no_results():
    with patch("app.generation.slide_generator.search_figures", new_callable=AsyncMock) as ms:
        ms.return_value = []
        from app.generation.slide_generator import _find_best_figure_for_slide
        r = await _find_best_figure_for_slide("X", "", "p.pdf")
        assert r is None


async def test_find_best_figure_for_slide_below_threshold(tmp_path):
    from PIL import Image
    img = tmp_path / "fig.png"
    Image.new("RGB", (100, 100)).save(str(img))
    with patch("app.generation.slide_generator.search_figures", new_callable=AsyncMock) as ms:
        ms.return_value = [{"figure_path": str(img), "score": 0.5}]
        from app.generation.slide_generator import _find_best_figure_for_slide
        r = await _find_best_figure_for_slide("X", "Ch1", "p.pdf")
        assert r is None
