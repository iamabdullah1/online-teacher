"""Tests for visual_indexer module."""
import pytest
import asyncio
import io
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import fitz
from PIL import Image as PILImage

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_pdf_with_image(tmp_path):
    """Create a real test PDF with an embedded image."""
    pdf_path = tmp_path / "test_figures.pdf"
    doc = fitz.open()
    page = doc.new_page()

    page.insert_text((50, 50),
        "Cell Biology: The cell membrane controls transport.",
        fontsize=12)

    img = PILImage.new("RGB", (300, 200), color=(100, 150, 200))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    rect = fitz.Rect(50, 100, 350, 300)
    page.insert_image(rect, stream=img_bytes.getvalue())

    page.insert_text((50, 320),
        "Figure 1.1: Cell membrane structure diagram.",
        fontsize=10)

    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


async def test_extract_figures_finds_image(sample_pdf_with_image):
    """Test that extract_figures_from_page finds embedded images."""
    from app.ingestion.visual_indexer import extract_figures_from_page
    figures = extract_figures_from_page(sample_pdf_with_image, 0)
    assert isinstance(figures, list)
    assert len(figures) >= 1
    assert "figure_path" in figures[0]
    assert Path(figures[0]["figure_path"]).exists()


async def test_extract_figures_returns_metadata(sample_pdf_with_image):
    """Test that extracted figures have correct metadata."""
    from app.ingestion.visual_indexer import extract_figures_from_page, MIN_FIGURE_WIDTH, MIN_FIGURE_HEIGHT
    figures = extract_figures_from_page(sample_pdf_with_image, 0)
    fig = figures[0]
    assert "page_num" in fig
    assert "width" in fig
    assert "height" in fig
    assert fig["width"] >= MIN_FIGURE_WIDTH
    assert fig["height"] >= MIN_FIGURE_HEIGHT


async def test_extract_figures_empty_page(tmp_path):
    """Test extraction from a page with no images."""
    from app.ingestion.visual_indexer import extract_figures_from_page

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Just text no images", fontsize=12)
    pdf_path = tmp_path / "text_only.pdf"
    doc.save(str(pdf_path))
    doc.close()

    figures = extract_figures_from_page(str(pdf_path), 0)
    assert isinstance(figures, list)


async def test_describe_from_page_text_returns_dict():
    """Test that _describe_from_page_text returns a dict with expected keys."""
    from app.ingestion.visual_indexer import _describe_from_page_text

    figure = {
        "figure_path": "/fake/path.png",
        "page_num": 1,
        "figure_index": 0,
        "width": 300,
        "height": 200
    }
    result = _describe_from_page_text(
        figure,
        page_text="The cell membrane is a phospholipid bilayer diagram.",
        chapter="Cell Biology",
        section_title="Membrane Structure"
    )
    assert "description" in result
    assert "keywords" in result
    assert "has_diagram" in result
    assert "has_table" in result
    assert "has_formula" in result
    assert "subject" in result
    assert isinstance(result["keywords"], list)
    assert "Cell Biology" in result["description"]
    assert "Membrane" in result["description"]


async def test_describe_from_page_text_no_text():
    """Test fallback when no page text is available."""
    from app.ingestion.visual_indexer import _describe_from_page_text

    figure = {"figure_path": "/fake/path.png", "page_num": 5}
    result = _describe_from_page_text(figure)
    assert "Figure on page 5" in result["description"]


async def test_index_pdf_figures_returns_list(sample_pdf_with_image):
    """Test that index_pdf_figures returns a list."""
    from app.ingestion.visual_indexer import index_pdf_figures

    # Note: index_pdf_figures now calls _describe_from_page_text
    # synchronously inside the loop, no async mocking needed.
    pages = [{
        "page_num": 0,
        "text": "Cell biology text about membrane diagram.",
        "chapter": "Cell Biology",
        "section_title": "Membrane Structure"
    }]
    results = await index_pdf_figures(sample_pdf_with_image, pages)
    assert isinstance(results, list)
    if results:
        assert "description" in results[0]
        assert "keywords" in results[0]


async def test_search_figures_returns_list():
    """Test that search_figures returns a list."""
    from app.ingestion.visual_indexer import search_figures

    with patch("app.ingestion.qdrant_client.search_figures_collection",
               new_callable=AsyncMock) as mock_search:
        with patch("app.ingestion.text_embedder.embed_query",
                   new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = {
                "dense_vector": [0.1] * 1024,
                "sparse_vector": {}
            }
            mock_search.return_value = [{
                "figure_path": "/fake/path.png",
                "score": 0.85,
                "description": "DNA diagram",
                "subject": "molecular biology"
            }]
            results = await search_figures(
                "DNA structure", "book.pdf", limit=3
            )
            assert isinstance(results, list)
