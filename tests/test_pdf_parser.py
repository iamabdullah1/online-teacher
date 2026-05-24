import pytest
import fitz
from pathlib import Path
from datetime import datetime

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a sample PDF for testing."""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    # Page 0: text heavy - longer text to get multiple chunks with overlap
    page = doc.new_page(width=800, height=1100)
    paragraphs = [
        "Introduction to algebraic expressions and variables forms the foundation "
        "for advanced mathematical thinking. Students learn to represent real "
        "world situations using symbols and equations. Practice with manipulation "
        "and simplification builds fluency in algebraic operations.",
        "Linear equations and graphing techniques help visualize relationships "
        "between variables. Understanding slope and intercept enables analysis "
        "of trends and predictions. Coordinate geometry connects algebra with "
        "spatial reasoning through visual representations.",
        "Quadratic functions and parabolas demonstrate exponential relationships "
        "in mathematical modeling. Vertex form reveals maximum and minimum values "
        "while standard form enables factorization. Applications include physics "
        "and engineering problems involving projectile motion.",
        "Polynomial operations include addition subtraction multiplication and "
        "division of expressions with multiple terms. Factoring techniques solve "
        "complex equations and reveal underlying structures. Remainder theorem "
        "provides insight into polynomial behavior.",
    ]
    # Repeat paragraphs multiple times to get enough text
    full_text = " ".join(paragraphs * 10)
    lines = []
    for i in range(0, len(full_text), 100):
        lines.append(full_text[i:i+100])
    for i, line in enumerate(lines[:50]):
        y_pos = 50 + (i * 12)
        page.insert_text((50, y_pos), line, fontsize=8)
    # Page 1: visual style (short text with visual keyword)
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Figure 1.1", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


async def test_returns_list_of_dicts(sample_pdf_path):
    """Test that extract_pages returns a list of dictionaries."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    assert isinstance(result, list)
    assert len(result) == 2
    for item in result:
        assert isinstance(item, dict)


async def test_dict_has_required_keys(sample_pdf_path):
    """Test that each dict has all required keys."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    required_keys = {"page_num", "text", "is_visual", "image_path", "chunks", "chunk_count"}
    for item in result:
        assert required_keys.issubset(item.keys())


async def test_text_page_has_chunks(sample_pdf_path):
    """Test that text pages have non-empty chunks."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    # First page should be text with chunks
    assert result[0]["is_visual"] is False
    assert len(result[0]["chunks"]) > 0
    assert result[0]["chunk_count"] == len(result[0]["chunks"])


async def test_visual_page_detection(sample_pdf_path):
    """Test that visual pages are detected correctly."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    # Second page should be visual
    assert result[1]["is_visual"] is True
    assert result[1]["chunks"] == []
    assert isinstance(result[1]["image_path"], str)


async def test_chunk_overlap(sample_pdf_path):
    """Test that consecutive chunks have proper overlap."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    chunks = result[0]["chunks"]
    if len(chunks) >= 2:
        chunk1_words = set(chunks[0].split())
        chunk2_words = set(chunks[1].split())
        overlap = len(chunk1_words & chunk2_words)
        assert overlap >= 40


async def test_invalid_path_raises_value_error():
    """Test that invalid path raises ValueError."""
    from app.ingestion.pdf_parser import extract_pages
    with pytest.raises(ValueError):
        await extract_pages("/nonexistent/path.pdf")


async def test_page_num_is_correct(sample_pdf_path):
    """Test that page numbers are correct."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    assert result[0]["page_num"] == 0
    assert result[1]["page_num"] == 1


async def test_page_has_metadata_keys(sample_pdf_path):
    """Test that each page dict has all metadata keys."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    metadata_keys = {"chapter", "section_title", "word_count", "ingested_at"}
    for item in result:
        assert metadata_keys.issubset(item.keys())


async def test_ingested_at_is_iso_format(sample_pdf_path):
    """Test that ingested_at is in ISO format."""
    from app.ingestion.pdf_parser import extract_pages
    result = await extract_pages(sample_pdf_path)
    datetime.fromisoformat(result[0]["ingested_at"])