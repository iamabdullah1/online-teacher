import pytest
import fitz
from pathlib import Path

pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a real minimal PDF for testing."""
    pdf_path = tmp_path / "test_physics.pdf"
    doc = fitz.open()
    # Text page with adequate content (larger page to fit more text)
    page = doc.new_page(width=800, height=1100)
    for i in range(30):
        y_pos = 50 + (i * 12)
        page.insert_text(
            (50, y_pos),
            "Newton's second law of motion states that force equals mass times acceleration.",
            fontsize=8,
        )
    # Visual-style page (short text with figure keyword)
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Figure 1: Physics diagram", fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


async def test_ingest_pdf_returns_dict(sample_pdf):
    """Test that ingest_pdf returns a dict."""
    from app.ingestion.ingestion_pipeline import ingest_pdf
    result = await ingest_pdf(sample_pdf)
    assert isinstance(result, dict)


async def test_ingest_pdf_success_status(sample_pdf):
    """Test that successful ingest returns status='success'."""
    from app.ingestion.ingestion_pipeline import ingest_pdf
    result = await ingest_pdf(sample_pdf)
    assert result["status"] == "success"


async def test_ingest_pdf_counts_pages(sample_pdf):
    """Test that page counts are correct."""
    from app.ingestion.ingestion_pipeline import ingest_pdf
    result = await ingest_pdf(sample_pdf)
    assert result["total_pages"] == 2
    assert result["text_pages"] >= 1


async def test_ingest_pdf_keys_present(sample_pdf):
    """Test that result has all required keys."""
    from app.ingestion.ingestion_pipeline import ingest_pdf
    result = await ingest_pdf(sample_pdf)
    required = [
        "pdf_path", "total_pages", "text_pages", "visual_pages",
        "text_chunks_stored", "visual_vectors_stored",
        "status", "error"
    ]
    for key in required:
        assert key in result


async def test_ingest_pdf_nonexistent_file():
    """Test that nonexistent file returns error status."""
    from app.ingestion.ingestion_pipeline import ingest_pdf
    result = await ingest_pdf("/nonexistent/file.pdf")
    assert result["status"] == "error"
    assert result["error"] is not None


async def test_ingest_directory_no_pdfs(tmp_path):
    """Test that empty directory returns empty list."""
    from app.ingestion.ingestion_pipeline import ingest_directory
    result = await ingest_directory(str(tmp_path))
    assert result == []


async def test_ingest_directory_not_found():
    """Test that nonexistent directory raises ValueError."""
    from app.ingestion.ingestion_pipeline import ingest_directory
    with pytest.raises(ValueError):
        await ingest_directory("/nonexistent/directory")