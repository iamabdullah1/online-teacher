import os
import pytest

pytestmark = pytest.mark.asyncio

from unittest.mock import patch, AsyncMock, MagicMock

import app.ingestion.qdrant_client as qc


@pytest.fixture(autouse=True)
def reset_qdrant_client():
    """Reset the Qdrant client singleton for each test."""
    qc._client = None


SAMPLE_TEXT_CHUNK = {
    "chunk": "Newton's second law: F = ma",
    "dense_vector": [0.1] * 384,
    "page_num": 1,
    "source_pdf": "test_delete.pdf",
    "chunk_index": 0,
    "chapter": "Chapter 1",
    "section_title": "Newton's Laws",
    "word_count": 6,
    "ingested_at": "2026-01-01T00:00:00",
    "is_first_chunk": True,
}

SAMPLE_FIGURE = {
    "figure_path": "/data/processed/figures/page_0001_fig_00.png",
    "description": "A diagram of a pulley system",
    "keywords": ["pulley", "force", "mechanics"],
    "subject": "physics",
    "has_diagram": True,
    "has_table": False,
    "has_formula": False,
    "source_pdf": "test_delete.pdf",
    "page_num": 1,
    "figure_index": 0,
    "figure_filename": "page_0001_fig_00.png",
    "dense_vector": [0.1] * 384,
    "doc_id": "abc123",
    "ingested_at": "2026-01-01T00:00:00",
}


async def test_delete_pdf_data_returns_success():
    """Test delete_pdf_data returns success."""
    with (
        patch.object(qc, "_get_client") as mock_get,
    ):
        mock_client = MagicMock()
        mock_get.return_value = mock_client

        result = await qc.delete_pdf_data("test_delete.pdf")
        assert result["status"] == "success"
        assert result["text_chunks_deleted"] is True
        assert result["figures_deleted"] is True


async def test_delete_pdf_data_calls_delete_on_text_collection():
    """Test delete_pdf_data calls delete on the text_chunks collection."""
    with patch.object(qc, "_get_client") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client
        mock_client.delete.return_value = MagicMock()

        await qc.delete_pdf_data("test_delete.pdf")

        text_calls = [
            call
            for call in mock_client.delete.call_args_list
            if call[1].get("collection_name") == qc.TEXT_COLLECTION
        ]
        assert len(text_calls) == 1
        selector = text_calls[0][1]["points_selector"]
        assert selector.filter.must[0].key == "source_pdf"
        assert selector.filter.must[0].match.value == "test_delete.pdf"


async def test_delete_pdf_data_calls_delete_on_figures_collection():
    """Test delete_pdf_data calls delete on the visual_index collection."""
    with patch.object(qc, "_get_client") as mock_get:
        mock_client = MagicMock()
        mock_get.return_value = mock_client
        mock_client.delete.return_value = MagicMock()

        await qc.delete_pdf_data("test_delete.pdf")

        fig_calls = [
            call
            for call in mock_client.delete.call_args_list
            if call[1].get("collection_name") == qc.FIGURES_COLLECTION
        ]
        assert len(fig_calls) == 1


from fastapi.testclient import TestClient
from app.main import app

test_client = TestClient(app)


def test_delete_endpoint_returns_success():
    """Test DELETE /upload/{filename} returns success."""
    with (
        patch("app.api.routes.Path.exists") as mock_exists,
        patch("app.api.routes.Path.unlink"),
        patch("app.ingestion.qdrant_client.delete_pdf_data", new_callable=AsyncMock) as mock_delete,
        patch("app.api.routes.Path.glob", return_value=[]),
    ):
        mock_exists.return_value = True
        mock_delete.return_value = {
            "text_chunks_deleted": True,
            "figures_deleted": True,
            "status": "success",
        }

        response = test_client.delete("/api/v1/upload/test.pdf")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["source_pdf"] == "test.pdf"


def test_delete_nonexistent_pdf_returns_404():
    """Test DELETE /upload/{filename} returns 404 for missing PDF."""
    with patch("app.api.routes.Path.exists") as mock_exists:
        mock_exists.return_value = False
        response = test_client.delete("/api/v1/upload/nonexistent.pdf")
        assert response.status_code == 404


def test_delete_invalid_filename_returns_400():
    """Test DELETE rejects filenames with path traversal."""
    response = test_client.delete("/api/v1/upload/invalid%5Cname.pdf")
    assert response.status_code == 400


def test_list_uploads_returns_pdfs():
    """Test GET /uploads returns list of PDFs."""
    mock_pdf = MagicMock()
    mock_pdf.name = "physics.pdf"
    mock_pdf.stat.return_value.st_size = 5 * 1024 * 1024

    mock_dir = MagicMock()
    mock_dir.glob.return_value = [mock_pdf]

    with (
        patch("app.api.routes.Path") as mock_path,
        patch("app.ingestion.qdrant_client._get_client") as mock_get_client,
    ):
        mock_path.return_value = mock_dir
        mock_path.return_value.mkdir.return_value = None

        mock_client = MagicMock()
        mock_client.count.return_value.count = 42
        mock_get_client.return_value = mock_client

        response = test_client.get("/api/v1/uploads")
        assert response.status_code == 200
        data = response.json()
        assert "pdfs" in data
        assert isinstance(data["pdfs"], list)
        assert len(data["pdfs"]) == 1
        assert data["pdfs"][0]["filename"] == "physics.pdf"
        assert data["pdfs"][0]["ingested"] is True
