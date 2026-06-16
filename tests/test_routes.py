"""Tests for API routes."""

from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from app.main import app

client = TestClient(app)

MOCK_INGEST_RESULT = {
    "pdf_path": "data/uploads/test.pdf",
    "total_pages": 5,
    "text_pages": 4,
    "visual_pages": 1,
    "text_chunks_stored": 12,
    "visual_vectors_stored": 8,
    "status": "success",
    "error": None
}

MOCK_ANSWER_RESULT = {
    "answer": "Newton's second law states F=ma [Source 1].",
    "sources": [{"index": 1, "page_num": 4,
                 "source_pdf": "physics.pdf",
                 "result_type": "text"}],
    "context_used": 1,
    "status": "success",
    "error": None
}


def test_root_endpoint():
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Online Teacher" in response.json()["message"]


def test_health_endpoint():
    """Test health endpoint returns status."""
    mock_collections = MagicMock()
    mock_collections.collections = []
    with patch("app.api.routes._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = mock_collections
        mock_get.return_value = mock_client

        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] in ["ok", "degraded"]


def test_upload_valid_pdf():
    """Test upload accepts valid PDF."""
    with patch("app.api.routes.ingest_pdf", new_callable=AsyncMock) as mock_ingest:
        mock_ingest.return_value = MOCK_INGEST_RESULT
        pdf_content = b"%PDF-1.4 fake pdf content"
        response = client.post(
            "/api/v1/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"


def test_upload_invalid_file_type():
    """Test upload rejects non-PDF files."""
    response = client.post(
        "/api/v1/upload",
        files={"file": ("test.txt", b"text content", "text/plain")}
    )
    assert response.status_code == 400


def test_query_endpoint():
    """Test query endpoint returns answer."""
    with patch("app.ingestion.text_embedder.embed_query", new_callable=AsyncMock) as mock_embed, \
         patch("app.retrieval.retriever.retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.generation.generator.generate_answer", new_callable=AsyncMock) as mock_gen:

        mock_embed.return_value = {
            "dense_vector": [0.1] * 384,
        }
        mock_retrieve.return_value = []
        mock_gen.return_value = MOCK_ANSWER_RESULT

        response = client.post(
            "/api/v1/query",
            json={"question": "What is Newton's law?"}
        )
        assert response.status_code == 200
        assert "answer" in response.json()


def test_query_empty_question():
    """Test query rejects empty question."""
    response = client.post(
        "/api/v1/query",
        json={"question": "   "}
    )
    assert response.status_code == 400


def test_query_response_has_required_keys():
    """Test query response has all required keys."""
    with patch("app.ingestion.text_embedder.embed_query", new_callable=AsyncMock) as mock_embed, \
         patch("app.retrieval.retriever.retrieve", new_callable=AsyncMock) as mock_retrieve, \
         patch("app.generation.generator.generate_answer", new_callable=AsyncMock) as mock_gen:

        mock_embed.return_value = {
            "dense_vector": [0.1] * 384,
        }
        mock_retrieve.return_value = []
        mock_gen.return_value = MOCK_ANSWER_RESULT

        response = client.post(
            "/api/v1/query",
            json={"question": "What is force?"}
        )
        data = response.json()
        keys = ["answer", "sources", "context_used", "status", "error"]
        assert all(k in data for k in keys)


def test_collections_endpoint():
    """Test collections endpoint returns Qdrant collections."""
    mock_collections = MagicMock()
    mock_collections.collections = []
    with patch("app.api.routes._get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = mock_collections
        mock_get.return_value = mock_client

        response = client.get("/api/v1/collections")
        assert response.status_code == 200
        assert "collections" in response.json()
