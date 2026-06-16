"""Tests for retriever module using mocking."""

import pytest
from unittest.mock import AsyncMock, patch

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_text_results():
    """Mock text search results."""
    return [
        {
            "chunk": "Newton's second law: F = ma",
            "page_num": 1,
            "source_pdf": "physics.pdf",
            "chunk_index": 0,
            "score": 0.95,
        },
        {
            "chunk": "Force equals mass times acceleration",
            "page_num": 2,
            "source_pdf": "physics.pdf",
            "chunk_index": 1,
            "score": 0.88,
        },
    ]


class TestRetrieve:
    """Tests for retrieve function."""

    @pytest.fixture
    def sample_dense(self):
        """Sample dense vector (384-dim)."""
        return [0.1] * 384

    async def test_returns_list(self, sample_dense):
        """Test retrieve returns a list."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=[])
            from app.retrieval.retriever import retrieve
            result = await retrieve(sample_dense)
            assert isinstance(result, list)

    async def test_returns_text_results(
        self, sample_dense, mock_text_results
    ):
        """Test retrieve returns text results."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=mock_text_results)
            from app.retrieval.retriever import retrieve
            result = await retrieve(sample_dense)
            assert len(result) == 2

    async def test_respects_limit(self, sample_dense):
        """Test limit param is respected."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            many_text = [
                {"chunk": f"chunk {i}", "page_num": i, "source_pdf": "x.pdf",
                 "chunk_index": i, "score": 0.9 - i * 0.01}
                for i in range(20)
            ]
            mock_qc.search_text = AsyncMock(return_value=many_text[:5])
            from app.retrieval.retriever import retrieve
            result = await retrieve(sample_dense, limit=5)
            assert len(result) == 5

    async def test_result_has_required_keys(
        self, sample_dense, mock_text_results
    ):
        """Test result items have required keys."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=mock_text_results)
            from app.retrieval.retriever import retrieve
            result = await retrieve(sample_dense)
            assert "chunk" in result[0]
            assert "score" in result[0]


class TestRetrieveTextOnly:
    """Tests for retrieve_text_only function."""

    @pytest.fixture
    def sample_dense(self):
        """Sample dense vector."""
        return [0.1] * 384

    async def test_returns_text_results(
        self, sample_dense, mock_text_results
    ):
        """Test text-only returns text results."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=mock_text_results)
            from app.retrieval.retriever import retrieve_text_only
            result = await retrieve_text_only(sample_dense)
            assert len(result) == 2
            assert "chunk" in result[0]

    async def test_uses_limit_param(self, sample_dense):
        """Test limit parameter is passed to search."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=[])
            from app.retrieval.retriever import retrieve_text_only
            await retrieve_text_only(sample_dense, limit=5)
            mock_qc.search_text.assert_called_once()
            call_args = mock_qc.search_text.call_args
            assert call_args[1]["limit"] == 5
