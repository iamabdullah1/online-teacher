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


@pytest.fixture
def mock_visual_results():
    """Mock visual search results."""
    return [
        {
            "image_path": "/data/processed/images/page_0001.png",
            "score": 0.92,
        },
    ]


class TestRRFScore:
    """Tests for _rrf_score function."""

    def test_first_place_highest_score(self):
        """Test that rank 1 gets highest score."""
        from app.retrieval.retriever import _rrf_score
        score = _rrf_score(1)
        assert score > _rrf_score(2)
        assert score > _rrf_score(3)

    def test_default_k_constant(self):
        """Test default k=60 produces expected scores."""
        from app.retrieval.retriever import _rrf_score
        assert _rrf_score(1, k=60) == pytest.approx(1.0 / 61)
        assert _rrf_score(10, k=60) == pytest.approx(1.0 / 70)

    def test_custom_k_constant(self):
        """Test custom k changes scores appropriately."""
        from app.retrieval.retriever import _rrf_score
        assert _rrf_score(1, k=10) > _rrf_score(1, k=60)

    def test_monotonic_decrease(self):
        """Test scores decrease monotonically with rank."""
        from app.retrieval.retriever import _rrf_score
        prev = 1.0
        for rank in range(1, 11):
            score = _rrf_score(rank)
            assert score < prev
            prev = score


class TestRetrieve:
    """Tests for retrieve function."""

    @pytest.fixture
    def sample_dense(self):
        """Sample dense vector (1024-dim)."""
        return [0.1] * 1024

    @pytest.fixture
    def sample_sparse(self):
        """Sample sparse vector."""
        return {101: 0.8, 202: 0.6}

    @pytest.fixture
    def sample_visual(self):
        """Sample visual vector (128-dim)."""
        return [0.2] * 128

    async def test_returns_list(self, sample_dense, sample_sparse):
        """Test retrieve returns a list."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=[])
            from app.retrieval.retriever import retrieve
            result = await retrieve(
                "test query", sample_dense, sample_sparse
            )
            assert isinstance(result, list)

    async def test_text_only_when_no_visual(
        self, sample_dense, sample_sparse, mock_text_results
    ):
        """Test text search when no visual vector provided."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=mock_text_results)
            from app.retrieval.retriever import retrieve
            result = await retrieve(
                "force", sample_dense, sample_sparse, query_visual=None
            )
            assert len(result) == 2
            assert result[0]["type"] == "text"

    async def test_fuses_text_and_visual(
        self, sample_dense, sample_sparse, sample_visual,
        mock_text_results, mock_visual_results
    ):
        """Test fusion of text and visual results."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=mock_text_results)
            mock_qc.search_visual = AsyncMock(return_value=mock_visual_results)
            from app.retrieval.retriever import retrieve
            result = await retrieve(
                "force", sample_dense, sample_sparse,
                query_visual=sample_visual
            )
            types = [r["type"] for r in result]
            assert "text" in types
            assert "visual" in types

    async def test_respects_limits(self, sample_dense, sample_sparse):
        """Test text_limit and fusion_limit are respected."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            # Return many results
            many_text = [
                {"chunk": f"chunk {i}", "page_num": i, "source_pdf": "x.pdf",
                 "chunk_index": i, "score": 0.9 - i * 0.01}
                for i in range(20)
            ]
            mock_qc.search_text = AsyncMock(return_value=many_text)
            from app.retrieval.retriever import retrieve
            result = await retrieve(
                "test", sample_dense, sample_sparse,
                text_limit=20, fusion_limit=5
            )
            assert len(result) == 5

    async def test_result_has_required_keys(
        self, sample_dense, sample_sparse, mock_text_results
    ):
        """Test result items have required keys."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=mock_text_results)
            from app.retrieval.retriever import retrieve
            result = await retrieve(
                "test", sample_dense, sample_sparse
            )
            assert "type" in result[0]
            assert "score" in result[0]


class TestRetrieveTextOnly:
    """Tests for retrieve_text_only function."""

    @pytest.fixture
    def sample_dense(self):
        """Sample dense vector."""
        return [0.1] * 1024

    @pytest.fixture
    def sample_sparse(self):
        """Sample sparse vector."""
        return {101: 0.8}

    async def test_returns_text_results(
        self, sample_dense, sample_sparse, mock_text_results
    ):
        """Test text-only returns text results."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=mock_text_results)
            from app.retrieval.retriever import retrieve_text_only
            result = await retrieve_text_only(sample_dense, sample_sparse)
            assert len(result) == 2
            assert "chunk" in result[0]

    async def test_uses_limit_param(self, sample_dense, sample_sparse):
        """Test limit parameter is passed to search."""
        with patch("app.retrieval.retriever.qc") as mock_qc:
            mock_qc.init_collections = AsyncMock()
            mock_qc.search_text = AsyncMock(return_value=[])
            from app.retrieval.retriever import retrieve_text_only
            await retrieve_text_only(sample_dense, sample_sparse, limit=5)
            mock_qc.search_text.assert_called_once()
            call_args = mock_qc.search_text.call_args
            assert call_args[0][1] == {101: 0.8}
            assert call_args[1]["limit"] == 5