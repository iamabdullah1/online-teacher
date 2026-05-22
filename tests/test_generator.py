"""Tests for generator module using mocking."""

import pytest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

import app.generation.generator as gen
from app.generation.generator import (
    _format_context,
    _build_messages,
    generate_answer,
)

pytestmark = pytest.mark.asyncio


MOCK_RESULTS = [
    {
        "type": "text",
        "chunk": "Newton's second law states F = ma.",
        "page_num": 4,
        "source_pdf": "physics.pdf",
        "chunk_index": 0,
        "score": 0.95
    },
    {
        "type": "visual",
        "image_path": "/data/processed/images/page_0005.png",
        "page_num": 5,
        "source_pdf": "physics.pdf",
        "score": 0.88
    }
]


async def test_format_context_returns_string():
    """Test _format_context returns a string."""
    result = _format_context(MOCK_RESULTS)
    assert isinstance(result, str)
    assert len(result) > 0


async def test_format_context_includes_sources():
    """Test formatted context includes source info."""
    result = _format_context(MOCK_RESULTS)
    assert "Source 1" in result
    assert "physics.pdf" in result
    assert "Page 4" in result


async def test_format_context_handles_visual():
    """Test formatted context handles visual results."""
    result = _format_context(MOCK_RESULTS)
    assert "Visual Source" in result or "Diagram" in result


async def test_build_messages_structure():
    """Test messages have correct structure."""
    messages = _build_messages("What is F=ma?", "context")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


async def test_build_messages_includes_query():
    """Test user message includes the query."""
    messages = _build_messages("What is force?", "context")
    assert "What is force?" in messages[1]["content"]


async def test_generate_answer_empty_results():
    """Test empty results returns appropriate response."""
    result = await generate_answer("What is X?", [])
    assert result["status"] == "success"
    assert result["context_used"] == 0
    assert "could not find" in result["answer"].lower()


async def test_generate_answer_success():
    """Test successful answer generation with mocked API."""
    mock_response = SimpleNamespace(
        message=SimpleNamespace(
            content=[SimpleNamespace(text="Newton's second law is F=ma [Source 1].")]
        )
    )
    with patch.object(gen, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await generate_answer("What is F=ma?", MOCK_RESULTS)
        assert result["status"] == "success"
        assert len(result["answer"]) > 0


async def test_generate_answer_has_sources():
    """Test answer includes sources."""
    mock_response = SimpleNamespace(
        message=SimpleNamespace(
            content=[SimpleNamespace(text="Answer text.")]
        )
    )
    with patch.object(gen, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await generate_answer("What is F=ma?", MOCK_RESULTS)
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0
        assert "page_num" in result["sources"][0]


async def test_generate_answer_api_error():
    """Test API error handling."""
    with patch.object(gen, "_get_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=Exception("API error"))
        mock_get_client.return_value = mock_client

        result = await generate_answer("What?", MOCK_RESULTS)
        assert result["status"] == "error"
        assert result["error"] is not None


async def test_generate_answer_required_keys():
    """Test result has all required keys."""
    result = await generate_answer("test", [])
    keys = ["answer", "sources", "context_used", "status", "error"]
    assert all(k in result for k in keys)
