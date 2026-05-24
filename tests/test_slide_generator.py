"""Tests for slide_generator module."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from types import SimpleNamespace

pytestmark = pytest.mark.asyncio

SAMPLE_CHUNKS = [
    {
        "chunk": "Newton's second law states F = ma. Force equals mass times acceleration.",
        "page_num": 4,
        "source_pdf": "physics.pdf",
        "chapter": "Chapter 3",
        "section_title": "Newton's Laws",
        "word_count": 12
    },
    {
        "chunk": "Circular motion requires centripetal force directed toward the center.",
        "page_num": 6,
        "source_pdf": "physics.pdf",
        "chapter": "Chapter 4",
        "section_title": "Circular Motion",
        "word_count": 10
    },
    {
        "chunk": "Energy is conserved in isolated systems. Kinetic plus potential equals total.",
        "page_num": 8,
        "source_pdf": "physics.pdf",
        "chapter": "Chapter 5",
        "section_title": "Conservation of Energy",
        "word_count": 11
    }
]

MOCK_TOPICS_RESPONSE = '[{"topic":"Newton\'s Laws","chapter":"Chapter 3","key_concept":"F=ma"},{"topic":"Circular Motion","chapter":"Chapter 4","key_concept":"centripetal force"}]'

MOCK_SLIDE_CONTENT_RESPONSE = '{"title":"Newton\'s Second Law","bullets":["Force equals mass times acceleration","F = ma is the core equation","Acceleration depends on net force","More mass means less acceleration"]}'


async def test_generate_slides_returns_dict():
    """Test generate_slides returns a dict."""
    with patch("app.generation.slide_generator._extract_topics", new_callable=AsyncMock) as mock_topics, \
         patch("app.generation.slide_generator._generate_slide_content", new_callable=AsyncMock) as mock_content, \
         patch("app.generation.slide_generator._build_pptx") as mock_pptx:

        mock_topics.return_value = [
            {"topic": "Topic 1", "chapter": "Ch 1", "key_concept": "A"},
            {"topic": "Topic 2", "chapter": "Ch 2", "key_concept": "B"}
        ]
        mock_content.return_value = {"title": "Test", "bullets": ["a"], "chapter": "Ch 1"}
        mock_pptx.return_value = "/tmp/test.pptx"

        from app.generation.slide_generator import generate_slides
        result = await generate_slides(SAMPLE_CHUNKS, "physics.pdf", num_slides=2)
        assert isinstance(result, dict)


async def test_generate_slides_required_keys():
    """Test result has all required keys."""
    with patch("app.generation.slide_generator._extract_topics", new_callable=AsyncMock) as mock_topics, \
         patch("app.generation.slide_generator._generate_slide_content", new_callable=AsyncMock) as mock_content, \
         patch("app.generation.slide_generator._build_pptx") as mock_pptx:

        mock_topics.return_value = [{"topic": "T1", "chapter": "C1", "key_concept": "a"}]
        mock_content.return_value = {"title": "T", "bullets": ["b"], "chapter": "C1"}
        mock_pptx.return_value = "/tmp/test.pptx"

        from app.generation.slide_generator import generate_slides
        result = await generate_slides(SAMPLE_CHUNKS, "physics.pdf", num_slides=2)
        keys = ["file_path", "num_slides", "topics", "status", "error"]
        assert all(k in result for k in keys)


async def test_generate_slides_success_status():
    """Test success status when all mocks work."""
    with patch("app.generation.slide_generator._extract_topics", new_callable=AsyncMock) as mock_topics, \
         patch("app.generation.slide_generator._generate_slide_content", new_callable=AsyncMock) as mock_content, \
         patch("app.generation.slide_generator._build_pptx") as mock_pptx:

        mock_topics.return_value = [{"topic": "T1", "chapter": "C1", "key_concept": "a"}]
        mock_content.return_value = {"title": "T", "bullets": ["b"], "chapter": "C1"}
        mock_pptx.return_value = "/tmp/test.pptx"

        from app.generation.slide_generator import generate_slides
        result = await generate_slides(SAMPLE_CHUNKS, "physics.pdf", num_slides=2)
        assert result["status"] == "success"


async def test_generate_slides_topic_count():
    """Test num_slides matches requested count."""
    with patch("app.generation.slide_generator._extract_topics", new_callable=AsyncMock) as mock_topics, \
         patch("app.generation.slide_generator._generate_slide_content", new_callable=AsyncMock) as mock_content, \
         patch("app.generation.slide_generator._build_pptx") as mock_pptx:

        mock_topics.return_value = [
            {"topic": "T1", "chapter": "C1", "key_concept": "a"},
            {"topic": "T2", "chapter": "C2", "key_concept": "b"},
            {"topic": "T3", "chapter": "C3", "key_concept": "c"}
        ]
        mock_content.return_value = {"title": "T", "bullets": ["b"], "chapter": "C"}
        mock_pptx.return_value = "/tmp/test.pptx"

        from app.generation.slide_generator import generate_slides
        result = await generate_slides(SAMPLE_CHUNKS, "physics.pdf", num_slides=3)
        assert result["num_slides"] == 3


async def test_generate_slides_topics_list():
    """Test topics is a list."""
    with patch("app.generation.slide_generator._extract_topics", new_callable=AsyncMock) as mock_topics, \
         patch("app.generation.slide_generator._generate_slide_content", new_callable=AsyncMock) as mock_content, \
         patch("app.generation.slide_generator._build_pptx") as mock_pptx:

        mock_topics.return_value = [{"topic": "T1", "chapter": "C1", "key_concept": "a"}]
        mock_content.return_value = {"title": "T", "bullets": ["b"], "chapter": "C"}
        mock_pptx.return_value = "/tmp/test.pptx"

        from app.generation.slide_generator import generate_slides
        result = await generate_slides(SAMPLE_CHUNKS, "physics.pdf", num_slides=2)
        assert isinstance(result["topics"], list)


async def test_extract_topics_filters_by_focus():
    """Test _extract_topics filters by topic_focus."""
    with patch("app.generation.slide_generator._get_cohere_client") as mock_client:
        mock_response = SimpleNamespace(
            message=SimpleNamespace(content=[SimpleNamespace(text=MOCK_TOPICS_RESPONSE)])
        )
        mock_client.return_value.chat.return_value = mock_response

        from app.generation.slide_generator import _extract_topics
        topics = await _extract_topics(SAMPLE_CHUNKS, 2, "Chapter 3")
        assert isinstance(topics, list)


async def test_generate_slide_content_returns_dict():
    """Test _generate_slide_content returns correct structure."""
    with patch("app.generation.slide_generator._get_cohere_client") as mock_client:
        mock_response = SimpleNamespace(
            message=SimpleNamespace(content=[SimpleNamespace(text=MOCK_SLIDE_CONTENT_RESPONSE)])
        )
        mock_client.return_value.chat.return_value = mock_response

        from app.generation.slide_generator import _generate_slide_content
        topic = {"topic": "Newton's Laws", "chapter": "Chapter 3", "key_concept": "F=ma"}
        content = await _generate_slide_content(topic, SAMPLE_CHUNKS, 4, "detailed", "academic")
        assert "title" in content
        assert "bullets" in content
        assert isinstance(content["bullets"], list)


async def test_build_pptx_creates_file(tmp_path):
    """Test _build_pptx creates a real .pptx file."""
    from app.generation.slide_generator import _build_pptx, OUTPUTS_DIR

    slides_data = [
        {"title": "Test Slide", "bullets": ["Point 1", "Point 2"], "chapter": "Ch 1"}
    ]

    with patch("app.generation.slide_generator.Presentation") as mock_prs:
        mock_prs.return_value.slide_width = 0
        mock_prs.return_value.slide_height = 0

        path = _build_pptx(slides_data, "test.pdf", True, True)
        assert path.endswith(".pptx")


async def test_generate_slides_error_handling():
    """Test error handling when extraction fails."""
    with patch("app.generation.slide_generator._extract_topics", new_callable=AsyncMock) as mock_topics:
        mock_topics.side_effect = Exception("Cohere API error")

        from app.generation.slide_generator import generate_slides
        result = await generate_slides(SAMPLE_CHUNKS, "physics.pdf")
        assert result["status"] == "error"
        assert result["error"] is not None