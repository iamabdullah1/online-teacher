"""Answer generation using Cohere API."""

import os
from typing import Any

import cohere
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
MODEL_NAME = "command-r-plus-08-2024"
MAX_TOKENS = 1024
TEMPERATURE = 0.2
MAX_CONTEXT_CHUNKS = 5

_client: cohere.AsyncClientV2 | None = None


def _get_client() -> cohere.AsyncClientV2:
    """Get or create async Cohere client."""
    global _client
    if _client is None:
        _client = cohere.AsyncClientV2(api_key=COHERE_API_KEY)
    return _client


def _format_context(results: list[dict[str, Any]]) -> str:
    """Format retrieved results into context string for LLM.

    Args:
        results: Fused results from retriever.retrieve()

    Returns:
        Formatted context string with source citations.
    """
    context_parts: list[str] = []
    for i, result in enumerate(results[:MAX_CONTEXT_CHUNKS]):
        result_type = result.get("result_type") or result.get("type", "text")
        page_num = result.get("page_num", 0)
        source_pdf = result.get("source_pdf", "unknown")
        if result_type == "text":
            chunk = result.get("chunk", "")
            chapter = result.get("chapter", "")
            section = result.get("section_title", "")
            location = f"Page {page_num}"
            if chapter:
                location = f"{chapter} | Page {page_num}"
            if section:
                location = f"{chapter} | {section} | Page {page_num}"
            context_parts.append(
                f"[Source {i+1} | {source_pdf} | {location}]\n{chunk}"
            )
        elif result_type == "visual":
            image_path = result.get("image_path", "")
            chapter = result.get("chapter", "")
            section = result.get("section_title", "")
            location = f"Page {page_num}"
            if chapter:
                location = f"{chapter} | Page {page_num}"
            if section:
                location = f"{chapter} | {section} | Page {page_num}"
            context_parts.append(
                f"[Visual Source {i+1} | {source_pdf} | {location}]\n"
                f"[Diagram/Figure at: {image_path}]"
            )
    return "\n\n".join(context_parts)


def _build_messages(query: str, context: str, history: list[dict] | None = None) -> list[dict[str, Any]]:
    """Build messages list for Cohere Chat API v2.

    Args:
        query: Student question string.
        context: Formatted context from _format_context()
        history: Previous conversation turns as {role, content} dicts.

    Returns:
        List of message dicts for Cohere API.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert teacher assistant. "
                "Use the conversation history to understand "
                "follow-up questions and references like 'it' or 'they'. "
                "Always ground your answer in the provided context "
                "from the textbook. "
                "Always cite sources using [Source N] notation. "
                "Be clear, accurate, and educational."
            )
        }
    ]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": (
            f"Context from textbook:\n\n{context}\n\n"
            f"Student question: {query}\n\n"
            f"Answer based only on the context above:"
        )
    })
    return messages


async def generate_answer(
    query: str,
    retrieved_results: list[dict[str, Any]],
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """Generate grounded answer from retrieved context via Cohere.

    Args:
        query: Student question string.
        retrieved_results: Results from retriever.retrieve()
        history: Previous conversation turns as {role, content} dicts.

    Returns:
        Dict with keys:
          - answer: str
          - sources: list[dict]
          - context_used: int
          - status: str ("success" or "error")
          - error: str | None
    """
    if not retrieved_results:
        return {
            "answer": "I could not find relevant information.",
            "sources": [],
            "context_used": 0,
            "status": "success",
            "error": None
        }

    # Skip Cohere call if all results are below relevance threshold
    max_score = max(r.get("score", 0) for r in retrieved_results)
    if max_score < 0.3:
        return {
            "answer": "I could not find enough information in the textbook to answer this question.",
            "sources": [{"page_num": r.get("page_num", 0), "source_pdf": r.get("source_pdf", "")}
                        for r in retrieved_results[:3]],
            "context_used": len(retrieved_results[:MAX_CONTEXT_CHUNKS]),
            "status": "success",
            "error": None
        }

    context = _format_context(retrieved_results)
    messages = _build_messages(query, context, history=history)
    sources = [
        {
            "index": i + 1,
            "page_num": r.get("page_num", 0),
            "source_pdf": r.get("source_pdf", ""),
            "result_type": r.get("result_type") or r.get("type", "text")
        }
        for i, r in enumerate(retrieved_results[:MAX_CONTEXT_CHUNKS])
    ]

    try:
        client = _get_client()
        response = await client.chat(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        answer = response.message.content[0].text
        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(retrieved_results[:MAX_CONTEXT_CHUNKS]),
            "status": "success",
            "error": None
        }
    except cohere.core.ApiError as e:
        return {
            "answer": "",
            "sources": [],
            "context_used": 0,
            "status": "error",
            "error": f"Cohere API error: {str(e)}"
        }
    except Exception as e:
        return {
            "answer": "",
            "sources": [],
            "context_used": 0,
            "status": "error",
            "error": str(e)
        }


async def generate_answer_stream(
    query: str,
    retrieved_results: list[dict[str, Any]],
    history: list[dict] | None = None,
):
    """Stream answer token by token via Cohere.

    Args:
        query: Student question string.
        retrieved_results: Results from retriever.retrieve()
        history: Previous conversation turns as {role, content} dicts.

    Yields:
        str chunks of the answer as they stream in.
    """
    context = _format_context(retrieved_results)
    messages = _build_messages(query, context, history=history)
    client = _get_client()
    async for event in client.chat_stream(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    ):
        if hasattr(event, "delta") and event.delta:
            if hasattr(event.delta, "message"):
                content = event.delta.message.content
                if content and hasattr(content, "text"):
                    yield content.text
