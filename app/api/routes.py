"""FastAPI routes for the Online Teacher Platform."""

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.ingestion.ingestion_pipeline import ingest_pdf
from app.retrieval.retriever import retrieve, retrieve_text_only
from app.generation.generator import generate_answer, generate_answer_stream
from app.ingestion.qdrant_client import _get_client

UPLOADS_DIR = Path("data/uploads")
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE_MB = 50

router = APIRouter()


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str
    text_only: bool = False
    limit: int = 5


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: list[dict[str, Any]]
    context_used: int
    status: str
    error: str | None


class IngestResponse(BaseModel):
    """Response model for upload endpoint."""
    pdf_path: str
    total_pages: int
    text_pages: int
    visual_pages: int
    text_chunks_stored: int
    visual_vectors_stored: int
    status: str
    error: str | None


@router.post("/upload", response_model=IngestResponse)
async def upload_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and ingest a PDF textbook.

    Accepts PDF files up to 50MB.
    """
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "Only PDF files are accepted.")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(400, f"File too large. Max {MAX_FILE_SIZE_MB}MB.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = UPLOADS_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(contents)

    result = await ingest_pdf(str(save_path))
    return IngestResponse(**result)


@router.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest) -> QueryResponse:
    """Answer a student question using the ingested textbooks."""
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    # Get embeddings for the query
    from app.ingestion.text_embedder import embed_query
    query_emb = await embed_query(request.question)

    if request.text_only:
        results = await retrieve_text_only(
            query_emb["dense_vector"],
            query_emb["sparse_vector"],
            limit=request.limit
        )
    else:
        results = await retrieve(
            request.question,
            query_emb["dense_vector"],
            query_emb["sparse_vector"],
            query_visual=None,
            fusion_limit=request.limit
        )

    answer = await generate_answer(request.question, results)
    return QueryResponse(**answer)


@router.get("/query/stream")
async def query_stream(question: str, text_only: bool = False):
    """Stream answer token by token using SSE."""
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    # Get embeddings for the query
    from app.ingestion.text_embedder import embed_query
    query_emb = await embed_query(question)

    results = await retrieve(
        question,
        query_emb["dense_vector"],
        query_emb["sparse_vector"],
        query_visual=None,
        fusion_limit=5
    )

    async def event_generator():
        async for token in generate_answer_stream(question, results):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Check API and Qdrant connection status."""
    try:
        client = _get_client()
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        return {
            "status": "ok",
            "qdrant": "connected",
            "collections": collection_names
        }
    except Exception as e:
        return {
            "status": "degraded",
            "qdrant": "disconnected",
            "error": str(e)
        }


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    """List all Qdrant collections and their point counts."""
    try:
        client = _get_client()
        collections = client.get_collections()
        result = []
        for col in collections.collections:
            info = client.get_collection(col.name)
            result.append({
                "name": col.name,
                "points_count": info.points_count
            })
        return {"collections": result}
    except Exception as e:
        raise HTTPException(500, str(e))
