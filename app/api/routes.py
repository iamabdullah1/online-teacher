"""FastAPI routes for the Online Teacher Platform."""

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

from app.ingestion.ingestion_pipeline import ingest_pdf
from app.retrieval.retriever import retrieve, retrieve_text_only
from app.generation.generator import generate_answer, generate_answer_stream
from app.generation.slide_generator import generate_slides
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
    figures_stored: int
    status: str
    error: str | None


class DeleteResponse(BaseModel):
    """Response model for PDF deletion."""
    source_pdf: str
    text_chunks_deleted: bool
    figures_deleted: bool
    files_deleted: int
    status: str
    error: str | None


class SlideRequest(BaseModel):
    """Request model for slide generation."""
    source_pdf: str
    num_slides: int = 10
    bullets_per_slide: int = 4
    depth: str = "detailed"
    topic_focus: str = "all"
    style: str = "academic"
    include_title_slide: bool = True
    include_summary_slide: bool = True


class SlideResponse(BaseModel):
    """Response model for slide generation."""
    file_path: str
    num_slides: int
    topics: list[str]
    status: str
    error: str | None
    download_url: str | None


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


@router.delete("/upload/{filename}", response_model=DeleteResponse)
async def delete_pdf(filename: str):
    """Delete a PDF and all associated data.

    Removes: PDF file, text chunks, figure vectors,
    figure images, page screenshots.
    """
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")

    pdf_path = Path("data/uploads") / filename
    if not pdf_path.exists():
        raise HTTPException(404, f"PDF not found: {filename}")

    files_deleted = 0

    try:
        pdf_path.unlink()
        files_deleted += 1
        print(f"[routes] Deleted PDF: {filename}")
    except Exception as e:
        print(f"[routes] Error deleting PDF: {e}")

    from app.ingestion.qdrant_client import delete_pdf_data
    qdrant_result = await delete_pdf_data(filename)

    figures_dir = Path("data/processed/figures")
    if figures_dir.exists():
        pdf_stem = Path(filename).stem
        for fig_file in figures_dir.glob("page_*.png"):
            try:
                if pdf_stem in fig_file.name:
                    fig_file.unlink()
                    files_deleted += 1
            except Exception as e:
                print(f"[routes] Error deleting figure: {e}")

    images_dir = Path("data/processed/images")
    if images_dir.exists():
        pdf_stem = Path(filename).stem
        for img_file in images_dir.glob("page_*.png"):
            try:
                if pdf_stem in img_file.name:
                    img_file.unlink()
                    files_deleted += 1
            except Exception as e:
                print(f"[routes] Error deleting screenshot: {e}")

    return DeleteResponse(
        source_pdf=filename,
        text_chunks_deleted=qdrant_result["text_chunks_deleted"],
        figures_deleted=qdrant_result["figures_deleted"],
        files_deleted=files_deleted,
        status="success",
        error=None,
    )


@router.get("/uploads")
async def list_uploads():
    """List all uploaded PDFs with their ingestion status."""
    uploads_dir = Path("data/uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    pdfs = []
    for pdf_file in uploads_dir.glob("*.pdf"):
        try:
            from app.ingestion.qdrant_client import _get_client
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            client = _get_client()
            result = client.count(
                collection_name="text_chunks",
                count_filter=Filter(
                    must=[FieldCondition(
                        key="source_pdf",
                        match=MatchValue(value=pdf_file.name)
                    )]
                ),
                exact=True
            )
            chunk_count = result.count
        except Exception as e:
            print(f"[routes] Count error: {e}")
            chunk_count = 0

        pdfs.append({
            "filename": pdf_file.name,
            "size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2),
            "chunk_count": chunk_count,
            "ingested": chunk_count > 0
        })

    return {"pdfs": pdfs}

@router.post("/query", response_model=QueryResponse)
async def query_pdf(request: QueryRequest) -> QueryResponse:
    """Answer a student question using the ingested textbooks."""
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    # Get embeddings for the query
    from app.ingestion.text_embedder import embed_query
    query_emb = await embed_query(request.question)

    results = await retrieve_text_only(
        query_emb["dense_vector"],
        query_emb["sparse_vector"],
        limit=request.limit
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
        query_emb["dense_vector"],
        query_emb["sparse_vector"],
        limit=5
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


@router.post("/generate/slides", response_model=SlideResponse)
async def generate_slides_endpoint(request: SlideRequest):
    """Generate a PowerPoint presentation from an ingested PDF."""
    if not request.source_pdf.strip():
        raise HTTPException(400, "source_pdf cannot be empty.")
    if not 1 <= request.num_slides <= 30:
        raise HTTPException(400, "num_slides must be between 1 and 30.")
    if not 2 <= request.bullets_per_slide <= 6:
        raise HTTPException(400, "bullets_per_slide must be between 2 and 6.")
    if request.depth not in ["summary", "detailed", "exam"]:
        raise HTTPException(400, "depth must be: summary, detailed, or exam.")
    if request.style not in ["academic", "simple", "visual_hints"]:
        raise HTTPException(400, "style must be: academic, simple, or visual_hints.")

    try:
        client = _get_client()
        results = client.scroll(
            collection_name="text_chunks",
            scroll_filter={
                "must": [{
                    "key": "source_pdf",
                    "match": {"value": request.source_pdf}
                }]
            },
            limit=200,
            with_payload=True
        )
        chunks = [point.payload for point in results[0]]
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch chunks: {str(e)}")

    if not chunks:
        raise HTTPException(
            404,
            f"No content found for '{request.source_pdf}'. "
            "Please upload and ingest the PDF first."
        )

    result = await generate_slides(
        chunks=chunks,
        source_pdf=request.source_pdf,
        num_slides=request.num_slides,
        bullets_per_slide=request.bullets_per_slide,
        depth=request.depth,
        topic_focus=request.topic_focus,
        style=request.style,
        include_title_slide=request.include_title_slide,
        include_summary_slide=request.include_summary_slide
    )

    if result["status"] == "success":
        filename = Path(result["file_path"]).name
        download_url = f"/api/v1/slides/download/{filename}"
    else:
        download_url = None

    return SlideResponse(**result, download_url=download_url)


@router.get("/slides/download/{filename}")
async def download_slides(filename: str):
    """Download a generated .pptx file."""
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Invalid filename.")
    file_path = Path("data/outputs") / filename
    if not file_path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )
