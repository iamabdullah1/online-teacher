import asyncio
import hashlib
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from app.ingestion.pdf_parser import extract_pages
from app.ingestion.text_embedder import embed_chunks
from app.ingestion.visual_indexer import index_pdf_figures
from app.ingestion.qdrant_client import (
    init_collections,
    upsert_text_chunks,
    upsert_figures,
)

load_dotenv()

UPLOADS_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")


async def ingest_pdf(pdf_path: str) -> dict:
    """Full ingestion pipeline for a single PDF file.

    Args:
        pdf_path: Absolute or relative path to PDF file.

    Returns:
        Dict with keys: pdf_path, total_pages, text_pages, visual_pages,
        text_chunks_stored, visual_vectors_stored, figures_stored, status, error
    """
    start_time = time.time()
    try:
        if not Path(pdf_path).exists():
            return {
                "pdf_path": str(pdf_path),
                "total_pages": 0,
                "text_pages": 0,
                "visual_pages": 0,
                "text_chunks_stored": 0,
                "visual_vectors_stored": 0,
                "figures_stored": 0,
                "status": "error",
                "error": f"File not found: {pdf_path}",
            }

        await init_collections()

        # Step 1: Parse PDF pages
        t0 = time.time()
        pages = await extract_pages(pdf_path)
        page_total = len(pages)
        doc_id = hashlib.md5(Path(pdf_path).name.encode()).hexdigest()[:12]
        ingested_at = datetime.utcnow().isoformat()
        print(f"[pipeline] Step 1: Parsed {page_total} pages ({time.time()-t0:.1f}s)")

        # Calculate total chunks across all pages
        total_chunks = sum(len(p.get("chunks", [])) for p in pages)

        text_pages = [p for p in pages if not p["is_visual"] and p["chunks"]]
        visual_pages_list = [p for p in pages if p["is_visual"] and p["image_path"]]

        # Build text chunks (only from text pages, no visual placeholder chunks)
        t1 = time.time()
        all_chunks = []
        for page in text_pages:
            for i, chunk in enumerate(page["chunks"]):
                all_chunks.append({
                    "chunk": chunk,
                    "page_num": page["page_num"],
                    "source_pdf": str(Path(pdf_path).name),
                    "chunk_index": i,
                    "total_chunks": total_chunks,
                    "page_total": page_total,
                    "chapter": page.get("chapter", ""),
                    "section_title": page.get("section_title", ""),
                    "word_count": len(chunk.split()),
                    "doc_id": doc_id,
                    "ingested_at": ingested_at,
                    "is_first_chunk": i == 0
                })
        print(f"[pipeline] Step 2: Built {len(all_chunks)} chunks ({time.time()-t1:.1f}s)")

        # Step 3: Run text embedding and figure extraction CONCURRENTLY
        t2 = time.time()
        chunk_texts = [c["chunk"] for c in all_chunks] if all_chunks else []

        # Start text embedding task
        text_embed_task = embed_chunks(chunk_texts) if chunk_texts else None

        # Start figure extraction task
        figures_task = index_pdf_figures(pdf_path, pages)

        # Run both concurrently
        if text_embed_task and figures_task:
            embeddings, figures = await asyncio.gather(text_embed_task, figures_task)
        elif text_embed_task:
            embeddings = await text_embed_task
            figures = []
        else:
            embeddings = []
            figures = await figures_task

        print(f"[pipeline] Step 3: Text embedding + figure extraction ({time.time()-t2:.1f}s)")

        # Attach embeddings to chunks
        t3 = time.time()
        for i, emb in enumerate(embeddings):
            all_chunks[i]["dense_vector"] = emb["dense_vector"]
            all_chunks[i]["sparse_vector"] = emb["sparse_vector"]

        # Add source_pdf to figures and attach embeddings
        pdf_name = str(Path(pdf_path).name)
        if figures:
            descriptions = [f["description"] for f in figures]
            if descriptions:
                desc_embeddings = await embed_chunks(descriptions)
                for i, fig in enumerate(figures):
                    fig["dense_vector"] = desc_embeddings[i]["dense_vector"]
                    fig["source_pdf"] = pdf_name
                    fig["doc_id"] = doc_id
                    fig["ingested_at"] = ingested_at
                    fig["figure_filename"] = Path(fig["figure_path"]).name

        print(f"[pipeline] Step 4: Attached embeddings ({time.time()-t3:.1f}s)")

        # Step 5: Run Qdrant upserts CONCURRENTLY
        t4 = time.time()
        if all_chunks and figures:
            text_stored, figures_stored = await asyncio.gather(
                upsert_text_chunks(all_chunks),
                upsert_figures(figures)
            )
        elif all_chunks:
            text_stored = await upsert_text_chunks(all_chunks)
            figures_stored = 0
        elif figures:
            text_stored = 0
            figures_stored = await upsert_figures(figures)
        else:
            text_stored = 0
            figures_stored = 0

        visual_stored = 0
        print(f"[pipeline] Step 5: Qdrant upserts ({time.time()-t4:.1f}s)")

        total_time = time.time() - start_time
        print(f"[pipeline] Total ingestion: {total_time:.1f}s")

        return {
            "pdf_path": str(pdf_path),
            "total_pages": len(pages),
            "text_pages": len(text_pages),
            "visual_pages": len(visual_pages_list),
            "text_chunks_stored": text_stored,
            "visual_vectors_stored": visual_stored,
            "figures_stored": figures_stored,
            "status": "success",
            "error": None,
        }
    except Exception as e:
        print(f"[pipeline] Error: {e}")
        return {
            "pdf_path": str(pdf_path),
            "total_pages": 0,
            "text_pages": 0,
            "visual_pages": 0,
            "text_chunks_stored": 0,
            "visual_vectors_stored": 0,
            "figures_stored": 0,
            "status": "error",
            "error": str(e),
        }


async def ingest_directory(directory: str) -> list[dict]:
    """Ingest all PDF files in a directory.

    Args:
        directory: Path to directory containing PDF files.

    Returns:
        List of result dicts, one per PDF file.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {directory}")
    pdf_files = list(dir_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[pipeline] No PDF files found in {directory}")
        return []
    print(f"[pipeline] Found {len(pdf_files)} PDF files")
    results = []
    for pdf_file in pdf_files:
        result = await ingest_pdf(str(pdf_file))
        results.append(result)
    return results