import asyncio
import os
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
        text_chunks_stored, visual_vectors_stored, status, error
    """
    try:
        if not Path(pdf_path).exists():
            return {
                "pdf_path": str(pdf_path),
                "total_pages": 0,
                "text_pages": 0,
                "visual_pages": 0,
                "text_chunks_stored": 0,
                "visual_vectors_stored": 0,
                "status": "error",
                "error": f"File not found: {pdf_path}",
            }

        await init_collections()

        pages = await extract_pages(pdf_path)
        print(f"[pipeline] Parsed {len(pages)} pages")

        text_pages = [p for p in pages if not p["is_visual"] and p["chunks"]]
        visual_pages_list = [p for p in pages if p["is_visual"] and p["image_path"]]

        all_chunks = []
        for page in text_pages:
            for i, chunk in enumerate(page["chunks"]):
                all_chunks.append({
                    "chunk": chunk,
                    "page_num": page["page_num"],
                    "source_pdf": str(Path(pdf_path).name),
                    "chunk_index": i,
                    "chapter": page.get("chapter", ""),
                    "section_title": page.get("section_title", ""),
                    "word_count": len(chunk.split()),
                    "ingested_at": page.get("ingested_at", ""),
                    "is_first_chunk": i == 0,
                    "screenshot_path": page.get("screenshot_path", "")
                })

        # Add visual pages as special text chunks (screenshot references only)
        for page in visual_pages_list:
            all_chunks.append({
                "chunk": f"[Visual page {page['page_num']}]",
                "page_num": page["page_num"],
                "source_pdf": str(Path(pdf_path).name),
                "chunk_index": 0,
                "chapter": page.get("chapter", ""),
                "section_title": page.get("section_title", ""),
                "word_count": 0,
                "ingested_at": page.get("ingested_at", ""),
                "is_first_chunk": True,
                "screenshot_path": page.get("image_path", ""),
                "is_visual_page": True
            })

        chunk_texts = [c["chunk"] for c in all_chunks] if all_chunks else []

        if chunk_texts:
            embeddings = await embed_chunks(chunk_texts)
            for i, emb in enumerate(embeddings):
                all_chunks[i]["dense_vector"] = emb["dense_vector"]
                all_chunks[i]["sparse_vector"] = emb["sparse_vector"]
            text_stored = await upsert_text_chunks(all_chunks)
        else:
            text_stored = 0

        print("[pipeline] Indexing figures...")
        figures = await index_pdf_figures(pdf_path, pages)

        if figures:
            descriptions = [f["description"] for f in figures]
            desc_embeddings = await embed_chunks(descriptions)

            for i, fig in enumerate(figures):
                fig["dense_vector"] = desc_embeddings[i]["dense_vector"]
                fig["source_pdf"] = str(Path(pdf_path).name)

            figures_stored = await upsert_figures(figures)
            print(f"[pipeline] Stored {figures_stored} figure descriptions")
        else:
            figures_stored = 0
            print("[pipeline] No figures found in PDF")

        visual_stored = 0
        print("[pipeline] Skipping visual embedding — handled at slide generation time")

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