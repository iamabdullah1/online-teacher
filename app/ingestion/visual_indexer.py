"""Visual indexer — detects figures in PDF pages, crops them, generates text descriptions from page content, and stores in Qdrant visual_index collection."""

import asyncio
import os
from pathlib import Path
from datetime import datetime
from typing import Any

import fitz
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

FIGURES_DIR = Path("data/processed/figures")
MIN_FIGURE_WIDTH = 100
MIN_FIGURE_HEIGHT = 100
MAX_TOTAL_FIGURES = int(os.getenv("MAX_TOTAL_FIGURES", "8"))


def extract_figures_from_page(
    pdf_path: str,
    page_num: int
) -> list[dict[str, Any]]:
    """Extract and crop figures from a single PDF page.

    Args:
        pdf_path: Path to PDF file.
        page_num: 0-indexed page number.

    Returns:
        List of dicts with keys:
          - figure_path: str (path to cropped PNG)
          - page_num: int
          - figure_index: int (0-indexed within page)
          - bbox: list[float] (x0, y0, x1, y1)
          - width: int (pixels)
          - height: int (pixels)
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figures = []
    image_list = page.get_images(full=True)

    for img_index, img in enumerate(image_list):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]

            pil_img = Image.open(
                __import__("io").BytesIO(img_bytes)
            ).convert("RGB")
            w, h = pil_img.size

            if w < MIN_FIGURE_WIDTH or h < MIN_FIGURE_HEIGHT:
                continue

            fig_filename = f"page_{page_num:04d}_fig_{img_index:02d}.png"
            fig_path = FIGURES_DIR / fig_filename
            pil_img.save(str(fig_path))

            figures.append({
                "figure_path": str(fig_path.resolve()),
                "page_num": page_num,
                "figure_index": img_index,
                "bbox": list(page.rect),
                "width": w,
                "height": h
            })
        except Exception as e:
            print(f"[visual_indexer] Error extracting image: {e}")

    drawings = page.get_drawings()
    if len(drawings) > 10:
        try:
            rects = [d["rect"] for d in drawings if d.get("rect")]
            if rects:
                clip = rects[0]
                for r in rects[1:]:
                    clip |= r
                clip &= page.rect
            else:
                clip = page.rect

            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            fig_filename = f"page_{page_num:04d}_drawing.png"
            fig_path = FIGURES_DIR / fig_filename
            pix.save(str(fig_path))

            figures.append({
                "figure_path": str(fig_path.resolve()),
                "page_num": page_num,
                "figure_index": 99,
                "bbox": list(clip),
                "width": pix.width,
                "height": pix.height
            })
        except Exception as e:
            print(f"[visual_indexer] Error rendering drawing: {e}")

    doc.close()
    return figures


def _describe_from_page_text(figure: dict[str, Any], page_text: str = "", chapter: str = "", section_title: str = "") -> dict[str, Any]:
    """Build a figure description from surrounding page content.

    Uses the page's chapter, section title, and nearby text to describe
    what the figure likely shows — no Cohere vision API needed.

    Args:
        figure: Figure dict with figure_path and metadata.
        page_text: Surrounding text from the page for context.
        chapter: Chapter heading from the page.
        section_title: Section heading from the page.

    Returns:
        Figure dict with added keys:
          - description: str
          - keywords: list[str]
          - has_diagram: bool
          - has_table: bool
          - has_formula: bool
          - subject: str (main topic)
    """
    text_snippet = page_text[:600].strip() if page_text else ""

    parts = []
    if chapter:
        parts.append(f"Chapter: {chapter}")
    if section_title:
        parts.append(f"Section: {section_title}")
    if text_snippet:
        parts.append(text_snippet)

    description = " | ".join(parts) if parts else f"Figure on page {figure['page_num']}"

    # Use page text to detect content type
    text_lower = text_snippet.lower()
    has_diagram = any(w in text_lower for w in ["diagram", "figure", "graph", "chart", "plot", "schematic"])
    has_table = any(w in text_lower for w in ["table", "tabular", "column"])
    has_formula = any(w in text_lower for w in ["formula", "equation", "f=", "="])

    # Extract keywords from section title and page text
    keywords = []
    if section_title:
        keywords = [w.strip().strip(".,:;!?") for w in section_title.split() if len(w) > 3]
    if chapter:
        chapter_keywords = [w.strip().strip(".,:;!?") for w in chapter.split() if len(w) > 3]
        keywords = list(dict.fromkeys(chapter_keywords + keywords))

    subject = chapter if chapter else (section_title if section_title else "general")

    return {
        **figure,
        "description": description[:800],
        "keywords": keywords[:10],
        "has_diagram": has_diagram,
        "has_table": has_table,
        "has_formula": has_formula,
        "subject": subject,
        "figure_filename": Path(figure["figure_path"]).name,
        "ingested_at": datetime.utcnow().isoformat()
    }


async def index_pdf_figures(
    pdf_path: str,
    pages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Extract and index all figures from a PDF.

    Args:
        pdf_path: Path to PDF file.
        pages: List of page dicts from pdf_parser.extract_pages()

    Returns:
        List of figure dicts with text-based descriptions ready for Qdrant.
    """
    print(f"[visual_indexer] Indexing figures in {pdf_path}")
    import time
    start = time.time()

    loop = asyncio.get_event_loop()

    extract_tasks = [
        loop.run_in_executor(
            None,
            extract_figures_from_page,
            pdf_path,
            page["page_num"]
        )
        for page in pages
    ]
    all_page_figures = await asyncio.gather(*extract_tasks)

    figures = []
    for page, page_figs in zip(pages, all_page_figures):
        page_text = page.get("text", "")
        chapter = page.get("chapter", "")
        section_title = page.get("section_title", "")
        for fig in page_figs:
            desc = _describe_from_page_text(fig, page_text, chapter, section_title)
            figures.append(desc)

    print(f"[visual_indexer] Extracted {len(figures)} figures in {time.time()-start:.1f}s")

    # Cap total figures
    if len(figures) > MAX_TOTAL_FIGURES:
        print(f"[visual_indexer] Capping at {MAX_TOTAL_FIGURES} figures (found {len(figures)})")
        figures = figures[:MAX_TOTAL_FIGURES]

    print(f"[visual_indexer] Total figures indexed: {len(figures)} ({time.time()-start:.1f}s)")
    return figures


async def search_figures(
    query: str,
    source_pdf: str,
    limit: int = 3
) -> list[dict[str, Any]]:
    """Search indexed figures by text query using MiniLM.

    Args:
        query: Search query (e.g. slide title)
        source_pdf: PDF filename to filter results
        limit: Max results to return

    Returns:
        List of matching figure dicts with score.
    """
    from app.ingestion.text_embedder import embed_query
    from app.ingestion.qdrant_client import search_figures_collection

    query_embedding = await embed_query(query)
    results = await search_figures_collection(
        dense_vector=query_embedding["dense_vector"],
        source_pdf=source_pdf,
        limit=limit
    )
    return results
