"""Visual indexer — detects figures in PDF pages, crops them, generates Cohere descriptions, and stores in Qdrant visual_index collection."""

import os
import json
import asyncio
import base64
from pathlib import Path
from datetime import datetime
from typing import Any

import cohere
import fitz
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_MODEL = "c4ai-aya-vision-32b"
FIGURES_DIR = Path("data/processed/figures")
MIN_FIGURE_WIDTH = 100
MIN_FIGURE_HEIGHT = 100
SIMILARITY_THRESHOLD = 0.65


def _get_cohere_client() -> cohere.ClientV2:
    """Get Cohere client."""
    return cohere.ClientV2(api_key=COHERE_API_KEY)


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
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            fig_filename = f"page_{page_num:04d}_drawing.png"
            fig_path = FIGURES_DIR / fig_filename
            pix.save(str(fig_path))

            figures.append({
                "figure_path": str(fig_path.resolve()),
                "page_num": page_num,
                "figure_index": 99,
                "bbox": list(page.rect),
                "width": pix.width,
                "height": pix.height
            })
        except Exception as e:
            print(f"[visual_indexer] Error rendering drawing: {e}")

    doc.close()
    return figures


def _image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def describe_figure(
    figure: dict[str, Any],
    page_text: str = ""
) -> dict[str, Any]:
    """Generate rich description of a figure using Cohere.

    Args:
        figure: Figure dict with figure_path and metadata.
        page_text: Surrounding text from the page for context.

    Returns:
        Figure dict with added keys:
          - description: str (rich text description)
          - keywords: list[str]
          - has_diagram: bool
          - has_table: bool
          - has_formula: bool
          - subject: str (main topic)
    """
    try:
        client = _get_cohere_client()
        img_b64 = _image_to_base64(figure["figure_path"])

        context = ""
        if page_text:
            context = f"\n\nSurrounding text context:\n{page_text[:500]}"

        prompt = f"""Analyze this figure from an educational textbook.
{context}

Provide a detailed description in JSON format:
{{
    "description": "detailed description of what this figure shows",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "has_diagram": true/false,
    "has_table": true/false,
    "has_formula": true/false,
    "subject": "main academic subject or topic shown"
}}

Be specific about visual elements: arrows, labels, structures,
data, relationships. Return ONLY the JSON object."""

        loop = asyncio.get_event_loop()

        def _call():
            response = client.chat(
                model=COHERE_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            return response.message.content[0].text

        raw = await loop.run_in_executor(None, _call)

        try:
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
        except Exception:
            data = {
                "description": raw[:500],
                "keywords": [],
                "has_diagram": True,
                "has_table": False,
                "has_formula": False,
                "subject": "unknown"
            }

        return {
            **figure,
            "description": data.get("description", ""),
            "keywords": data.get("keywords", []),
            "has_diagram": data.get("has_diagram", False),
            "has_table": data.get("has_table", False),
            "has_formula": data.get("has_formula", False),
            "subject": data.get("subject", ""),
            "indexed_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        print(f"[visual_indexer] Description error: {e}")
        return {
            **figure,
            "description": f"Figure on page {figure['page_num']}",
            "keywords": [],
            "has_diagram": False,
            "has_table": False,
            "has_formula": False,
            "subject": "",
            "indexed_at": datetime.utcnow().isoformat()
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
        List of fully described figure dicts ready for Qdrant.
    """
    print(f"[visual_indexer] Indexing figures in {pdf_path}")
    all_figures = []

    for page in pages:
        page_num = page["page_num"]
        page_text = page.get("text", "")

        figures = extract_figures_from_page(pdf_path, page_num)

        if not figures:
            continue

        print(f"[visual_indexer] Page {page_num}: {len(figures)} figures found")

        tasks = [
            describe_figure(fig, page_text)
            for fig in figures
        ]
        described = await asyncio.gather(*tasks)
        all_figures.extend(described)

    print(f"[visual_indexer] Total figures indexed: {len(all_figures)}")
    return all_figures


async def search_figures(
    query: str,
    source_pdf: str,
    limit: int = 3
) -> list[dict[str, Any]]:
    """Search indexed figures by text query using BGE-M3.

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
