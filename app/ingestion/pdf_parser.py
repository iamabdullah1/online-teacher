import asyncio
import re
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

CHUNK_SIZE_WORDS = 384
CHUNK_OVERLAP_WORDS = 48
CHUNK_STEP = 336
IMAGE_SAVE_DIR = Path("data/processed/images")
TEXT_COVERAGE_THRESHOLD = 0.0005
MIN_WORDS_FOR_TEXT_PAGE = 10
VISUAL_KEYWORDS = ["figure", "diagram", "table", "chart", "graph", "illustration"]
HEADING_FONT_SIZE_THRESHOLD = 14.0
CHAPTER_KEYWORDS = ["chapter", "unit", "part", "section"]


def _clean_text(text: str) -> str:
    """Remove header/footer noise from extracted PDF text.

    Args:
        text: Raw extracted text from PyMuPDF.

    Returns:
        Cleaned text with noise removed.
    """
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip lines with file paths
        if 'File:' in line and ('\\' in line or '/' in line):
            continue
        # Skip lines with .ppt references
        if '.ppt' in line.lower():
            continue
        # Skip copyright lines
        if '©' in line or 'copyright' in line.lower():
            continue
        # Skip short header-style lines (page numbers etc)
        if re.match(r'^Molecular Biology:\s*\d+', line):
            continue
        cleaned.append(line)
    return ' '.join(cleaned).strip()


def _is_visual_page(page: fitz.Page, text: str) -> bool:
    """Determine if a PDF page is visual (image-heavy or diagram-heavy).

    Args:
        page: The PyMuPDF Page object.
        text: The extracted text from the page.

    Returns:
        True if the page is visual, False otherwise.
    """
    if len(page.get_images()) >= 1:
        return True
    text_len = len(text.strip())
    area = page.rect.width * page.rect.height
    if text_len / area < TEXT_COVERAGE_THRESHOLD:
        return True
    if text_len < 200:
        text_lower = text.lower()
        if any(kw in text_lower for kw in VISUAL_KEYWORDS):
            return True
    return False


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks using sliding window.

    Args:
        text: The text to chunk.

    Returns:
        List of text chunks.
    """
    words = text.split()
    if len(words) < MIN_WORDS_FOR_TEXT_PAGE:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += CHUNK_STEP
    return chunks


def _save_page_image(page: fitz.Page, page_num: int) -> str:
    """Save a PDF page as an image.

    Args:
        page: The PyMuPDF Page object.
        page_num: The page number (1-indexed for naming).

    Returns:
        Absolute path to the saved image file.
    """
    IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image_path = IMAGE_SAVE_DIR / f"page_{page_num:04d}.png"
    pix.save(str(image_path))
    return str(image_path.resolve())


def _extract_heading(page: fitz.Page) -> dict[str, str]:
    """Extract chapter and section heading from a page using
    font size and keyword detection.

    Args:
        page: PyMuPDF page object.

    Returns:
        Dict with keys:
          - chapter: str (e.g. "Chapter 3" or "")
          - section_title: str (e.g. "Newton's Laws" or "")
    """
    chapter = ""
    section_title = ""
    try:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0)
                    if not text or len(text) < 2:
                        continue
                    text_lower = text.lower()
                    # Detect chapter heading
                    if any(kw in text_lower for kw in CHAPTER_KEYWORDS):
                        if size >= HEADING_FONT_SIZE_THRESHOLD:
                            chapter = text
                    # Detect section title (large font, not a chapter)
                    elif size >= HEADING_FONT_SIZE_THRESHOLD and not section_title:
                        section_title = text
    except Exception:
        pass
    return {"chapter": chapter, "section_title": section_title}


async def extract_pages(pdf_path: str) -> list[dict]:
    """Extract pages from a PDF file, separating text and visual pages.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries containing page data.

    Raises:
        ValueError: If the PDF cannot be opened.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        raise ValueError(f"Cannot open PDF at path: {pdf_path}")
    print(f"[pdf_parser] Processing: {pdf_path} ({doc.page_count} pages)")
    loop = asyncio.get_event_loop()
    results = []
    for page_num in range(doc.page_count):
        try:
            page = doc.load_page(page_num)
            text = page.get_text()
            text = _clean_text(text)
            is_visual = _is_visual_page(page, text)
            heading = _extract_heading(page)

            # Always save screenshot for all pages (for slides)
            screenshot_path = await loop.run_in_executor(
                None, _save_page_image, page, page_num + 1
            )

            if is_visual or len(text.split()) < MIN_WORDS_FOR_TEXT_PAGE:
                # Visual page: use screenshot for figure extraction
                image_path = screenshot_path
                chunks = []
            else:
                image_path = None
                chunks = _chunk_text(text)

            ingested_at = datetime.utcnow().isoformat()
            results.append({
                "page_num": page_num,
                "text": text,
                "is_visual": is_visual,
                "image_path": image_path,
                "screenshot_path": screenshot_path,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "chapter": heading["chapter"],
                "section_title": heading["section_title"],
                "word_count": len(text.split()),
                "ingested_at": ingested_at,
            })
            print(
                f"[pdf_parser] Page {page_num}: "
                f"{'visual' if is_visual else 'text'} | chunks: {len(chunks)}"
            )
        except Exception as e:
            print(f"[pdf_parser] Warning: skipping page {page_num}: {e}")
            ingested_at = datetime.utcnow().isoformat()
            results.append({
                "page_num": page_num,
                "text": "",
                "is_visual": False,
                "image_path": None,
                "screenshot_path": "",
                "chunks": [],
                "chunk_count": 0,
                "chapter": "",
                "section_title": "",
                "word_count": 0,
                "ingested_at": ingested_at,
            })
    print("[pdf_parser] Done.")
    return results