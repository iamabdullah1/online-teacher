import asyncio
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
            is_visual = _is_visual_page(page, text)
            if is_visual or len(text.split()) < MIN_WORDS_FOR_TEXT_PAGE:
                image_path = await loop.run_in_executor(
                    None, _save_page_image, page, page_num + 1
                )
                chunks = []
            else:
                image_path = None
                chunks = _chunk_text(text)
            results.append({
                "page_num": page_num,
                "text": text,
                "is_visual": is_visual,
                "image_path": image_path,
                "chunks": chunks,
                "chunk_count": len(chunks),
            })
            print(
                f"[pdf_parser] Page {page_num}: "
                f"{'visual' if is_visual else 'text'} | chunks: {len(chunks)}"
            )
        except Exception as e:
            print(f"[pdf_parser] Warning: skipping page {page_num}: {e}")
            results.append({
                "page_num": page_num,
                "text": "",
                "is_visual": False,
                "image_path": None,
                "chunks": [],
                "chunk_count": 0,
            })
    print("[pdf_parser] Done.")
    return results