"""Cloudinary image storage for PDF figures."""

import os
from pathlib import Path
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import cloudinary.api

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
    api_key=os.getenv("CLOUDINARY_API_KEY", ""),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", ""),
)

TEMP_DOWNLOAD_DIR = Path("data/processed/cloudinary_cache")


def upload_figure(local_path: str, public_id: str | None = None) -> dict:
    """Upload a figure image to Cloudinary.

    Args:
        local_path: Absolute or relative path to the figure PNG.
        public_id: Optional public ID (auto-generated if not provided).

    Returns:
        Dict with keys:
          - url: str (direct Cloudinary URL)
          - public_id: str
          - success: bool
          - error: str | None
    """
    if not os.getenv("CLOUDINARY_CLOUD_NAME"):
        print("[cloudinary] Skipping upload — Cloudinary not configured")
        return {"url": "", "public_id": "", "success": False, "error": "Not configured"}

    try:
        if not Path(local_path).exists():
            return {"url": "", "public_id": "", "success": False, "error": f"File not found: {local_path}"}

        response = cloudinary.uploader.upload(
            local_path,
            public_id=public_id,
            folder="online-teacher/figures",
            resource_type="image",
            overwrite=True
        )
        print(f"[cloudinary] Uploaded: {response.get('public_id')}")
        return {
            "url": response.get("secure_url", ""),
            "public_id": response.get("public_id", ""),
            "success": True,
            "error": None
        }
    except Exception as e:
        print(f"[cloudinary] Upload error: {e}")
        return {"url": "", "public_id": "", "success": False, "error": str(e)}


def delete_figure(public_id: str) -> bool:
    """Delete a figure from Cloudinary by public ID.

    Args:
        public_id: The Cloudinary public ID to delete.

    Returns:
        True if deleted or not found, False on error.
    """
    if not public_id or not os.getenv("CLOUDINARY_CLOUD_NAME"):
        return False

    try:
        result = cloudinary.uploader.destroy(public_id)
        if result.get("result") == "ok":
            print(f"[cloudinary] Deleted: {public_id}")
        else:
            print(f"[cloudinary] Delete result for {public_id}: {result}")
        return True
    except Exception as e:
        print(f"[cloudinary] Delete error: {e}")
        return False


def download_figure(url: str, filename: str) -> str | None:
    """Download a figure from Cloudinary to a local temp file.

    Args:
        url: Cloudinary URL of the figure.
        filename: Basename for the local file (e.g. page_0001_fig_00.png).

    Returns:
        Local file path string, or None on failure.
    """
    return _download_file(url, filename, TEMP_DOWNLOAD_DIR)


def download_pdf(cloudinary_url: str, filename: str, target_dir: str | Path = "data/uploads") -> str | None:
    """Download a PDF from Cloudinary to a local directory for ingestion.

    Args:
        cloudinary_url: Cloudinary secure_url of the uploaded PDF.
        filename: Original PDF filename (e.g. book.pdf).
        target_dir: Directory to save the downloaded PDF.

    Returns:
        Local file path string, or None on failure.
    """
    return _download_file(cloudinary_url, filename, Path(target_dir))


def _download_file(url: str, filename: str, target_dir: Path) -> str | None:
    """Download a file from a URL to a local directory."""
    if not url:
        return None

    import requests

    target_dir.mkdir(parents=True, exist_ok=True)
    local_path = target_dir / filename

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        local_path.write_bytes(resp.content)
        print(f"[cloudinary] Downloaded: {local_path}")
        return str(local_path)
    except Exception as e:
        print(f"[cloudinary] Download error: {e}")
        return None
