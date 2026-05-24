"""
ColPali microservice — standalone FastAPI server.
Runs in its own Docker container with its own dependencies.
Exposes two endpoints for the main app to call via HTTP.
"""

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

app = FastAPI(title="ColPali Microservice", version="1.0.0")

MODEL_NAME = "vidore/colpali-v1.3"
DEVICE = "cpu"

# Lazy loaded model
_model: ColPali | None = None
_processor: ColPaliProcessor | None = None


def _get_model() -> tuple[ColPali, ColPaliProcessor]:
    """Load ColPali model once on first request."""
    global _model, _processor
    if _model is None:
        print("[colpali_service] Loading model...")
        _processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
        _model = ColPali.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=DEVICE
        )
        _model.eval()
        print("[colpali_service] Model ready.")
    return _model, _processor


# ── Pydantic models ──────────────────────────────────────

class EmbedImagesRequest(BaseModel):
    image_paths: list[str]


class EmbedQueryRequest(BaseModel):
    query_text: str


class PatchVectors(BaseModel):
    image_path: str
    vectors: list[list[float]]
    vector_count: int


class EmbedImagesResponse(BaseModel):
    results: list[PatchVectors]


class EmbedQueryResponse(BaseModel):
    vector: list[float]


# ── Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/embed/images", response_model=EmbedImagesResponse)
def embed_images(request: EmbedImagesRequest):
    """
    Embed a list of page images using ColPali.
    Returns multi-vector per image (one vector per patch).
    """
    model, processor = _get_model()
    results = []
    for image_path in request.image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            batch = processor.process_images([image])
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                embeddings = model(**batch)
            vectors = embeddings[0].cpu().numpy()
            results.append(PatchVectors(
                image_path=image_path,
                vectors=[vec.tolist() for vec in vectors],
                vector_count=len(vectors)
            ))
        except Exception as e:
            print(f"[colpali_service] Error on {image_path}: {e}")
    return EmbedImagesResponse(results=results)


@app.post("/embed/query", response_model=EmbedQueryResponse)
def embed_query(request: EmbedQueryRequest):
    """
    Embed a text query for visual search using ColPali.
    Returns a single query vector (128 dims).
    """
    model, processor = _get_model()
    batch = processor.process_queries([request.query_text])
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        embeddings = model(**batch)
    vector = embeddings[0].cpu().numpy().tolist()
    return EmbedQueryResponse(vector=vector)