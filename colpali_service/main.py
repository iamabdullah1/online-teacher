import torch
import gc
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from colpali_engine.models import ColPali, ColPaliProcessor

app = FastAPI(title="ColPali Microservice", version="1.0.0")

MODEL_NAME = "vidore/colpali-v1.2"
DEVICE = "cpu"

_model: ColPali | None = None
_processor: ColPaliProcessor | None = None

def _get_model() -> tuple[ColPali, ColPaliProcessor]:
    global _model, _processor
    if _model is None:
        print("[colpali_service] Loading model...")
        gc.collect()
        _processor = ColPaliProcessor.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )
        _model = ColPali.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        _model.eval()
        gc.collect()
        print("[colpali_service] Model ready.")
    return _model, _processor

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

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/embed/images", response_model=EmbedImagesResponse)
def embed_images(request: EmbedImagesRequest):
    model, processor = _get_model()
    results = []
    for image_path in request.image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            batch = processor.process_images([image])
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.no_grad():
                embeddings = model(**batch)
            vectors = embeddings[0].cpu().float().numpy()
            results.append(PatchVectors(
                image_path=image_path,
                vectors=[vec.tolist() for vec in vectors],
                vector_count=len(vectors)
            ))
            gc.collect()
        except Exception as e:
            print(f"[colpali_service] Error on {image_path}: {e}")
    return EmbedImagesResponse(results=results)

@app.post("/embed/query", response_model=EmbedQueryResponse)
def embed_query(request: EmbedQueryRequest):
    model, processor = _get_model()
    batch = processor.process_queries([request.query_text])
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    with torch.no_grad():
        embeddings = model(**batch)
    vector = embeddings[0].cpu().float().numpy().tolist()
    gc.collect()
    return EmbedQueryResponse(vector=vector)
