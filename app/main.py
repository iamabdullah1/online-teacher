"""FastAPI application entry point."""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models on startup."""
    print("[startup] Preloading BGE-M3 model...")
    try:
        from app.ingestion.text_embedder import _get_model
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _get_model)
        print("[startup] BGE-M3 ready")
    except Exception as e:
        print(f"[startup] BGE-M3 preload failed: {e}")

    yield

    print("[shutdown] Shutting down...")


app = FastAPI(
    title="Online Teacher Platform",
    description="AI-powered teaching assistant for PDF textbooks",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint returning API info."""
    return {
        "message": "Online Teacher Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }
