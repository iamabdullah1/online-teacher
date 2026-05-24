"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(
    title="Online Teacher Platform",
    description="AI-powered teaching assistant for PDF textbooks",
    version="1.0.0"
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
