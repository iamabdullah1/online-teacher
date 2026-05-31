# Handoff

## Goal
Build a three-layer visual indexing system that replaces ColPali. The system:
1. Extracts figures from PDF pages (embedded images + vector drawings)
2. Generates rich text descriptions using Cohere's multimodal model
3. Stores descriptions in Qdrant for semantic search during slide generation

## Current State
- All 8 ingestion modules complete
- Visual indexer fully implemented and tested
- 81 tests passing
- System ready for end-to-end testing

## Files in Play
- `app/ingestion/visual_indexer.py` — figure extraction + Cohere description
- `app/ingestion/qdrant_client.py` — FIGURES_COLLECTION, upsert_figures(), search_figures_collection()
- `app/ingestion/ingestion_pipeline.py` — calls index_pdf_figures() during PDF ingestion
- `app/generation/slide_generator.py` — uses search_figures() with 0.65 similarity threshold
- `app/ingestion/visual_embedder.py` — stub, not currently used (ColPali unavailable)

## Changes Made
1. Created `visual_indexer.py` with:
   - `extract_figures_from_page()` — extracts embedded images and vector drawings
   - `describe_figure()` — calls Cohere with base64 image, returns JSON description
   - `index_pdf_figures()` — orchestrates extraction + description for all pages
   - `search_figures()` — queries Qdrant using BGE-M3 embeddings

2. Modified `qdrant_client.py`:
   - Added `FIGURES_COLLECTION = "visual_index"`
   - Added `upsert_figures()` — stores figure descriptions with dense vectors
   - Added `search_figures_collection()` — searches with source_pdf filter

3. Modified `ingestion_pipeline.py`:
   - Added call to `index_pdf_figures()` after text chunk upsert
   - Embeds figure descriptions using BGE-M3 before Qdrant storage
   - Added `figures_stored` to return dict

4. Modified `slide_generator.py`:
   - Replaced `_find_best_image_for_topic()` with `_find_best_figure_for_slide()`
   - Uses BGE-M3 semantic search instead of ColPali
   - Similarity threshold: 0.65
   - Removed visual_chunks assembly and fallback logic

## Dead Ends — Do Not Retry
- **ColPali visual embedding**: Abandoned due to transformers version conflict (>=5.3.0 vs <5.0 with FlagEmbedding). Visual indexer replaces this entirely.
- **Old image matching in slide_generator**: Removed `_find_matching_image_fallback()` — not needed with figure descriptions.
- **visual_pages collection**: Still exists in Qdrant but not used during ingestion (visual_stored=0). Could be deprecated.

## Key Decisions & Assumptions
- Cohere `command-r-plus-08-2024` model handles multimodal image descriptions
- Figure descriptions embedded with BGE-M3 (same as text chunks) for consistent search
- Similarity threshold 0.65 balances precision vs recall for slide images
- Qdrant collection `visual_index` uses 1024-dim dense vectors only (no sparse)

## Next Steps
1. Run full ingestion on a real PDF:
   ```bash
   python -c "from app.ingestion.ingestion_pipeline import ingest_pdf; import asyncio; asyncio.run(ingest_pdf('data/uploads/your.pdf'))"
   ```
2. Verify figures extracted to `data/processed/figures/`
3. Verify `visual_index` collection populated in Qdrant
4. Generate slides and verify images appear:
   ```bash
   python -c "from app.generation.slide_generator import generate_slides; import asyncio; asyncio.run(generate_slides([], 'your.pdf'))"
   ```

## Open Questions
- Does Cohere description generation handle all figure types well?
- Is 0.65 threshold optimal or should it be tuned?
- Should we also extract text captions near figures for better descriptions?
- visual_embedder.py still exists — worth removing or keeping as reference?

## Environment Notes
- Qdrant must be running: `docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant`
- COHERE_API_KEY required in .env for figure descriptions
- BGE-M3 model required for embedding figure descriptions
- Tests: `pytest tests/ -v --tb=short`