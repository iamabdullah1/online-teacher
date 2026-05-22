# Online Teacher — Build Progress

## Current Status
🔄 Setting up project (pre-module 1)

---

## Module Checklist

- [ ] Module 1 — pdf_parser.py
- [ ] Module 2 — text_embedder.py
- [ ] Module 3 — visual_embedder.py
- [ ] Module 4 — qdrant_client.py
- [ ] Module 5 — ingestion_pipeline.py
- [ ] Module 6 — retriever.py
- [ ] Module 7 — generator.py
- [ ] Module 8 — api/routes.py

---

## Module 1 — pdf_parser.py
**Status:** Complete
**File:** app/ingestion/pdf_parser.py

### Steps
## Module 1 — pdf_parser.py
**Status:** ✅ Complete
- [x] File created
- [x] _is_visual_page() written
- [x] _chunk_text() written
- [x] _save_page_image() written
- [x] extract_pages() written
- [x] Tests written (tests/test_pdf_parser.py)
- [x] All 7 tests passing
### Notes
_(paste any errors or decisions here)_

---

## Module 2 — text_embedder.py
**Status:** Complete

### Steps
## Module 2 — text_embedder.py
**Status:** ✅ Complete
- [x] embed_chunks() written
- [x] embed_query() written  
- [x] BGE-M3 model downloaded and working
- [x] All 8 tests passing

---

## Module 3 — visual_embedder.py
**Status:** Complete (stub - see notes)

### Steps
- [x] File created
- [x] embed_images() written
- [x] embed_query_image() written
- [x] Tests written (tests/test_visual_embedder.py)
- [x] All 8 tests passing

### Notes
Stub implementation due to dependency conflict:
- ColPali requires transformers>=5.3.0 but FlagEmbedding requires <5.0
- Use separate venvs for production: one for text, one for visual

## Module 3 — visual_embedder.py
**Status:** ✅ Complete (stub — see note)
- [x] embed_images() written
- [x] embed_query_image() written
- [x] All 8 tests passing
- ⚠️ KNOWN ISSUE: ColPali conflicts with FlagEmbedding
  (transformers>=5.3.0 vs <5.0)
  Currently returns dummy vectors.
  Fix: separate venv or Docker microservice for ColPali.
  Address after all 8 modules are built.

---

## Module 4 — qdrant_client.py
**Status:** Complete

### Steps
- [x] File created
- [x] init_collections() written
- [x] upsert_text_chunks() written
- [x] upsert_visual_pages() written
- [x] search_text() written
- [x] search_visual() written
- [x] Tests written (tests/test_qdrant_client.py)
- [x] All 8 tests passing

---

## Module 5 — ingestion_pipeline.py
**Status:** Complete

### Steps
- [x] File created
- [x] ingest_pdf() written
- [x] ingest_directory() written
- [x] Tests written (tests/test_ingestion_pipeline.py)
- [x] All 7 tests passing

---

## Module 6 — retriever.py
**Status:** Not started

---

## Module 7 — generator.py
**Status:** Not started

---

## Module 8 — api/routes.py
**Status:** Not started

---

## Setup Checklist
- [x] Project folder created
- [x] Venv created and activated
- [x] Claude Code configured with OpenRouter
- [x] Qdrant Docker command ready
- [ ] code-review-graph installed and built
- [ ] Skills folder created
- [ ] Skills installed
- [ ] CLAUDE.md written
- [x] PROGRESS.md created