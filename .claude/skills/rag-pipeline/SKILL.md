---
name: rag-pipeline
description: >
  Use for RAG pipeline, hybrid search, vector DB, embedding,
  Qdrant, BGE-M3, ColPali, RRF fusion, chunking, ingestion.
  Triggers on: retrieval, embedding, vector, qdrant, colpali,
  bge, chunks, ingestion, hybrid search, pdf parsing.
---

## Online Teacher RAG Rules

### Stack
- Text embeddings: BGE-M3 via FlagEmbedding (dense + sparse)
- Visual embeddings: ColPali vidore/colpali-v1.3
- Vector DB: Qdrant local http://localhost:6333
- Collections: text_chunks and visual_pages

### Chunking
- Size: 384 words (~512 tokens)
- Overlap: 48 words (~64 tokens)
- Step: 336 words

### Visual page detection — visual if ANY:
1. page.get_images() returns >= 1 image
2. text coverage ratio < 0.0005
3. text < 200 chars AND contains figure/diagram/table/chart/graph

### RRF Fusion
- Formula: score = 1 / (60 + rank)
- Sum scores across all lists
- Sort descending, return top 5

### Rules
- All functions async
- Type hints + Google docstrings everywhere
- pathlib.Path never os.path
- Wrap PyMuPDF sync calls in run_in_executor
- Never Pinecone, never OpenAI embeddings
