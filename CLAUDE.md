# Online Teacher Platform

## Session Start Protocol
At the start of EVERY session, before doing ANYTHING else:
1. Read this entire CLAUDE.md file completely
2. Read PROGRESS.md from the project root
3. Report back in this exact format:
   ✅ Done: [list completed modules/steps]
   🔄 In Progress: [current module and how far]
   ⏭️ Next: [exact next step]
4. Wait for my instruction — do NOT write any code yet

---

## Stack
- Language: Python 3.11
- API framework: FastAPI + async/await throughout
- Text embeddings: BGE-M3 (local, HuggingFace)
- Visual embeddings: ColPali vidore/colpali-v1.3 (local)
- Vector DB: Qdrant (local Docker instance)
- PDF extraction: PyMuPDF (fitz)
- Retrieval: Hybrid — BGE-M3 dense+sparse + ColPali visual, fused with RRF
- LLM for answers: OpenRouter / MiniMax M2.5

---

## Project Structure
app/ingestion/   → PDF parsing, chunking, embedding, Qdrant indexing
app/retrieval/   → hybrid search, RRF fusion
app/api/         → FastAPI routes
app/generation/  → LLM answer assembly
data/uploads/    → raw PDF uploads
data/processed/  → extracted chunks and page images
tests/           → pytest test suite

---

## Coding Rules — enforce on every line
- Every function must have type hints on params AND return value
- Every function must have a Google-style docstring
- All I/O must be async — no synchronous operations inside async functions
- PyMuPDF calls are synchronous — always wrap in run_in_executor
- Never hardcode API keys — always read from .env via os.getenv()
- Use pathlib.Path everywhere — never os.path or string concatenation
- Chunk size: 384 words (~512 tokens), overlap: 48 words (~64 tokens)
- Text pages → Qdrant collection: "text_chunks"
- Visual pages → Qdrant collection: "visual_pages"
- A page is visual if: has images OR text coverage < 0.0005 OR
  text < 200 chars AND contains figure/diagram/table/chart/graph
- RRF constant k=60
- Always write pytest tests for every new function
- All test functions must be async using pytest-asyncio
- Line length max 88 characters

---

## What NOT to Do
- Do NOT use synchronous file I/O inside async functions
- Do NOT use Pinecone — Qdrant only
- Do NOT use OpenAI embeddings — BGE-M3 locally only
- Do NOT use os.path — pathlib.Path only
- Do NOT add packages not already in requirements.txt
- Do NOT use classes — functions only
- Do NOT add if __name__ == "__main__" blocks
- Do NOT use the logging module — print() only

---

## Build Order
Complete modules in this exact order — each depends on the previous:
1. app/ingestion/pdf_parser.py        ← START HERE
2. app/ingestion/text_embedder.py
3. app/ingestion/visual_embedder.py
4. app/ingestion/qdrant_client.py
5. app/ingestion/ingestion_pipeline.py
6. app/retrieval/retriever.py
7. app/generation/generator.py
8. app/api/routes.py

---

## MCP Tools: code-review-graph
ALWAYS use graph tools BEFORE Grep/Glob/Read to explore the codebase.
The graph is faster, cheaper, and gives structural context file scanning cannot.

| Tool | Use when |
|---|---|
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

Fall back to Grep/Glob/Read ONLY when the graph doesn't cover what you need.
Graph auto-updates on every file save via hooks.