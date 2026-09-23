![Status: Beta](https://img.shields.io/badge/status-beta-orange) ![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

# JustRag — Beta

Local RAG for PDF: LlamaParse parsing, chunking, Gemini embeddings, LanceDB hybrid search (RRF), LangGraph/Ollama agent, Typer+Rich CLI and FastAPI SSE API.

> **Beta — local dev only.** For now only `pip install -e .` (editable) is supported. Data paths default to the repo (`just_rag_lancedb`, `backend/stored_pdfs`, `backend/citation_images`). A global `pip install just-rag` will write inside `site-packages` unless you set `JUST_RAG_*` env vars to a writable absolute path. See [Data](#data).

> **100% cloud for fast free testing.** Beta uses LlamaCloud (parsing), Google Gemini (embeddings) and Ollama Cloud `gpt-oss:20b-cloud` — all have free tiers. Your PDFs and queries are sent to these providers for processing; vectors and source PDFs stay local in `just_rag_lancedb` / `stored_pdfs` (LanceDB is already local). Local alternatives are coming soon.

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .          # editable only for now
cp env.example .env
```

Fill `.env`:

- `GOOGLE_API_KEY` — Gemini embeddings (`gemini-embedding-001`, 768d) → https://aistudio.google.com/app/apikey
- `LLAMA_CLOUD_API_KEY` — LlamaParse (`tier agentic`, `expand=["items"]`) → https://cloud.llamaindex.ai/ → API Keys
- `OLLAMA_API_KEY` or `LLM_API_KEY` — Ollama Cloud (`gpt-oss:20b-cloud`, `reasoning=low`) → https://ollama.com/settings/keys
  - Optional: `OLLAMA_BASE_URL` (default `https://ollama.com`), `OLLAMA_MODEL` / `LLM_MODEL` (default `gpt-oss:20b-cloud`)

Verify:

```bash
just-rag doctor
```

## CLI

```bash
just-rag doctor
just-rag add <path/to/file.pdf> [--doc-id <id>]
just-rag docs
just-rag attach-pdf <doc_id> <path/to/file.pdf>
just-rag ask "question" [--doc <doc_id>] [--thread <name>] [--thinking] [--no-save-images]
just-rag chat [--doc <doc_id>] [--thread <name>] [--thinking] [--no-save-images]
# fallback: python -m just_rag ...
```

Notes:

- `--doc` is optional. If omitted, search runs on the full LanceDB corpus.
- `attach-pdf` attaches a source PDF to an already indexed document to enable citation image generation.
- `--no-save-images` disables saving PNGs of cited pages.

## FastAPI

```bash
uvicorn main:app --reload --port 8000
# or: python -m uvicorn main:app --reload --port 8000
```

Main endpoints:

- `POST /documents` — upload and index a PDF
- `GET /documents` — list documents seen by the API (in-memory)
- `GET /documents/{doc_id}/status` — index status
- `POST /query` — SSE RAG with events `meta`, `thinking`, `tool`, `token`, `citations`, `images`, `done`

The `images` event contains PNGs rendered for cited pages:

```json
{
  "images": [
    {"doc_id": "demo", "page": 4, "path": "backend/citation_images/demo/<answer_id>/page_004.png", "bbox_count": 3}
  ],
  "warnings": []
}
```

## Data

Defaults:

- LanceDB: `<repo_root>/just_rag_lancedb`
- Source PDFs: `backend/stored_pdfs/<doc_id>.pdf`
- Citation images: `backend/citation_images/<doc_id>/<answer_id>/page_NNN.png`

Everything is configurable via `.env`:

```bash
JUST_RAG_LANCEDB_PATH=/absolute/path/to/lancedb
JUST_RAG_STORED_PDF_DIR=/absolute/path/to/stored_pdfs
JUST_RAG_CITATION_IMAGES_DIR=/absolute/path/to/citation_images
JUST_RAG_IMAGE_ZOOM=2
```

If you have a legacy LanceDB folder (e.g. `folio_lancedb`), either move it to `just_rag_lancedb` or point `JUST_RAG_LANCEDB_PATH` to it.

For a global `pip install just-rag` (not yet recommended), set all three `JUST_RAG_*` to writable absolute paths such as `~/.local/share/just-rag/...` — otherwise defaults will try to write inside `site-packages` and fail or be wiped on upgrade.

## Verification scripts

```bash
python test_embeddings.py   # asserts dim == 768
python test_chunking.py
python test_vectorstore.py
python bbox_validation.py   # renders verification_bbox_*.png via PyMuPDF
python test_agent.py        # 4 Qs (List-item, Text, hors corpus, isolation) via gpt-oss:20b-cloud top_k=6, needs OLLAMA_API_KEY/LLM_API_KEY and an indexed corpus
```
