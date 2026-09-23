# JustRag backend

Backend RAG local pour PDF : parsing LlamaParse, chunking, embeddings Gemini, recherche hybride LanceDB, agent LangGraph/Ollama, CLI Typer+Rich et API FastAPI SSE.

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp env.example .env
```

Puis remplis `.env` :

- `GOOGLE_API_KEY` pour les embeddings Gemini
- `LLAMA_CLOUD_API_KEY` pour le parsing LlamaParse
- `OLLAMA_API_KEY` ou `LLM_API_KEY` pour l’agent Ollama Cloud

Vérification :

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
```

Notes :

- `--doc` est optionnel. S’il est absent, la recherche porte sur tout le corpus LanceDB.
- `attach-pdf` permet de rattacher un PDF source à un document déjà indexé, pour activer la génération d’images de citations.
- `--no-save-images` désactive la sauvegarde des PNG des pages citées.

## API FastAPI

```bash
uvicorn main:app --reload --port 8000
```

Endpoints principaux :

- `POST /documents` : upload et indexation d’un PDF
- `GET /documents` : liste des documents vus par l’API en mémoire
- `GET /documents/{doc_id}/status` : statut d’indexation
- `POST /query` : SSE RAG avec events `meta`, `thinking`, `tool`, `token`, `citations`, `images`, `done`

L’event `images` contient les PNG générés pour les pages citées :

```json
{
  "images": [
    {"doc_id": "demo", "page": 4, "path": "backend/citation_images/demo/<answer_id>/page_004.png", "bbox_count": 3}
  ],
  "warnings": []
}
```

## Données

Par défaut :

- LanceDB : `<repo_root>/just_rag_lancedb`
- PDF sources : `backend/stored_pdfs/<doc_id>.pdf`
- Images de citations : `backend/citation_images/<doc_id>/<answer_id>/page_NNN.png`

Tout est configurable via `.env` :

```bash
JUST_RAG_LANCEDB_PATH=/chemin/absolu/vers/lancedb
JUST_RAG_STORED_PDF_DIR=/chemin/absolu/vers/stored_pdfs
JUST_RAG_CITATION_IMAGES_DIR=/chemin/absolu/vers/citation_images
JUST_RAG_IMAGE_ZOOM=2
```

Si tu as un ancien dossier LanceDB, soit tu le déplaces vers `just_rag_lancedb`, soit tu defines `JUST_RAG_LANCEDB_PATH` pour pointer dessus.

## Scripts de vérification

```bash
python test_embeddings.py
python test_chunking.py
python test_vectorstore.py
python bbox_validation.py
python test_agent.py
```

`test_agent.py` nécessite un corpus déjà indexé et une clé Ollama/LLM valide.
