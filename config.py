from pathlib import Path

EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIM: int = 768
# LanceDB is at repo root (folio/folio_lancedb), not backend/folio_lancedb
# Use absolute path so CWD doesn't matter (backend vs repo root)
LANCEDB_PATH: str = str((Path(__file__).resolve().parent.parent / "folio_lancedb").resolve())