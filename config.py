import os
from pathlib import Path

EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIM: int = 768

BACKEND_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = BACKEND_DIR.parent

# LanceDB is configurable via env. Default is repo_root/just_rag_lancedb.
# Set JUST_RAG_LANCEDB_PATH to reuse an existing LanceDB folder.
DEFAULT_LANCEDB_PATH: Path = REPO_ROOT / "just_rag_lancedb"
LANCEDB_PATH: str = str(Path(os.getenv("JUST_RAG_LANCEDB_PATH", str(DEFAULT_LANCEDB_PATH))).expanduser().resolve())

DEFAULT_STORED_PDF_DIR: Path = BACKEND_DIR / "stored_pdfs"
STORED_PDF_DIR: Path = Path(os.getenv("JUST_RAG_STORED_PDF_DIR", str(DEFAULT_STORED_PDF_DIR))).expanduser().resolve()

DEFAULT_CITATION_IMAGES_DIR: Path = BACKEND_DIR / "citation_images"
CITATION_IMAGES_DIR: Path = Path(
    os.getenv("JUST_RAG_CITATION_IMAGES_DIR", str(DEFAULT_CITATION_IMAGES_DIR))
).expanduser().resolve()

try:
    CITATION_IMAGE_ZOOM: int = int(os.getenv("JUST_RAG_IMAGE_ZOOM", "2"))
except ValueError:
    CITATION_IMAGE_ZOOM = 2
if CITATION_IMAGE_ZOOM < 1:
    CITATION_IMAGE_ZOOM = 1
