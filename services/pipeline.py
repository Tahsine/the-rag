import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from schemas import Chunk
from services.chunking import build_chunks
from services.embeddings import embed_texts
from services.parsing import parse_pdf
from services.vectorstore import insert_chunks


class PipelineError(Exception):
    """Erreur du pipeline d'indexation (côté Folio)."""


class UpstreamError(PipelineError):
    """Erreur d'un service upstream (LlamaParse, embeddings)."""


@dataclass
class IndexResult:
    doc_id: str
    pages: int
    chunks: int


def extract_pages(parsed: dict) -> list[dict]:
    # parsed shape: {"items": {"pages": [...]}}  ou {"pages": [...]}
    items = parsed.get("items", parsed)
    if isinstance(items, dict) and "pages" in items:
        return items["pages"]
    if isinstance(parsed, dict) and "pages" in parsed:
        return parsed["pages"]
    raise PipelineError("Format LlamaParse inattendu: pas de pages")


def index_pdf(
    file_path: str | Path,
    doc_id: str | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> IndexResult:
    """Pipeline complet: LlamaParse -> chunks -> embeddings -> LanceDB.

    on_stage: callback optionnel (nom d'etape) pour progress UI (CLI).
    """
    doc_id = doc_id or str(uuid.uuid4())

    def stage(name: str) -> None:
        if on_stage:
            on_stage(name)

    stage("parse")
    try:
        parsed = parse_pdf(str(file_path))
    except Exception as e:
        raise UpstreamError(f"LlamaParse: {e}") from e

    pages = extract_pages(parsed)

    stage("chunks")
    all_chunks: list[Chunk] = []
    seen_texts: set[str] = set()
    for page_idx, page in enumerate(pages):
        page_number = page.get("page_number", page_idx + 1)
        page_items = page.get("items", [])
        if not page_items and isinstance(page, list):
            page_items = page
        if not isinstance(page_items, list):
            continue
        all_chunks.extend(
            build_chunks(
                items=page_items,
                page_number=page_number,
                doc_id=doc_id,
                seen_texts=seen_texts,
            )
        )

    if not all_chunks:
        raise PipelineError("Aucun chunk extrait du PDF")

    stage("embed")
    try:
        vectors = embed_texts([c.text for c in all_chunks])
    except Exception as e:
        raise UpstreamError(f"Embeddings: {e}") from e

    stage("insert")
    try:
        n_inserted = insert_chunks(all_chunks, vectors)
    except Exception as e:
        raise PipelineError(f"LanceDB: {e}") from e

    return IndexResult(doc_id=doc_id, pages=len(pages), chunks=n_inserted)
