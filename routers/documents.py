import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from schemas import DocumentStatus, DocumentUploadResponse
from services.chunking import build_chunks
from services.embeddings import embed_texts
from services.parsing import parse_pdf
from services.vectorstore import insert_chunks

router = APIRouter()

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Mo
# In-memory registry V0 (mono-PDF, pas de persistance SQLite encore)
# doc_id -> DocumentStatus
_registry: dict[str, DocumentStatus] = {}


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    # 1. Validation content_type (faible) + magic bytes %PDF-
    if file.content_type and file.content_type != "application/pdf":
        # on ne bloque pas seulement sur content_type, on vérifie magic bytes après
        pass

    # Lire en limitant à 5 Mo sans charger 2Go en RAM (stream par chunks)
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"PDF trop volumineux ({len(contents)} bytes), limite {MAX_FILE_SIZE} bytes",
        )
    if len(contents) < 4 or contents[:4] != b"%PDF":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fichier non PDF (magic bytes %PDF- manquant)",
        )

    doc_id = str(uuid.uuid4())
    filename = file.filename or f"{doc_id}.pdf"

    # Enregistre indexing
    _registry[doc_id] = DocumentStatus(
        doc_id=doc_id, filename=filename, status="indexing", pages=0, chunks=0
    )

    # 2. Ecriture temp file pour LlamaParse (attend file path)
    suffix = ".pdf"
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        # 3. Parsing LlamaParse
        try:
            parsed = parse_pdf(str(tmp_path))
        except Exception as e:
            _registry[doc_id] = DocumentStatus(
                doc_id=doc_id, filename=filename, status="error", error=f"parse_pdf: {e}"
            )
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LlamaParse error: {e}")

        # parsed shape: {"items": {"pages": [{"items": [...]}, ...]}}  ou {"pages": [...]}
        pages = []
        items = parsed.get("items", parsed)
        if isinstance(items, dict) and "pages" in items:
            pages = items["pages"]
        elif isinstance(parsed, dict) and "pages" in parsed:
            pages = parsed["pages"]
        else:
            raise HTTPException(status_code=500, detail="Format LlamaParse inattendu: pas de pages")

        # 4. Chunking + dedup seen_texts across pages
        all_chunks = []
        seen_texts: set[str] = set()
        for page_idx, page in enumerate(pages):
            # page_number 1-based pour locator
            page_number = page.get("page_number", page_idx + 1)
            # items may be at page["items"] or page itself
            page_items = page.get("items", [])
            if not page_items and isinstance(page, list):
                page_items = page
            if not isinstance(page_items, list):
                continue
            chunks = build_chunks(
                items=page_items,
                page_number=page_number,
                doc_id=doc_id,
                seen_texts=seen_texts,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            _registry[doc_id] = DocumentStatus(doc_id=doc_id, filename=filename, status="error", error="Aucun chunk extrait")
            raise HTTPException(status_code=500, detail="Aucun chunk extrait du PDF")

        # 5. Embeddings batch + insert LanceDB
        texts = [c.text for c in all_chunks]
        try:
            vectors = embed_texts(texts)
        except Exception as e:
            _registry[doc_id] = DocumentStatus(doc_id=doc_id, filename=filename, status="error", error=f"embed_texts: {e}")
            raise HTTPException(status_code=502, detail=f"Embedding error: {e}")

        try:
            n_inserted = insert_chunks(all_chunks, vectors)
        except Exception as e:
            _registry[doc_id] = DocumentStatus(doc_id=doc_id, filename=filename, status="error", error=f"insert_chunks: {e}")
            raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

        # 6. Ready
        _registry[doc_id] = DocumentStatus(
            doc_id=doc_id, filename=filename, status="ready", pages=len(pages), chunks=n_inserted
        )
        return DocumentUploadResponse(doc_id=doc_id, filename=filename, pages=len(pages), chunks=n_inserted, status="ready")

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


@router.get("", response_model=list[DocumentStatus])
async def list_documents():
    return list(_registry.values())


@router.get("/{doc_id}/status", response_model=DocumentStatus)
async def get_status(doc_id: str):
    doc = _registry.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="doc_id not found")
    return doc
