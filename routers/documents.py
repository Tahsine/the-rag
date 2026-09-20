import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from schemas import DocumentStatus, DocumentUploadResponse
from services.pipeline import PipelineError, UpstreamError, index_pdf

router = APIRouter()

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Mo
MULTIPART_OVERHEAD = 1024 * 1024  # boundaires + headers multipart (guard précoce)
CHUNK_READ = 256 * 1024
# In-memory registry V0 (mono-PDF, pas de persistance SQLite encore)
# doc_id -> DocumentStatus
_registry: dict[str, DocumentStatus] = {}


@router.post("", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(request: Request, file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())
    filename = file.filename or f"{doc_id}.pdf"

    # 1. Guard précoce Content-Length (heuristic: corps multipart > fichier)
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_FILE_SIZE + MULTIPART_OVERHEAD:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"PDF trop volumineux (Content-Length {content_length}), limite {MAX_FILE_SIZE} bytes",
        )

    # 2. Lecture chunkée avec cutoff: mémoire bornée à MAX_FILE_SIZE + 1 chunk
    contents = bytearray()
    while True:
        chunk = await file.read(CHUNK_READ)
        if not chunk:
            break
        contents.extend(chunk)
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"PDF trop volumineux ({len(contents)} bytes), limite {MAX_FILE_SIZE} bytes",
            )

    # 3. Validation magic bytes %PDF-
    if len(contents) < 4 or bytes(contents[:4]) != b"%PDF":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Fichier non PDF (magic bytes %PDF- manquant)",
        )

    _registry[doc_id] = DocumentStatus(
        doc_id=doc_id, filename=filename, status="indexing", pages=0, chunks=0
    )

    # 4. Temp file (LlamaParse attend un path) + pipeline
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)

        try:
            result = index_pdf(tmp_path, doc_id=doc_id)
        except UpstreamError as e:
            _registry[doc_id] = DocumentStatus(doc_id=doc_id, filename=filename, status="error", error=str(e))
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))
        except PipelineError as e:
            _registry[doc_id] = DocumentStatus(doc_id=doc_id, filename=filename, status="error", error=str(e))
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

        _registry[doc_id] = DocumentStatus(
            doc_id=doc_id, filename=filename, status="ready", pages=result.pages, chunks=result.chunks
        )
        return DocumentUploadResponse(
            doc_id=result.doc_id, filename=filename, pages=result.pages, chunks=result.chunks, status="ready"
        )
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
