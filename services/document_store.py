from __future__ import annotations

import re
import shutil
from pathlib import Path

from config import STORED_PDF_DIR

_SAFE_DOC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_doc_id(doc_id: str) -> str:
    doc_id = str(doc_id).strip()
    if not doc_id or not _SAFE_DOC_ID.fullmatch(doc_id) or ".." in doc_id:
        raise ValueError(f"doc_id non sûr pour le stockage du PDF source: {doc_id!r}")
    return doc_id


def stored_pdf_path(doc_id: str) -> Path:
    return STORED_PDF_DIR / f"{validate_doc_id(doc_id)}.pdf"


def store_source_pdf(file_path: str | Path, doc_id: str) -> Path:
    source = Path(file_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"PDF source introuvable: {source}")
    if source.stat().st_size < 4:
        raise ValueError(f"PDF source trop petit pour être valide: {source}")
    destination = stored_pdf_path(doc_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def has_source_pdf(doc_id: str) -> bool:
    try:
        return stored_pdf_path(doc_id).exists()
    except ValueError:
        return False
