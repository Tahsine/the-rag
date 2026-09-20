from typing import Literal

from pydantic import BaseModel, Field

class BBox(BaseModel):
    h: float
    w: float
    x: float
    y: float
    confidence: float | None = None
    label: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    r: float | None = None

class Source(BaseModel):
    item_index: int
    bbox: list[BBox]

class Chunk(BaseModel):
    doc_id: str
    page: int
    text: str
    sources: list[Source]


# --- V0 API DTOs (etape 5) ---
class DocumentStatus(BaseModel):
    doc_id: str
    filename: str
    status: Literal["indexing", "ready", "error"]
    pages: int = 0
    chunks: int = 0
    error: str | None = None


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    status: str = "ready"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, pattern=r"\S")
    doc_id: str | None = None  # optionnel: filtre la recherche ; absent = corpus global
    session_id: str = Field(..., min_length=1, max_length=128, description="Frontend UUID, part of thread_id")
    top_k: int = Field(default=6, ge=1, le=12)


class Locator(BaseModel):
    type: Literal["pdf"] = "pdf"
    doc_id: str
    page: int
    bbox: list[float] | None = None  # [x,y,w,h] or [x0,y0,x1,y1] depending on source
    textAnchor: str | None = None


class Citation(BaseModel):
    citationId: str
    locator: Locator
    snippet: str
    score: float | None = None