from pydantic import BaseModel

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