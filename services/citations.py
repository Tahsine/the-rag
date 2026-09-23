import json
from typing import Any, Iterator

from schemas import Citation, Locator
from services.agent.prompt import RagAnswer
from services.vectorstore import db, TABLE_NAME


def load_id_to_row() -> dict[int, Any]:
    """Charge les lignes LanceDB indexées par chunk id."""
    try:
        table = db.open_table(TABLE_NAME)
        try:
            df = table.to_pandas()
        except Exception:
            df = None
        if df is None or df.empty:
            return {}
        return {int(row["id"]): row for _, row in df.iterrows()}
    except Exception:
        return {}


def iter_cited_rows(structured: RagAnswer | None, doc_id: str | None) -> Iterator[tuple[int, Any]]:
    """Yield les lignes LanceDB correspondant aux ids cités par l'agent."""
    if not structured or not structured.cited:
        return
    id_to_row = load_id_to_row()
    seen: set[int] = set()
    for cid in structured.cited:
        try:
            chunk_id = int(cid)
        except (TypeError, ValueError):
            continue
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        row = id_to_row.get(chunk_id)
        if row is None:
            continue
        row_doc_id = str(row.get("doc_id", "") or "")
        if doc_id and row_doc_id != doc_id:
            continue
        if not row_doc_id and not doc_id:
            continue
        yield chunk_id, row


def parse_sources(row: Any) -> list[dict]:
    sources_raw = row.get("sources", "[]")
    try:
        sources = json.loads(sources_raw) if isinstance(sources_raw, str) else sources_raw
        if isinstance(sources, list):
            return [source for source in sources if isinstance(source, dict)]
    except Exception:
        pass
    return []


def build_citations(structured: RagAnswer | None, doc_id: str | None) -> list[Citation]:
    """Citations pour les ids cités, sources relues depuis LanceDB (bbox + page)."""
    if not structured or not structured.cited:
        return []
    citations: list[Citation] = []
    try:
        for cid, row in iter_cited_rows(structured, doc_id):
            text = str(row.get("text", ""))[:500]
            page = int(row.get("page", 1))
            bbox = None
            sources = parse_sources(row)
            if sources and isinstance(sources[0].get("bbox"), list) and sources[0]["bbox"]:
                b = sources[0]["bbox"][0]
                # BBox {x,y,w,h}
                bbox = [float(b.get("x", 0)), float(b.get("y", 0)), float(b.get("w", 0)), float(b.get("h", 0))]
            locator = Locator(
                type="pdf",
                doc_id=str(row.get("doc_id", "") or doc_id or ""),
                page=page,
                bbox=bbox,
                textAnchor=text[:80],
            )
            citations.append(
                Citation(
                    citationId=f"c{cid}",
                    locator=locator,
                    snippet=text,
                    score=None,
                )
            )
        return citations
    except Exception:
        return []
