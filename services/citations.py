import json

from schemas import Citation, Locator
from services.agent.prompt import RagAnswer
from services.vectorstore import db, TABLE_NAME


def build_citations(structured: RagAnswer | None, doc_id: str | None) -> list[Citation]:
    """Citations pour les ids cités, sources relues depuis LanceDB (bbox + page)."""
    if not structured or not structured.cited:
        return []
    try:
        table = db.open_table(TABLE_NAME)
        try:
            df = table.to_pandas()
        except Exception:
            df = None
        id_to_row = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                id_to_row[int(row["id"])] = row

        citations: list[Citation] = []
        for cid in structured.cited:
            row = id_to_row.get(int(cid))
            if row is None:
                continue
            row_doc_id = str(row.get("doc_id", "") or "")
            if doc_id and row_doc_id != doc_id:
                continue
            if not row_doc_id and not doc_id:
                continue
            text = str(row.get("text", ""))[:500]
            page = int(row.get("page", 1))
            sources_raw = row.get("sources", "[]")
            bbox = None
            try:
                sources = json.loads(sources_raw) if isinstance(sources_raw, str) else sources_raw
                if sources and isinstance(sources, list) and sources[0].get("bbox"):
                    b = sources[0]["bbox"][0]
                    # BBox {x,y,w,h}
                    bbox = [float(b.get("x", 0)), float(b.get("y", 0)), float(b.get("w", 0)), float(b.get("h", 0))]
            except Exception:
                bbox = None
            locator = Locator(
                type="pdf",
                doc_id=row_doc_id or doc_id or "",
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
