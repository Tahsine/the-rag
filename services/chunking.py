from schemas import BBox, Source, Chunk

def build_chunks(
    items: list[dict],
    page_number: int,
    doc_id: str,
    seen_texts: set[str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    pending_heading: str | None = None

    for item_index, item in enumerate(items):
        raw_text = item.get("md")
        text: str = raw_text if isinstance(raw_text, str) else str(raw_text or "")
        if not text.strip():
            continue
        normalized: str = " ".join(text.lower().split())

        if item.get("type") == "heading":
            pending_heading = text
            continue

        if normalized in seen_texts:
            continue 

        seen_texts.add(normalized)

        final_text: str = f"{pending_heading}: {text}" if pending_heading else text
        pending_heading = None

        # LlamaParse agentic peut renvoyer bbox=None (ex: 2206.01062v1.pdf complet) — on garde le texte mais sans géométrie
        raw_bbox = item.get("bbox")
        bboxes: list[BBox] = []
        if isinstance(raw_bbox, list):
            for b in raw_bbox:
                if not isinstance(b, dict):
                    continue
                # requis: x,y,w,h — si manque, on skip ce bbox (pas le chunk)
                if not all(k in b and b[k] is not None for k in ("x", "y", "w", "h")):
                    continue
                try:
                    bboxes.append(BBox(**b))
                except Exception:
                    continue
        source: Source = Source(item_index=item_index, bbox=bboxes)
        chunk: Chunk = Chunk(doc_id=doc_id, page=page_number, text=final_text, sources=[source])
        chunks.append(chunk)
        
    return chunks