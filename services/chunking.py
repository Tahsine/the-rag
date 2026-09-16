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
        text: str = item["md"]
        normalized: str = " ".join(text.lower().split())

        if item["type"] == "heading":
            pending_heading = text
            continue

        if normalized in seen_texts:
            continue 

        seen_texts.add(normalized)

        final_text: str = f"{pending_heading}: {text}" if pending_heading else text
        pending_heading = None

        bboxes: list[BBox] = [BBox(**b) for b in item["bbox"]]
        source: Source = Source(item_index=item_index, bbox=bboxes)
        chunk: Chunk = Chunk(doc_id=doc_id, page=page_number, text=final_text, sources=[source])
        chunks.append(chunk)
        
    return chunks