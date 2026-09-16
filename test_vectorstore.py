import json

from services.chunking import build_chunks
from services.embeddings import embed_texts
from services.vectorstore import insert_chunks, hybrid_search
from schemas import Chunk

DOC_ID: str = "test-doclaynet-kdd22"
seen_texts: set[str] = set()
all_chunks: list[Chunk] = []

for real_page_number in [4, 5, 6]:
    with open(f"/home/borrelle/Working/Github Projects/folio/backend/docs_test/resultat_parsing_2206.01062v1-pages-{real_page_number}.json") as f:
        data = json.load(f)
    for page in data["items"]["pages"]:
        chunks: list[Chunk] = build_chunks(
            items=page["items"],
            page_number=real_page_number,
            doc_id=DOC_ID,
            seen_texts=seen_texts,
        )
        all_chunks.extend(chunks)

print(f"{len(all_chunks)} chunks construits")

texts: list[str] = [c.text for c in all_chunks]
vectors: list[list[float]] = embed_texts(texts)

n_inserted: int = insert_chunks(all_chunks, vectors)
print(f"{n_inserted} chunks insérés")

query_vector: list[float] = embed_texts(["What is a List-item?"])[0]
results: list[dict] = hybrid_search(query_vector, "List-item", limit=3)

for r in results:
    print(f"[p{r['page']}] {r['text'][:80]}")