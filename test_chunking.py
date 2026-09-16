import json

from services.chunking import build_chunks

DOC_ID = "test-doclaynet-kdd22"
seen_text: set[str] = set()

for real_page_number in [4, 5, 6]:
    with open(f"/home/borrelle/Working/Github Projects/folio/backend/docs_test/resultat_parsing_2206.01062v1-pages-{real_page_number}.json") as f:
        data = json.load(f)

    for page in data["items"]["pages"]:
        chunks = build_chunks(items=page["items"],page_number=real_page_number, doc_id=DOC_ID, seen_texts=seen_text)
        for c in chunks:
            print(c.text)
            print(c.sources)