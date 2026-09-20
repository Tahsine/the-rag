import json
from pathlib import Path

from services.chunking import build_chunks

DOCS_DIR = Path(__file__).parent / "docs_test"
DOC_ID = "test-doclaynet-kdd22"
seen_text: set[str] = set()

for real_page_number in [4, 5, 6]:
    with open(DOCS_DIR / f"resultat_parsing_2206.01062v1-pages-{real_page_number}.json") as f:
        data = json.load(f)

    for page in data["items"]["pages"]:
        chunks = build_chunks(items=page["items"],page_number=real_page_number, doc_id=DOC_ID, seen_texts=seen_text)
        for c in chunks:
            print(c.text)
            print(c.sources)