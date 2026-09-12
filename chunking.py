import json
import lancedb
from lancedb.index import FTS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM
        )
    )
    return [e.values for e in response.embeddings]

# Chunking function

# Chunking fonction

def build_chunks(
    items: list[dict],
    page_number: int,
    doc_id: str,
    seen_texts: set[str],
) -> list[dict]:
    chunks: list[dict] = []
    pending_heading = None # dernier titre vu à ne pas dupliqué

    for item_index, item in enumerate(items):
        text: str = item["md"]
        normalized = " ".join(text.lower().split())

        if item["type"] == "heading":
            pending_heading = text
            continue # un titre seul ne peut pas être un chunk, pas bon pour le embedding

        if normalized in seen_texts:
            continue # passe si texte déjà vue ailleur
        seen_texts.add(normalized)

        final_text = f"{pending_heading}: {text}" if pending_heading else text
        pending_heading = None

        chunks.append({
            "doc_id": doc_id,
            "page": page_number,
            "text": final_text,
            "sources": [
                {
                    "item_index": item_index,
                    "bbox": item["bbox"],
                }
            ]
        })

    return chunks


# Test Chunking

DOC_ID = "test-doclaynet-kdd22"
seen_texts: set[str] = set()
all_chunks: list = []

for real_page_number in [4, 5, 6]:
    with open(f"/home/borrelle/Working/Github Projects/folio/backend/docs_test/resultat_parsing_2206.01062v1-pages-{real_page_number}.json") as f:
        data = json.load(f)

    # parse_result = data["items"]
    # pages = parse_result["pages"]

    for page in data["items"]["pages"]:
        page_chunks = build_chunks(
            items=page["items"],
            page_number=real_page_number,
            doc_id=DOC_ID,
            seen_texts=seen_texts
        )
        all_chunks.extend(page_chunks)

print(f"{len(all_chunks)} chunks construits")

# Embeddigs

texts = [c["text"] for c in all_chunks]
vectors = embed_texts(texts)

# Stockage lanceDB

db = lancedb.connect("./folio_lancedb")

rows = [
    {
        "id": i,
        "doc_id": chunk["doc_id"],
        "page": chunk["page"],
        "text": chunk["text"],
        "sources": json.dumps(chunk["sources"]),
        "vector": vector,
    }
    for i, (chunk, vector) in enumerate(zip(all_chunks, vectors))
]

table = db.create_table("chunks", data=rows, mode="overwrite")
table.create_index("text", config=FTS())

print(f"{len(rows)} chunks insérés dans LanceDB")

# Test

query_vector = embed_texts(["What is a List-item?"])[0]

results = (
    table.search(query_type="hybrid")
    .vector(query_vector)
    .text("List-item")
    .limit(3)
    .to_list()
)

for r in results:
    print(f"[p{r['page']}, {r['text'][:80]}")