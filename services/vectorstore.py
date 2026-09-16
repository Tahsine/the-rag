import json

import lancedb
from lancedb.index import FTS
from lancedb.table import Table

from config import LANCEDB_PATH
from schemas import Chunk

db: lancedb.DBConnection = lancedb.connect(LANCEDB_PATH)
TABLE_NAME: str = "chunks"


def insert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> int:
    table_exists: bool = TABLE_NAME in db.table_names()

    next_id: int = 0
    if table_exists:
        table: Table = db.open_table(TABLE_NAME)
        existing_ids: list[int] = table.to_pandas()["id"].tolist()
        next_id = max(existing_ids, default=-1) + 1

    rows: list[dict] = [
        {
            "id": next_id + i,
            "doc_id": chunk.doc_id,
            "page": chunk.page,
            "text": chunk.text,
            "sources": json.dumps([s.model_dump() for s in chunk.sources]),
            "vector": vector,
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    if table_exists:
        table = db.open_table(TABLE_NAME)
        table.add(rows)
    else:
        table = db.create_table(TABLE_NAME, data=rows)
        table.create_index("text", config=FTS())

    return len(rows)


def hybrid_search(query_vector: list[float], query_text: str, limit: int = 3) -> list[dict]:
    table: Table = db.open_table(TABLE_NAME)
    results: list[dict] = (
        table.search(query_type="hybrid")
        .vector(query_vector)
        .text(query_text)
        .limit(limit)
        .to_list()
    )
    return results