import json

import lancedb
from lancedb.expr import col
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


def list_documents() -> list[dict]:
    """Docs indexés dans LanceDB: [{doc_id, pages, chunks}] (CLI, sans serveur)."""
    try:
        table: Table = db.open_table(TABLE_NAME)
    except Exception:
        return []
    df = table.to_pandas()
    if df is None or df.empty:
        return []
    grouped = df.groupby("doc_id").agg(pages=("page", "nunique"), chunks=("id", "count")).reset_index()
    return grouped.to_dict("records")


def hybrid_search(query_vector: list[float], query_text: str, limit: int = 3, doc_id: str | None = None) -> list[dict]:
    try:
        table: Table = db.open_table(TABLE_NAME)
    except Exception:
        # Table not yet created -> no results (standalone before indexing)
        return []
    query = table.search(query_type="hybrid").vector(query_vector).text(query_text)
    if doc_id:
        query = query.where(col("doc_id") == doc_id)
    results: list[dict] = query.limit(limit).to_list()
    return results