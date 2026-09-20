import json

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from services.embeddings import embed_texts
from services.vectorstore import hybrid_search


@tool
def retrieve(query: str, config: RunnableConfig, top_k: int = 6) -> str:
    """Search the indexed PDF corpus (LanceDB hybrid dense+BM25 RRF).
    Use when the question needs facts from the uploaded PDF.
    Returns passages as: [id] (doc:X p.Y) text ... | sources=JSON.
    If no passage is relevant, returns 'NO_RESULTS'."""
    if not query or not query.strip():
        return "NO_RESULTS: empty query."

    # clamp top_k for safety
    top_k = max(1, min(top_k, 12))

    doc_id: str | None = None
    if config:
        configurable = config.get("configurable") or {}
        doc_id = configurable.get("doc_id") or None

    try:
        q_vec: list[float] = embed_texts([query])[0]
    except Exception as e:
        return f"ERROR embedding query: {e}"

    try:
        hits: list[dict] = hybrid_search(q_vec, query, limit=top_k, doc_id=doc_id)
    except Exception as e:
        return f"ERROR hybrid_search: {e}"

    if not hits:
        return "NO_RESULTS: aucun passage pertinent trouvé pour cette question."

    # Dedup by normalized text (LanceDB accumulates duplicates across test_vectorstore runs)
    seen: set[str] = set()
    uniq_hits: list[dict] = []
    for h in hits:
        norm = " ".join(str(h.get("text", "")).lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        uniq_hits.append(h)

    lines: list[str] = []
    for h in uniq_hits:
        snippet = str(h.get("text", ""))[:500].replace("\n", " ").strip()
        page = h.get("page", "?")
        doc = h.get("doc_id", "?")
        hid = h.get("id", "?")
        sources = h.get("sources", "[]")
        if isinstance(sources, (dict, list)):
            sources = json.dumps(sources, ensure_ascii=False)
        lines.append(f"[{hid}] (doc:{doc} p.{page}) {snippet} | sources={sources}")

    return "\n\n".join(lines)
