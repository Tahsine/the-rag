import asyncio
import json
import time
from typing import AsyncIterable

from fastapi import APIRouter, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, ToolMessage

from schemas import Citation, Locator, QueryRequest
from services.agent.core import get_agent
from services.agent.prompt import RagAnswer
from services.vectorstore import db, TABLE_NAME

router = APIRouter()


def _thread_id(req: QueryRequest) -> str:
    # mono-PDF V0: doc_id:session_id si doc_id fourni, sinon session_id seul
    if req.doc_id:
        return f"{req.doc_id}:{req.session_id}"
    return req.session_id


def _build_citations(structured: RagAnswer | None, thread_id: str, doc_id: str | None) -> list[Citation]:
    if not structured or not structured.cited:
        return []
    # On récupère les sources depuis LanceDB pour les ids cités
    # hybrid_search a déjà stocké sources JSON; on les relit par id
    try:
        table = db.open_table(TABLE_NAME)
        # LanceDB filter: id IN (...)
        # Fallback: to_pandas puis filter si filter non supporté
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
            text = str(row.get("text", ""))[:500]
            page = int(row.get("page", 1))
            sources_raw = row.get("sources", "[]")
            bbox = None
            try:
                import json as _json

                sources = _json.loads(sources_raw) if isinstance(sources_raw, str) else sources_raw
                if sources and isinstance(sources, list) and sources[0].get("bbox"):
                    b = sources[0]["bbox"][0]
                    # BBox {x,y,w,h}
                    bbox = [float(b.get("x", 0)), float(b.get("y", 0)), float(b.get("w", 0)), float(b.get("h", 0))]
            except Exception:
                bbox = None
            locator = Locator(
                type="pdf",
                doc_id=doc_id or str(row.get("doc_id", "")),
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


@router.post("", response_class=EventSourceResponse)
async def query_sse(req: QueryRequest) -> AsyncIterable[ServerSentEvent]:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question vide")

    thread_id = _thread_id(req)
    config = {"configurable": {"thread_id": thread_id}}
    agent = get_agent()
    start = time.time()

    # meta first
    yield ServerSentEvent(
        event="meta",
        data={"mode": "rag", "thread_id": thread_id, "doc_id": req.doc_id},
    )

    # Stream agent — on utilise astream_events v2 pour capter thinking + tokens
    full_answer = ""
    structured: RagAnswer | None = None

    try:
        async for ev in agent.astream_events(
            {"messages": [{"role": "user", "content": req.question}]},
            config,
            version="v2",
        ):
            kind = ev.get("event")
            data = ev.get("data", {})
            name = ev.get("name", "")

            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                # thinking channel (Ollama gpt-oss: thinking in additional_kwargs)
                reasoning = None
                if hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_content")
                if reasoning:
                    yield ServerSentEvent(event="thinking", data={"delta": str(reasoning)})
                # token channel
                content = getattr(chunk, "content", None)
                if content:
                    if isinstance(content, str):
                        full_answer += content
                        yield ServerSentEvent(event="token", data={"delta": content})
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                delta = block.get("text", "")
                                full_answer += delta
                                yield ServerSentEvent(event="token", data={"delta": delta})

            elif kind == "on_tool_start":
                if "retrieve" in name.lower() or ev.get("name") == "retrieve":
                    yield ServerSentEvent(event="tool", data={"tool": "retrieve", "status": "start"})

            elif kind == "on_tool_end":
                yield ServerSentEvent(event="tool", data={"tool": "retrieve", "status": "end"})

            await asyncio.sleep(0)

    except Exception as e:
        yield ServerSentEvent(event="error", data={"code": "AGENT_ERROR", "message": str(e)})
        return

    # Après streaming, récupérer structured_response depuis le checkpoint
    try:
        state = agent.get_state(config)
        sr = state.values.get("structured_response")
        if isinstance(sr, RagAnswer):
            structured = sr
        elif isinstance(sr, dict) and "answer" in sr:
            try:
                structured = RagAnswer(**sr)
            except Exception:
                structured = None
        if structured is None and full_answer.strip().startswith("{"):
            try:
                parsed = json.loads(full_answer)
                if "answer" in parsed:
                    structured = RagAnswer(**parsed)
            except Exception:
                pass
    except Exception:
        structured = None

    citations = _build_citations(structured, thread_id, req.doc_id)

    if structured is None and full_answer:
        structured = RagAnswer(answer=full_answer, cited=[], confidence="low", refusal=None)

    yield ServerSentEvent(
        event="citations",
        data={"citations": [c.model_dump() for c in citations]},
    )

    latency = int((time.time() - start) * 1000)
    yield ServerSentEvent(
        event="done",
        data={
            "answer": structured.answer if structured else full_answer,
            "cited": structured.cited if structured else [],
            "confidence": structured.confidence if structured else "low",
            "refusal": structured.refusal if structured else None,
            "latencyMs": latency,
            "provider": "ollama",
            "model": "gpt-oss:20b-cloud",
        },
    )
