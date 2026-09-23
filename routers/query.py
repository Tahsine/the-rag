import asyncio
import json
import time
import uuid
from typing import AsyncIterable

from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent

from schemas import QueryRequest
from services.agent.core import get_agent
from services.agent.prompt import RagAnswer
from services.citations import build_citations
from services.citation_images import save_citation_images

router = APIRouter()


def _thread_id(req: QueryRequest) -> str:
    # doc_id optionnel: doc_id:session_id si fourni, sinon session_id seul (corpus global)
    if req.doc_id:
        return f"{req.doc_id}:{req.session_id}"
    return req.session_id


@router.post("", response_class=EventSourceResponse)
async def query_sse(req: QueryRequest) -> AsyncIterable[ServerSentEvent]:
    thread_id = _thread_id(req)
    config = {"configurable": {"thread_id": thread_id, "doc_id": req.doc_id}}
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

    citations = build_citations(structured, req.doc_id)

    if structured is None and full_answer:
        structured = RagAnswer(answer=full_answer, cited=[], confidence="low", refusal=None)

    yield ServerSentEvent(
        event="citations",
        data={"citations": [c.model_dump() for c in citations]},
    )

    if req.save_citation_images:
        image_result = save_citation_images(structured, req.doc_id, answer_id=uuid.uuid4().hex[:10])
        yield ServerSentEvent(
            event="images",
            data={
                "images": [image.model_dump() for image in image_result.images],
                "warnings": image_result.warnings,
            },
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
