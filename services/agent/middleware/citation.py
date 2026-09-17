import json
import re
from typing import Annotated, Any, TypedDict

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.message import add_messages


_ID_RE = re.compile(r"\[(\d+)\]")


def _parse_retrieved_ids(tool_content: str) -> set[int]:
    """Extract [id] from retrieve tool output. Handles NO_RESULTS gracefully."""
    if not tool_content or tool_content.startswith("NO_RESULTS") or tool_content.startswith("ERROR"):
        return set()
    ids: set[int] = set()
    for m in _ID_RE.finditer(tool_content):
        try:
            ids.add(int(m.group(1)))
        except ValueError:
            continue
    return ids


class CitationState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    retrieved_ids: list[int]
    structured_response: Any  # RagAnswer


class CitationValidationMiddleware(AgentMiddleware):
    """Post-validation anti-hallucination pour citations RAG.

    - capture les IDs réellement retournés par retrieve via wrap_tool_call
    - filtre cited ∉ retrieved_ids dans after_model
    """

    state_schema = CitationState

    def wrap_tool_call(self, request, handler):
        """Intercepte l'exécution de retrieve pour mémoriser les IDs (sync)."""
        result = handler(request)
        if request.tool_call.get("name") == "retrieve":
            tool_content = ""
            if isinstance(result, ToolMessage):
                tool_content = str(result.content)
            elif isinstance(result, dict):
                tool_content = str(result.get("content", ""))
            else:
                tool_content = str(result)
            ids = _parse_retrieved_ids(tool_content)
            if isinstance(result, ToolMessage):
                result.additional_kwargs = {
                    **getattr(result, "additional_kwargs", {}),
                    "retrieved_ids": list(ids),
                }
                return result
        return result

    async def awrap_tool_call(self, request, handler):
        """Async version for astream/astream_events (required in async context)."""
        result = await handler(request)
        if request.tool_call.get("name") == "retrieve":
            tool_content = ""
            if isinstance(result, ToolMessage):
                tool_content = str(result.content)
            elif isinstance(result, dict):
                tool_content = str(result.get("content", ""))
            else:
                tool_content = str(result)
            ids = _parse_retrieved_ids(tool_content)
            if isinstance(result, ToolMessage):
                result.additional_kwargs = {
                    **getattr(result, "additional_kwargs", {}),
                    "retrieved_ids": list(ids),
                }
                return result
        return result

    async def aafter_model(self, state, runtime):
        # Delegate to sync implementation for async path
        return self.after_model(state, runtime)

    def after_model(self, state, runtime):
        """Filtre cited invalides après génération de RagAnswer.

        Fallback : si structured_response est None mais le dernier AIMessage
        contient un JSON RagAnswer (cas Ollama Cloud ToolStrategy qui rend
        du JSON en content au lieu de tool_call), on le parse pour créer
        structured_response — sinon test_agent voit 'No structured_response'.
        """
        structured = state.get("structured_response")
        # Fallback parsing si le LLM a écrit du JSON en content au lieu d'appeler l'outil
        if structured is None:
            messages = state.get("messages", [])
            if messages:
                last = messages[-1]
                if isinstance(last, AIMessage):
                    content = last.content
                    if isinstance(content, str) and content.strip().startswith("{") and '"answer"' in content:
                        try:
                            data = json.loads(content)
                            # mapping confiance FR -> EN
                            conf = data.get("confidence", "low")
                            if conf in ("Élevée", "élevée", "haute"):
                                conf = "high"
                            elif conf in ("Moyenne", "moyenne"):
                                conf = "medium"
                            elif conf in ("Faible", "faible", "low", "Low"):
                                conf = "low"
                            # normalise cited
                            cited = data.get("cited", [])
                            if isinstance(cited, str):
                                cited = []
                            # Lazy import pour éviter cycle
                            from services.agent.prompt import RagAnswer

                            parsed = RagAnswer(
                                answer=str(data.get("answer", "")),
                                cited=[int(x) for x in cited if str(x).isdigit() or isinstance(x, int)],
                                confidence=conf if conf in ("high", "medium", "low") else "low",
                                refusal=data.get("refusal"),
                            )
                            # continue to validation below with parsed as structured
                            structured = parsed
                        except Exception:
                            return None
            if structured is None:
                return None

        retrieved_ids: set[int] = set(state.get("retrieved_ids") or [])
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, ToolMessage):
                if "retrieved_ids" in getattr(msg, "additional_kwargs", {}):
                    retrieved_ids.update(msg.additional_kwargs["retrieved_ids"])
                elif isinstance(msg.content, str) and "[" in msg.content:
                    retrieved_ids.update(_parse_retrieved_ids(msg.content))

        if not retrieved_ids:
            return None

        try:
            cited = list(getattr(structured, "cited", []))
        except Exception:
            return None

        if not cited:
            return None

        valid = [c for c in cited if c in retrieved_ids]
        if len(valid) != len(cited):
            invalid = [c for c in cited if c not in retrieved_ids]
            try:
                if not valid and not getattr(structured, "refusal", None):
                    new_structured = structured.model_copy(
                        update={
                            "cited": [],
                            "confidence": "low",
                            "refusal": f"Citations invalides filtrées {invalid} — réponse non sourcée.",
                        }
                    )
                else:
                    new_structured = structured.model_copy(update={"cited": valid})
            except Exception:
                return None
            return {"structured_response": new_structured}

        return None
