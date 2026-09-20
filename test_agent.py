"""Standalone test for RAG agent (etape.md:10).

Usage:
  OLLAMA_API_KEY=... python test_agent.py              # Cloud gpt-oss:20b-cloud
  # or local:
  OLLAMA_BASE_URL=http://localhost:11434 OLLAMA_MODEL=gpt-oss:20b python test_agent.py

Tests:
  Q1: What is a List-item?         -> doit citer [id] p.4-6 (corpus test_vectorstore.py)
  Q2: And how does it relate to Text? -> même thread_id, utilise mémoire
  Q3: hors corpus                  -> doit refuser honnêtement (cited=[] ou refusal)
"""

import os
import re
import sys
from pathlib import Path

# ensure backend is on path when run as `python test_agent.py`
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from services.agent.core import get_agent
from services.agent.prompt import RagAnswer

THREAD_ID = "test-doclaynet-kdd22-cli"

_id_re = re.compile(r"\[(\d+)\]")


def _parse_ids(text: str) -> set[int]:
    return {int(m.group(1)) for m in _id_re.finditer(text or "")}


def _print_result(label: str, result: dict):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    structured: RagAnswer | None = result.get("structured_response")
    messages = result.get("messages", [])
    # last AIMessage may have reasoning_content
    for m in messages[-4:]:
        role = getattr(m, "type", m.__class__.__name__)
        content = getattr(m, "content", "") or ""
        tool_calls = getattr(m, "tool_calls", None)
        if content:
            # truncate
            preview = content[:800].replace("\n", " ")
            print(f"[{role}] {preview}")
        if tool_calls:
            print(f"  tool_calls: {tool_calls}")
        if isinstance(m, dict) and m.get("tool_call_id"):
            print(f"[tool] {str(m.get('content',''))[:400]}")

    if structured:
        print(f"\n→ structured_response: answer={structured.answer[:400]!r}")
        print(f"  cited={structured.cited} confidence={structured.confidence} refusal={structured.refusal}")
    else:
        print("\n→ No structured_response (model may have answered without tool)")


def main():
    # quick sanity: check LanceDB has data
    from services.vectorstore import hybrid_search
    from services.embeddings import embed_texts

    ollama_key_set = bool(
        os.environ.get("OLLAMA_API_KEY")
        or os.environ.get("OLLAMA_CLOUD_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )
    print(f"OLLAMA_API_KEY set: {ollama_key_set} (fallback LLM_API_KEY supported)")
    print(f"OLLAMA_BASE_URL: {os.environ.get('OLLAMA_BASE_URL', 'https://ollama.com')}")
    print(f"OLLAMA_MODEL: {os.environ.get('OLLAMA_MODEL') or os.environ.get('LLM_MODEL') or 'gpt-oss:20b-cloud'}")

    try:
        hits = hybrid_search(embed_texts(["List-item"])[0], "List-item", limit=3)
        print(f"LanceDB sanity: {len(hits)} hits (expected >0 if indexed). Sample: {[h['id'] for h in hits[:3]]}")
        if not hits:
            print("⚠️  LanceDB empty — run `python test_vectorstore.py` first to index docs_test pages 4-6.")
    except Exception as e:
        print(f"⚠️  LanceDB check failed: {e}")
        print("   Run `python test_vectorstore.py` first.")

    agent = get_agent()
    config = {"configurable": {"thread_id": THREAD_ID}}

    # Q1
    q1 = "What is a List-item?"
    print(f"\n>>> Q1: {q1} (thread {THREAD_ID})")
    r1 = agent.invoke({"messages": [{"role": "user", "content": q1}]}, config)
    _print_result("Q1 result", r1)
    s1: RagAnswer | None = r1.get("structured_response")
    if s1:
        # collect retrieved ids from ToolMessages
        retrieved = set()
        for m in r1.get("messages", []):
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and "[" in content and ("(doc:" in content or "(p." in content):
                retrieved.update(_parse_ids(content))
        if s1.cited:
            invalid = [c for c in s1.cited if c not in retrieved]
            assert not invalid, f"Q1 hallucinated cited IDs {invalid} not in retrieved {retrieved}"
            print(f"✅ Q1 cited {s1.cited} ⊆ retrieved {retrieved}")
        else:
            print(f"⚠️ Q1 has no citations, refusal={s1.refusal}")
    else:
        print("⚠️ Q1 no structured_response — check model ToolStrategy support")

    # Q2 — same thread, multi-turn
    q2 = "And how does it relate to Text?"
    print(f"\n>>> Q2: {q2} (same thread)")
    r2 = agent.invoke({"messages": [{"role": "user", "content": q2}]}, config)
    _print_result("Q2 result", r2)
    # verify history preserved
    state = agent.get_state(config)
    print(f"\n[checkpointer] state messages count: {len(state.values.get('messages', []))}")
    print(f"  next: {state.next}")

    # Q3 — hors corpus
    q3 = "Qui a gagné la coupe du monde 2030 ?"
    print(f"\n>>> Q3: {q3} (hors corpus, same thread)")
    r3 = agent.invoke({"messages": [{"role": "user", "content": q3}]}, config)
    _print_result("Q3 result", r3)
    s3: RagAnswer | None = r3.get("structured_response")
    if s3:
        if s3.refusal or not s3.cited:
            print(f"✅ Q3 correctly refused or empty cited (refusal={s3.refusal!r})")
        else:
            print(f"⚠️ Q3 expected refusal/empty cited but got cited={s3.cited}")

    # Q4 — new thread isolation
    q4 = "What is a List-item?"
    new_thread = "other-doc-test"
    print(f"\n>>> Q4 (new thread {new_thread}) isolation check")
    r4 = agent.invoke({"messages": [{"role": "user", "content": q4}]}, {"configurable": {"thread_id": new_thread}})
    _print_result("Q4 result (new thread)", r4)
    s4 = agent.get_state({"configurable": {"thread_id": new_thread}})
    print(f"[new thread] messages: {len(s4.values.get('messages', []))}")

    print("\n✅ test_agent done — if Q1 cited non-empty and Q3 refused, étape 4 is OK.")


if __name__ == "__main__":
    main()
