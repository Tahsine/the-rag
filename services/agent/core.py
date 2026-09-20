import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware import ToolCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama

from .prompt import SYSTEM_PROMPT, RagAnswer
from .tools import retrieve
from .middleware.citation import CitationValidationMiddleware

load_dotenv()

# singletons lazy
_agent = None
_checkpointer = None


def _build_llm() -> ChatOllama:
    """Construit le LLM Ollama Cloud gpt-oss:20b-cloud avec reasoning low.

    Env requis:
      OLLAMA_API_KEY  (ou OLLAMA_HOST si local)
      Optionnel: OLLAMA_BASE_URL (défaut https://ollama.com)
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")
    # Support both naming conventions: OLLAMA_API_KEY (new) and LLM_API_KEY (legacy .env)
    api_key = (
        os.environ.get("OLLAMA_API_KEY")
        or os.environ.get("OLLAMA_CLOUD_API_KEY")
        or os.environ.get("LLM_API_KEY")
    )

    # ChatOllama gère l'auth via client_kwargs headers ou via base_url avec userinfo
    client_kwargs: dict = {}
    if api_key:
        client_kwargs = {"headers": {"Authorization": f"Bearer {api_key}"}}

    # reasoning="low" : CoT courte pour RAG factuel, latence minimale
    # num_ctx=8192 : 6k retrieval + prompt + answer = confortable, évite KV cache inutile
    # temperature=0 : factuel, pas de créativité
    # Model name: support LLM_MODEL fallback, normalize gpt-oss-20b:cloud -> gpt-oss:20b-cloud
    raw_model = os.environ.get("OLLAMA_MODEL") or os.environ.get("LLM_MODEL") or "gpt-oss:20b-cloud"
    # normalize dash/colon variants
    if raw_model == "gpt-oss-20b:cloud":
        raw_model = "gpt-oss:20b-cloud"
    llm = ChatOllama(
        model=raw_model,
        base_url=base_url,
        temperature=0,
        reasoning="low",
        num_ctx=8192,
        validate_model_on_init=False,
        client_kwargs=client_kwargs,
    )
    return llm


def get_checkpointer():
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = InMemorySaver()
    return _checkpointer


def get_agent():
    """Retourne l'agent compilé (singleton)."""
    global _agent
    if _agent is not None:
        return _agent

    llm = _build_llm()
    checkpointer = get_checkpointer()

    _agent = create_agent(
        model=llm,
        tools=[retrieve],
        system_prompt=SYSTEM_PROMPT,
        # ToolStrategy obligatoire en Ollama Cloud (ProviderStrategy utilise format/strict ignoré)
        response_format=ToolStrategy(RagAnswer),
        checkpointer=checkpointer,
        middleware=[
            ToolCallLimitMiddleware(thread_limit=12, run_limit=8, exit_behavior="continue"),
            CitationValidationMiddleware(),
        ],
        name="folio-rag-agent",
    )
    return _agent


__all__ = ["get_agent", "get_checkpointer"]
