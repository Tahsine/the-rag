from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

logging.getLogger("langgraph.checkpoint.serde.jsonplus").setLevel(logging.ERROR)

from schemas import Citation
from services.agent.core import get_agent
from services.agent.prompt import RagAnswer
from services.citations import build_citations
from services.pipeline import PipelineError, UpstreamError, index_pdf
from services.vectorstore import list_documents

app = typer.Typer(help="Folio CLI — RAG local sur les PDF indexés dans LanceDB.", no_args_is_help=True)
console = Console()

EXIT_COMMANDS = {"exit", "quit", "q", "/exit", "/quit"}


def _fail(message: str) -> None:
    console.print(Panel(message, title="Erreur", border_style="red"))
    raise typer.Exit(code=1)


def _ensure_pdf(path: Path) -> None:
    if not path.exists():
        _fail(f"Fichier introuvable: {path}")
    if path.stat().st_size < 4:
        _fail("Fichier trop petit pour être un PDF")
    with path.open("rb") as f:
        if f.read(4) != b"%PDF":
            _fail(f"{path.name} n'est pas un PDF (magic bytes %PDF- manquant)")


def _thread_id(doc_id: str | None, base: str) -> str:
    return f"{doc_id}:{base}" if doc_id else base


def _ensure_doc(doc_id: str | None) -> None:
    if not doc_id:
        return
    documents = list_documents()
    if not any(str(document.get("doc_id")) == doc_id for document in documents):
        _fail(f"doc_id inconnu: {doc_id} (utilisez `python -m cli docs`)")


def _confidence_style(confidence: str) -> str:
    return {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "red")


def _write(raw: str, style: str | None = None) -> None:
    console.print(Text(raw, style=style), end="")
    console.file.flush()


def _print_citations(citations: list[Citation]) -> None:
    if not citations:
        return
    table = Table(title="Citations", title_justify="left")
    table.add_column("id", style="cyan")
    table.add_column("doc", style="dim")
    table.add_column("page", style="cyan")
    table.add_column("snippet", ratio=3)
    for citation in citations:
        table.add_row(
            citation.citationId,
            citation.locator.doc_id,
            str(citation.locator.page),
            citation.snippet.replace("\n", " ")[:180],
        )
    console.print(table)


def _extract_structured(agent, config: dict, full_answer: str) -> RagAnswer | None:
    structured: RagAnswer | None = None
    try:
        state = agent.get_state(config)
        candidate = state.values.get("structured_response")
        if isinstance(candidate, RagAnswer):
            structured = candidate
        elif isinstance(candidate, dict) and "answer" in candidate:
            structured = RagAnswer(**candidate)
    except Exception:
        structured = None
    if structured is None and full_answer.strip().startswith("{"):
        try:
            parsed = json.loads(full_answer)
            if "answer" in parsed:
                structured = RagAnswer(**parsed)
        except Exception:
            structured = None
    return structured


async def _stream_answer(question: str, thread_id: str, doc_id: str | None, show_thinking: bool) -> tuple[str, RagAnswer | None]:
    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id, "doc_id": doc_id}}
    full_answer = ""
    console.print(Text("› ", style="bold cyan") + Text(question, style="bold"))
    try:
        async for ev in agent.astream_events(
            {"messages": [{"role": "user", "content": question}]},
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
                reasoning = None
                if hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_content")
                if reasoning and show_thinking:
                    _write(str(reasoning), "dim italic")
                content = getattr(chunk, "content", None)
                if content:
                    if isinstance(content, str):
                        full_answer += content
                        _write(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                delta = block.get("text", "")
                                full_answer += delta
                                _write(delta)
            elif kind == "on_tool_start" and "retrieve" in name.lower():
                _write("\nrécherche…\n", "dim")
            elif kind == "on_tool_end" and "retrieve" in name.lower():
                _write("contexte récupéré\n", "dim green")
    except Exception as e:
        console.print()
        _fail(f"Agent error: {e}")
    console.print()

    structured = _extract_structured(agent, config, full_answer)
    if structured is None and full_answer:
        structured = RagAnswer(answer=full_answer, cited=[], confidence="low", refusal=None)
    return full_answer, structured


async def _ask(question: str, thread_id: str, doc_id: str | None, show_thinking: bool) -> None:
    streamed, structured = await _stream_answer(question, thread_id, doc_id, show_thinking)
    if not structured:
        return
    streamed_clean = streamed.strip()
    answer_clean = structured.answer.strip()
    if answer_clean and (not streamed_clean or answer_clean not in streamed_clean):
        console.print(Markdown(structured.answer))
    console.print(Text("confiance: ", style="bold") + Text(structured.confidence, style=_confidence_style(structured.confidence)))
    if structured.refusal:
        console.print(Panel(structured.refusal, title="Refus", border_style="red"))
    _print_citations(build_citations(structured, doc_id))


@app.command()
def add(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Chemin vers le PDF à indexer."),
    doc_id: str | None = typer.Option(None, "--doc-id", help="doc_id stable (sinon UUID)."),
) -> None:
    _ensure_pdf(pdf)
    base_doc_id = doc_id or uuid.uuid4().hex
    with console.status("préparation…", spinner="dots") as status:
        def on_stage(stage_name: str) -> None:
            status.update(status=f"{stage_name}…")

        try:
            result = index_pdf(pdf, doc_id=base_doc_id, on_stage=on_stage)
        except UpstreamError as e:
            _fail(f"Erreur upstream: {e}")
        except PipelineError as e:
            _fail(f"Pipeline: {e}")
    console.print(
        Text("✓ ", style="bold green")
        + Text(f"doc_id: {result.doc_id}", style="cyan")
        + Text(f"  pages: {result.pages}  chunks: {result.chunks}")
    )


@app.command()
def docs() -> None:
    documents = list_documents()
    if not documents:
        console.print("Aucun document indexé. Utilisez `folio add <pdf>`.")
        return
    table = Table(title="Documents indexés", title_justify="left")
    table.add_column("doc_id", style="cyan")
    table.add_column("pages", justify="right")
    table.add_column("chunks", justify="right")
    for document in documents:
        table.add_row(str(document["doc_id"]), str(document["pages"]), str(document["chunks"]))
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question pour l'agent RAG."),
    doc_id: str | None = typer.Option(None, "--doc", "--doc-id", help="Limite la recherche à ce doc_id (sinon corpus global)."),
    thread: str | None = typer.Option(None, "--thread", help="Réutilise un thread de conversation."),
    thinking: bool = typer.Option(False, "--thinking", help="Affiche le raisonnement du modèle."),
) -> None:
    _ensure_doc(doc_id)
    base_thread = thread or f"ask-{uuid.uuid4().hex[:8]}"
    try:
        asyncio.run(_ask(question, _thread_id(doc_id, base_thread), doc_id, thinking))
    except KeyboardInterrupt:
        console.print()
        raise typer.Exit(code=130)


@app.command()
def chat(
    doc_id: str | None = typer.Option(None, "--doc", "--doc-id", help="Limite la recherche à ce doc_id (sinon corpus global)."),
    thread: str = typer.Option("cli-chat", "--thread", help="Nom du thread de conversation."),
    thinking: bool = typer.Option(False, "--thinking", help="Affiche le raisonnement du modèle."),
) -> None:
    _ensure_doc(doc_id)
    console.print(Panel("Folio chat — tapez `exit` pour quitter.", border_style="blue"))
    with asyncio.Runner() as runner:
        while True:
            try:
                question = Prompt.ask(Text("question", style="bold cyan"))
            except (KeyboardInterrupt, EOFError):
                console.print()
                break
            if not question.strip():
                continue
            if question.strip().lower() in EXIT_COMMANDS:
                break
            try:
                runner.run(_ask(question, _thread_id(doc_id, thread), doc_id, thinking))
            except KeyboardInterrupt:
                console.print()
            console.print()


if __name__ == "__main__":
    app()
