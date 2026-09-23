from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import sys
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

from config import CITATION_IMAGES_DIR, LANCEDB_PATH, REPO_ROOT, STORED_PDF_DIR
from schemas import Citation
from services.agent.core import get_agent
from services.agent.prompt import RagAnswer
from services.citations import build_citations
from services.citation_images import CitationImageResult, save_citation_images
from services.document_store import has_source_pdf, store_source_pdf
from services.pipeline import PipelineError, UpstreamError, index_pdf
from services.vectorstore import list_documents

app = typer.Typer(help="just-rag — RAG local sur les PDF indexés dans LanceDB.", no_args_is_help=True)
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
        _fail(f"doc_id inconnu: {doc_id} (utilisez `python -m just_rag docs`)")


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


def _print_citation_images(result: CitationImageResult) -> None:
    if result.images:
        console.print(Text("\nImages citations sauvegardées:", style="bold green"))
        for image in result.images:
            console.print(Text(f"  {image.path}", style="green") + Text(f"  ({image.bbox_count} boxes)", style="dim"))
    for warning in result.warnings:
        console.print(Text(warning, style="yellow"))


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


async def _ask(
    question: str,
    thread_id: str,
    doc_id: str | None,
    show_thinking: bool,
    save_images: bool = True,
) -> None:
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
    if save_images:
        image_result = save_citation_images(structured, doc_id, answer_id=uuid.uuid4().hex[:10])
        _print_citation_images(image_result)


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
    if has_source_pdf(result.doc_id):
        console.print(Text("PDF source conservé pour les images de citations.", style="dim"))


@app.command()
def docs() -> None:
    documents = list_documents()
    if not documents:
        console.print("Aucun document indexé. Utilisez `just-rag add <pdf>`.")
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
    save_images: bool = typer.Option(True, "--save-images", "--no-save-images", help="Sauvegarde les PNG des pages citées."),
) -> None:
    _ensure_doc(doc_id)
    base_thread = thread or f"ask-{uuid.uuid4().hex[:8]}"
    try:
        asyncio.run(_ask(question, _thread_id(doc_id, base_thread), doc_id, thinking, save_images))
    except KeyboardInterrupt:
        console.print()
        raise typer.Exit(code=130)


@app.command()
def chat(
    doc_id: str | None = typer.Option(None, "--doc", "--doc-id", help="Limite la recherche à ce doc_id (sinon corpus global)."),
    thread: str = typer.Option("cli-chat", "--thread", help="Nom du thread de conversation."),
    thinking: bool = typer.Option(False, "--thinking", help="Affiche le raisonnement du modèle."),
    save_images: bool = typer.Option(True, "--save-images", "--no-save-images", help="Sauvegarde les PNG des pages citées."),
) -> None:
    _ensure_doc(doc_id)
    console.print(Panel("JustRag chat — tapez `exit` pour quitter.", border_style="blue"))
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
                runner.run(_ask(question, _thread_id(doc_id, thread), doc_id, thinking, save_images))
            except KeyboardInterrupt:
                console.print()
            console.print()


@app.command("attach-pdf")
def attach_pdf(
    doc_id: str = typer.Argument(..., help="doc_id déjà indexé."),
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Chemin vers le PDF source."),
) -> None:
    """Attache/rattache un PDF source pour un document déjà indexé."""
    _ensure_pdf(pdf)
    _ensure_doc(doc_id)
    try:
        destination = store_source_pdf(pdf, doc_id)
    except Exception as e:
        _fail(f"Stockage du PDF source: {e}")
    console.print(Text("✓ ", style="bold green") + Text(f"PDF source attaché pour {doc_id}: {destination}", style="cyan"))


@app.command()
def doctor() -> None:
    """Vérifie les dépendances, clés API, dossiers de données et LanceDB."""
    table = Table(title="JustRag doctor", title_justify="left")
    table.add_column("check", style="cyan")
    table.add_column("status", ratio=3)

    warnings: list[str] = []
    failures: list[str] = []

    table.add_row("Python", f"{sys.version.split()[0]}")

    dependencies = [
        ("typer", "typer"),
        ("rich", "rich"),
        ("lancedb", "lancedb"),
        ("pymupdf", "pymupdf"),
        ("pillow", "PIL"),
        ("llama-cloud", "llama_cloud"),
        ("google-genai", "google.genai"),
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("langchain-ollama", "langchain_ollama"),
        ("python-dotenv", "dotenv"),
    ]
    for label, module in dependencies:
        spec = importlib.util.find_spec(module)
        if spec is None:
            failures.append(f"dépendance manquante: {label}")
            table.add_row(label, "manquant", style="red")
        else:
            table.add_row(label, "OK")

    google_key = bool(os.getenv("GOOGLE_API_KEY"))
    llama_key = bool(os.getenv("LLAMA_CLOUD_API_KEY"))
    ollama_key = bool(
        os.getenv("OLLAMA_API_KEY")
        or os.getenv("OLLAMA_CLOUD_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    table.add_row("GOOGLE_API_KEY", "OK" if google_key else "manquante", style="green" if google_key else "red")
    table.add_row("LLAMA_CLOUD_API_KEY", "OK" if llama_key else "manquante", style="green" if llama_key else "red")
    table.add_row("OLLAMA_API_KEY/LLM_API_KEY", "OK" if ollama_key else "manquante", style="green" if ollama_key else "red")
    if not google_key:
        failures.append("GOOGLE_API_KEY manquante (embeddings Gemini)")
    if not llama_key:
        failures.append("LLAMA_CLOUD_API_KEY manquante (parsing LlamaParse)")
    if not ollama_key:
        failures.append("OLLAMA_API_KEY ou LLM_API_KEY manquante (agent Ollama)")

    lancedb_path = Path(LANCEDB_PATH)
    lancedb_exists = lancedb_path.exists()
    try:
        documents = list_documents()
        table.add_row(
            "LanceDB",
            f"{LANCEDB_PATH} — {len(documents)} doc(s)" if lancedb_exists else f"{LANCEDB_PATH} — pas encore créé",
        )
    except Exception as e:
        failures.append(f"LanceDB illisible: {e}")
        table.add_row("LanceDB", f"erreur: {e}", style="red")

    legacy_lancedb_dirs = [
        path for path in REPO_ROOT.glob("*_lancedb")
        if path.resolve() != lancedb_path.resolve()
    ]
    if legacy_lancedb_dirs:
        for legacy in legacy_lancedb_dirs:
            warnings.append(
                f"Autre dossier LanceDB détecté: {legacy}. "
                f"Set JUST_RAG_LANCEDB_PATH pour l'utiliser ou déplacez ses données vers {lancedb_path}."
            )

    for label, directory in (("stored_pdfs", STORED_PDF_DIR), ("citation_images", CITATION_IMAGES_DIR)):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            writable = os.access(directory, os.W_OK)
            table.add_row(label, str(directory), style="green" if writable else "red")
            if not writable:
                failures.append(f"dossier non accessible en écriture: {directory}")
        except Exception as e:
            failures.append(f"dossier {label} non créable: {e}")
            table.add_row(label, f"erreur: {e}", style="red")

    console.print(table)
    if warnings:
        console.print(Panel("\n".join(warnings), title="Avertissements", border_style="yellow"))
    if failures:
        console.print(Panel("\n".join(failures), title="Problèmes bloquants", border_style="red"))
        raise typer.Exit(code=1)
    console.print(Text("OK — backend prêt pour l'usage local.", style="bold green"))


if __name__ == "__main__":
    app()
