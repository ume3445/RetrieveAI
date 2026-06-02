#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(name="retrieve-ai", add_completion=False)
console = Console()


def _spinner(label: str) -> Progress:
    return Progress(SpinnerColumn(), TextColumn(label), console=console, transient=True)


@app.command()
def ingest(
    pdf: Path = typer.Argument(..., help="PDF to ingest."),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already stored."),
) -> None:
    """Chunk, embed, and store a PDF in the vector database."""
    from src.ingestion import ingest as _ingest

    with _spinner("Ingesting..."):
        collection, added = _ingest(str(pdf), force=force)

    if added == 0:
        console.print(
            f"[yellow]Already ingested.[/] {collection.count()} chunks in '{collection.name}'. "
            "Pass --force to re-ingest."
        )
    else:
        console.print(f"[green]Done.[/] Stored {added} chunks in '{collection.name}'.")


@app.command()
def ask(
    pdf: Path = typer.Argument(..., help="PDF to query (must already be ingested)."),
    question: str = typer.Argument(..., help="Question to answer."),
    top_k: int = typer.Option(0, "--top-k", "-k", help="Chunks to retrieve (0 = config default)."),
    ingest_first: bool = typer.Option(False, "--ingest", "-i", help="Ingest before asking."),
) -> None:
    """Ask a question against an ingested PDF."""
    from src.ingestion import ingest as _ingest, load_collection
    from src.workflow import run

    if ingest_first:
        with _spinner("Ingesting..."):
            _ingest(str(pdf))

    try:
        collection = load_collection(str(pdf))
    except Exception:
        console.print("[red]Collection not found.[/] Run `ingest` first, or pass --ingest.")
        raise typer.Exit(1)

    with _spinner("Thinking..."):
        answer = run(collection, question, top_k=top_k or None)

    console.print(Panel(Markdown(answer.text), title="[bold cyan]Answer[/]", border_style="cyan"))

    if answer.citations:
        console.print("\n[bold]Sources:[/]")
        for i, cit in enumerate(answer.citations, 1):
            console.print(f"  [dim]{i}.[/] Page {cit['page']} — [italic]{cit['excerpt'][:120]}[/]")


if __name__ == "__main__":
    app()
