#!/usr/bin/env python3
"""
Run retrieval + answer accuracy checks against a PDF.

Each entry in QA_PAIRS needs a question and a list of keywords; at least one
keyword must appear in the retrieved chunks AND in the generated answer.
Empty keywords list passes if any non-empty text is returned.

Override QA_PAIRS with domain-specific pairs by creating eval_pairs.py:
    QA_PAIRS = [{"question": "...", "keywords": ["expected", "terms"]}, ...]
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

console = Console()

QA_PAIRS = [
    {"question": "What is the main topic of the document?", "keywords": []},
    {"question": "Who is the intended audience?", "keywords": []},
    {"question": "What conclusions does the document make?", "keywords": []},
]

try:
    from eval_pairs import QA_PAIRS  # type: ignore
except ImportError:
    pass


def _matches(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return bool(text.strip())
    low = text.lower()
    return any(kw.lower() in low for kw in keywords)


def _mark(ok: bool) -> str:
    return "[green]PASS[/]" if ok else "[red]FAIL[/]"


def evaluate(pdf_path: str) -> None:
    from src.ingestion import ingest, load_collection
    from src.retrieval import retrieve
    from src.workflow import run

    console.print(f"\n[bold]RetrieveAI Eval[/] — {pdf_path}\n")

    with console.status("Ingesting..."):
        ingest(pdf_path)
    collection = load_collection(pdf_path)

    table = Table("Q#", "Question", "Retrieval", "Answer", "Overall",
                  show_lines=True, header_style="bold magenta")

    passed = 0
    for i, qa in enumerate(QA_PAIRS, 1):
        question = qa["question"]
        keywords = qa.get("keywords", [])

        chunks = retrieve(collection, question)
        retrieval_ok = _matches(" ".join(c.text for c in chunks), keywords)

        answer = run(collection, question)
        answer_ok = _matches(answer.text, keywords)

        overall = retrieval_ok and answer_ok
        if overall:
            passed += 1

        table.add_row(
            str(i),
            question[:60] + ("..." if len(question) > 60 else ""),
            _mark(retrieval_ok),
            _mark(answer_ok),
            _mark(overall),
        )

        console.print(f"\n[bold]Q{i}:[/] {question}")
        console.print(f"[dim]{answer.text[:300]}[/]")
        if answer.citations:
            pages = [c["page"] for c in answer.citations]
            console.print(f"[dim]cited pages: {pages}[/]")

    console.print()
    console.print(table)
    summary = "[green]ALL PASS[/]" if passed == len(QA_PAIRS) else "[yellow]SOME FAILED[/]"
    console.print(f"\n[bold]Result:[/] {passed}/{len(QA_PAIRS)} — {summary}\n")

    if passed < len(QA_PAIRS):
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[red]Usage:[/] python eval.py path/to/doc.pdf")
        sys.exit(1)
    evaluate(sys.argv[1])
