#!/usr/bin/env python3
"""
IR evaluation harness for RetrieveAI retrieval quality.

Loads (question, pdf_path, relevant_pages) triples from EVAL_PAIRS in
eval_pairs.py and scores retrieval with recall@5, recall@10, and MRR. Each
retrieved chunk's page number is used as a chunk_id proxy since chunks don't
yet carry finer-grained IDs.

Usage:
    python eval.py                       Run the IR eval harness over EVAL_PAIRS
    python eval.py --smoke-test doc.pdf  Old-style keyword pass/fail smoke test
                                          against a single ad-hoc PDF
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from src.metrics import mean_reciprocal_rank, recall_at_k

console = Console()

try:
    from eval_pairs import EVAL_PAIRS  # type: ignore
except ImportError:
    EVAL_PAIRS = []

SMOKE_QA_PAIRS = [
    {"question": "What is the main topic of the document?", "keywords": []},
    {"question": "Who is the intended audience?", "keywords": []},
    {"question": "What conclusions does the document make?", "keywords": []},
]


def evaluate_retrieval() -> None:
    from src.ingestion import ingest
    from src.retrieval import retrieve

    if not EVAL_PAIRS:
        console.print(
            "[yellow]EVAL_PAIRS is empty.[/] Add question/pdf_path/relevant_pages "
            "entries to eval_pairs.py to run the IR eval.\n"
        )
        return

    console.print(f"\n[bold]RetrieveAI IR Eval[/] — {len(EVAL_PAIRS)} question(s)\n")

    table = Table(
        "Q#", "Question", "PDF", "Recall@5", "Recall@10", "MRR",
        show_lines=True, header_style="bold magenta",
    )

    recall5_scores: list[float] = []
    recall10_scores: list[float] = []
    mrr_scores: list[float] = []

    for i, pair in enumerate(EVAL_PAIRS, 1):
        question = pair["question"]
        pdf_path = pair["pdf_path"]
        relevant_chunk_ids = {str(p) for p in pair["relevant_pages"]}

        with console.status(f"Q{i}: ingesting + retrieving..."):
            collection, _ = ingest(pdf_path)
            chunks = retrieve(collection, question, top_k=10)

        retrieved_chunk_ids = [str(c.page) for c in chunks]

        r5 = recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, 5)
        r10 = recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, 10)
        mrr = mean_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids)

        recall5_scores.append(r5)
        recall10_scores.append(r10)
        mrr_scores.append(mrr)

        table.add_row(
            str(i),
            question[:50] + ("..." if len(question) > 50 else ""),
            pdf_path.rsplit("/", 1)[-1],
            f"{r5:.2f}",
            f"{r10:.2f}",
            f"{mrr:.2f}",
        )

    n = len(EVAL_PAIRS)
    table.add_row(
        "", "[bold]Average[/]", "",
        f"[bold]{sum(recall5_scores) / n:.2f}[/]",
        f"[bold]{sum(recall10_scores) / n:.2f}[/]",
        f"[bold]{sum(mrr_scores) / n:.2f}[/]",
    )

    console.print(table)
    console.print()


def _matches(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return bool(text.strip())
    low = text.lower()
    return any(kw.lower() in low for kw in keywords)


def _mark(ok: bool) -> str:
    return "[green]PASS[/]" if ok else "[red]FAIL[/]"


def smoke_test(pdf_path: str) -> None:
    from src.ingestion import ingest, load_collection
    from src.retrieval import retrieve
    from src.workflow import run

    console.print(f"\n[bold]RetrieveAI Smoke Test[/] — {pdf_path}\n")

    with console.status("Ingesting..."):
        ingest(pdf_path)
    collection = load_collection(pdf_path)

    table = Table("Q#", "Question", "Retrieval", "Answer", "Overall",
                  show_lines=True, header_style="bold magenta")

    passed = 0
    for i, qa in enumerate(SMOKE_QA_PAIRS, 1):
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
    summary = "[green]ALL PASS[/]" if passed == len(SMOKE_QA_PAIRS) else "[yellow]SOME FAILED[/]"
    console.print(f"\n[bold]Result:[/] {passed}/{len(SMOKE_QA_PAIRS)} — {summary}\n")

    if passed < len(SMOKE_QA_PAIRS):
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="RetrieveAI eval harness.")
    parser.add_argument(
        "--smoke-test",
        metavar="PDF",
        default=None,
        help="Run the old-style keyword pass/fail smoke test against a single ad-hoc PDF.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args.smoke_test)
    else:
        evaluate_retrieval()


if __name__ == "__main__":
    main()
