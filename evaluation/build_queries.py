#!/usr/bin/env python3
"""Rebuild evaluation/queries.jsonl from the *baseline* index (commit tagged in
scripts/reproduce_eval.sh as BASELINE_REF, chunked with the original chunker).

How the 100 queries were chosen, so nobody has to take it on faith:
  1. Load every chunk of every paper from ChromaDB (1,051 chunks, 5 papers).
  2. Drop obvious boilerplate with a cheap heuristic (very short chunks,
     reference lists, author blocks, mostly-numeric tables).
  3. Shuffle each paper's remaining chunks with a fixed seed and walk the
     shuffled order, skipping chunks rejected on manual review
     (annotations.REJECTED), until the paper's quota is filled. Quotas are
     proportional to each paper's chunk count (largest-remainder rounding).
  4. Every selected chunk must have a hand-checked question + verbatim gold
     span in annotations.QUESTIONS; the script asserts the two sets match, so
     the questions can't have been cherry-picked after the fact.

Usage:  CHROMA_PERSIST_DIR=./chroma_db_baseline python -m evaluation.build_queries
"""
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import chromadb

from .annotations import QUESTIONS, REJECTED
from .textnorm import normalize

SEED = 20261006
N_QUERIES = 100
OUT = Path(__file__).parent / "queries.jsonl"


def is_boilerplate(text: str) -> bool:
    n = normalize(text)
    if len(n) < 200:
        return True
    years = len(re.findall(r"\b(19|20)\d\d\b", n))
    if years >= 4 and any(w in n for w in ("proceedings", "arxiv", "in advances", "conference", "journal")):
        return True
    names = len(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text))
    if re.search(r"@\w", n) or "equal contribution" in n or ("university" in n and names > 6):
        return True
    words = re.findall(r"[a-z]{3,}", n)
    return len(words) / max(1, len(n.split())) < 0.45


def quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    n = sum(sizes.values())
    exact = {k: v * total / n for k, v in sizes.items()}
    q = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - q[k], reverse=True)[: total - sum(q.values())]:
        q[k] += 1
    return q


def main() -> None:
    client = chromadb.PersistentClient(path=os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db_baseline"))
    chunks: dict[str, list[dict]] = {}
    for col in client.list_collections():
        r = col.get(include=["documents", "metadatas"])
        rows = [
            {"id": i, "text": d, "page": m["page"], "paper": Path(m["source"]).name}
            for i, d, m in zip(r["ids"], r["documents"], r["metadatas"])
        ]
        rows.sort(key=lambda x: int(x["id"].rsplit("-", 1)[1]))
        chunks[col.name] = rows

    quota = quotas({k: len(v) for k, v in chunks.items()}, N_QUERIES)
    rng = random.Random(SEED)
    selected: list[dict] = []
    for name in sorted(chunks):
        pool = [c for c in chunks[name] if not is_boilerplate(c["text"])]
        rng.shuffle(pool)
        picked = [c for c in pool if c["id"] not in REJECTED][: quota[name]]
        selected += picked

    annotated = {cid: (q, s) for cid, q, s in QUESTIONS}
    assert {c["id"] for c in selected} == set(annotated), "sampled chunks != annotated chunks"

    with OUT.open("w") as f:
        for i, c in enumerate(selected, 1):
            question, span = annotated[c["id"]]
            assert normalize(span) in normalize(c["text"]), f"gold span not in chunk {c['id']}"
            f.write(json.dumps({
                "qid": f"q{i:03d}", "paper": c["paper"], "question": question, "gold_span": span,
                "source_chunk_id": c["id"], "page": c["page"], "chunk_text": c["text"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(selected)} queries to {OUT} (quotas: {quota})")


if __name__ == "__main__":
    main()
