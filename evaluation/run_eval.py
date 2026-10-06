#!/usr/bin/env python3
"""Retrieval eval: 100 questions, scored by whether a retrieved chunk contains
the gold answer span (so the score survives re-chunking).

Version-agnostic: it imports whatever src/ is on the path and only uses names
that exist in every version (retrieve, hybrid_retrieve, _get_bm25_index,
_tokenize, workflow._node_retrieve/_node_rerank), so the exact same harness
scores the baseline commit and the fixed one.

Modes (all called with top_k=5, each query searches only its own paper):
  random   analytic expectation of picking 5 chunks uniformly at random
  bm25     BM25 index alone
  dense    retrieve()        (embedding search + exact cosine rescore)
  hybrid   hybrid_retrieve()
  app      what the app actually hands the LLM: workflow retrieve + rerank nodes

Usage:
  python -m evaluation.run_eval --label after [--reingest] [--offline]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .embed_cache import EmbeddingCache
from .textnorm import contains_span

HERE = Path(__file__).parent
K = 5
MODES = ["random", "bm25", "dense", "hybrid", "app"]


def first_hit_rank(texts: list[str], span: str) -> int | None:
    for rank, t in enumerate(texts, 1):
        if contains_span(t, span):
            return rank
    return None


def random_expectation(n_chunks: int, n_gold: int, k: int) -> tuple[float, dict[int, float]]:
    """P(first gold chunk lands at rank r) for r=1..k when sampling k of n without replacement."""
    p_rank: dict[int, float] = {}
    p_none_before = 1.0
    for r in range(1, k + 1):
        remaining = n_chunks - (r - 1)
        p_gold_here = n_gold / remaining if remaining > 0 else 0.0
        p_rank[r] = p_none_before * p_gold_here
        p_none_before *= 1 - p_gold_here
    return 1 - p_none_before, p_rank


def git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--queries", default=str(HERE / "queries.jsonl"))
    ap.add_argument("--cache", default=str(HERE / ".cache" / "embeddings.sqlite"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--reingest", action="store_true", help="force re-ingest every paper with the current src/ first")
    ap.add_argument("--offline", action="store_true", help="fail instead of calling the API on a cache miss")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from src import config, ingestion, retrieval, workflow  # noqa: E402

    cache = EmbeddingCache(args.cache, offline=args.offline)
    cache._api_key = config.OPENAI_API_KEY or None
    Cached = cache.client_class()
    ingestion.OpenAI = Cached
    retrieval.OpenAI = Cached

    queries = [json.loads(l) for l in open(args.queries)]
    papers = sorted({q["paper"] for q in queries})
    papers_dir = Path(args.papers_dir).resolve()

    collections, chunk_stats = {}, {}
    for p in papers:
        pdf = str(papers_dir / p)
        if args.reingest:
            col, added = ingestion.ingest(pdf, force=True)
            print(f"re-ingested {p}: {added} chunks")
        else:
            col = ingestion.load_collection(pdf)
        collections[p] = col
        docs = col.get(include=["documents"])["documents"]
        lens = [len(d) for d in docs]
        chunk_stats[p] = {
            "chunks": len(docs),
            "max_chars": max(lens),
            "over_limit": sum(l > config.CHUNK_SIZE for l in lens),
            "_docs": docs,
        }

    # Embed every query up front (one batched call) so per-query latency below
    # measures retrieval, not the OpenAI round trip.
    cache.get_many(config.EMBED_MODEL, [q["question"] for q in queries])

    per_query = []
    latency: dict[str, list[float]] = {m: [] for m in MODES if m != "random"}
    for q in queries:
        col, span = collections[q["paper"]], q["gold_span"]
        docs = chunk_stats[q["paper"]]["_docs"]
        n_gold = sum(contains_span(d, span) for d in docs)
        row = {k: q[k] for k in ("qid", "paper", "question", "gold_span")}
        row["gold_chunks_in_index"] = n_gold
        _, p_rank = random_expectation(len(docs), n_gold, K)
        row["random"] = {"p_rank": p_rank}

        runs = {
            "bm25": lambda: _bm25(retrieval, col, q["question"]),
            "dense": lambda: [c.text for c in retrieval.retrieve(col, q["question"], top_k=K)],
            "hybrid": lambda: [c.text for c in retrieval.hybrid_retrieve(col, q["question"], top_k=K)],
            "app": lambda: _app(workflow, col, q["question"]),
        }
        for mode, fn in runs.items():
            t0 = time.perf_counter()
            texts = fn()
            latency[mode].append((time.perf_counter() - t0) * 1000)
            rank = first_hit_rank(texts, span)
            row[mode] = {"rank": rank, "n_returned": len(texts), "top": [t[:240] for t in texts]}
        per_query.append(row)

    summary = {}
    n = len(per_query)
    for mode in MODES:
        if mode == "random":
            hit = {k: sum(sum(r["random"]["p_rank"][j] for j in range(1, k + 1)) for r in per_query) / n for k in (1, 3, 5)}
            mrr = sum(sum(p / j for j, p in r["random"]["p_rank"].items()) for r in per_query) / n
        else:
            ranks = [r[mode]["rank"] for r in per_query]
            hit = {k: sum(1 for x in ranks if x is not None and x <= k) / n for k in (1, 3, 5)}
            mrr = sum(1 / x for x in ranks if x is not None and x <= K) / n
        summary[mode] = {"hit@1": hit[1], "hit@3": hit[3], "hit@5": hit[5], "mrr@5": mrr}
        if mode in latency:
            summary[mode]["latency_ms_p50"] = statistics.median(latency[mode])
            summary[mode]["latency_ms_p95"] = pct(latency[mode], 0.95)

    for p in chunk_stats:
        chunk_stats[p].pop("_docs")
    result = {
        "label": args.label,
        "git_sha": git_sha(Path.cwd()),
        "n_queries": n,
        "k": K,
        "config": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP, "embed_model": config.EMBED_MODEL},
        "chunk_stats": chunk_stats,
        "summary": summary,
        "embedding_api": {"calls": cache.api_calls, "texts": cache.api_texts},
        "per_query": per_query,
    }
    out = Path(args.out or HERE / "results" / f"{args.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False))

    print(f"\n{args.label} @ {result['git_sha']}  ({n} queries, k={K})")
    print(f"{'mode':8} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'mrr@5':>6} {'p50 ms':>7}")
    for m, s in summary.items():
        lat = f"{s['latency_ms_p50']:7.1f}" if "latency_ms_p50" in s else f"{'-':>7}"
        print(f"{m:8} {s['hit@1']:6.2f} {s['hit@3']:6.2f} {s['hit@5']:6.2f} {s['mrr@5']:6.2f} {lat}")
    print(f"embedding API: {cache.api_calls} calls, {cache.api_texts} texts. Wrote {out}")


def _bm25(retrieval, col, question: str) -> list[str]:
    idx = retrieval._get_bm25_index(col)
    scores = idx.bm25.get_scores(retrieval._tokenize(question))
    order = sorted(range(len(idx.ids)), key=lambda i: scores[i], reverse=True)[:K]
    return [idx.documents[idx.ids[i]] for i in order]


def _app(workflow, col, question: str) -> list[str]:
    state = {"query": question, "collection": col, "top_k": K, "chunks": [], "answer": None}
    state = workflow._node_rerank(workflow._node_retrieve(state))
    return [c.text for c in state["chunks"]]


if __name__ == "__main__":
    main()
