#!/usr/bin/env python3
"""Compare evaluation/results/baseline.json and after.json -> evaluation/RESULTS.md"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).parent
MODES = ["random", "bm25", "dense", "hybrid", "app"]
LABELS = {
    "random": "Random (expected)",
    "bm25": "BM25 only",
    "dense": "Dense only",
    "hybrid": "Hybrid",
    "app": "**App path** (what the LLM sees)",
}


def hits(res: dict, mode: str, k: int = 5) -> dict[str, int]:
    return {r["qid"]: int(r[mode]["rank"] is not None and r[mode]["rank"] <= k) for r in res["per_query"]}


def paired_bootstrap(a: dict[str, int], b: dict[str, int], n: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    qids = sorted(a)
    diffs = [b[q] - a[q] for q in qids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return sum(diffs) / len(diffs), boots[int(0.025 * n)], boots[int(0.975 * n)]


def main() -> None:
    base = json.loads((HERE / "results" / "baseline.json").read_text())
    after = json.loads((HERE / "results" / "after.json").read_text())
    n = after["n_queries"]
    out = [
        "# Retrieval eval results",
        "",
        f"{n} questions over 5 papers, top_k=5, each query searches only its own paper. A hit means a "
        "retrieved chunk contains the question's gold answer span (whitespace/ligature-insensitive). "
        "See the README for how the questions were sampled.",
        "",
        f"Baseline: `{base['git_sha']}` (original pipeline). After: `{after['git_sha']}`.",
        "",
        "| Mode | Hit@1 | Hit@5 | MRR@5 | | Hit@1 | Hit@5 | MRR@5 |",
        "|---|---|---|---|---|---|---|---|",
        "| | **baseline** | | | | **after fixes** | | |",
    ]
    for m in MODES:
        b, a = base["summary"][m], after["summary"][m]
        out.append(
            f"| {LABELS[m]} | {b['hit@1']:.2f} | {b['hit@5']:.2f} | {b['mrr@5']:.2f} | "
            f"| {a['hit@1']:.2f} | {a['hit@5']:.2f} | {a['mrr@5']:.2f} |"
        )

    out += ["", "## Change on the app path and the hybrid retriever (paired bootstrap, 10k resamples)", ""]
    for m in ("app", "hybrid", "bm25"):
        d, lo, hi = paired_bootstrap(hits(base, m), hits(after, m))
        out.append(f"- {LABELS[m].replace('**', '')}: Hit@5 {base['summary'][m]['hit@5']:.2f} -> "
                   f"{after['summary'][m]['hit@5']:.2f} (diff {d:+.2f}, 95% CI {lo:+.2f} to {hi:+.2f})")
    d1, lo1, hi1 = paired_bootstrap(hits(base, "app", 1), hits(after, "app", 1))
    out.append(f"- App path Hit@1: {base['summary']['app']['hit@1']:.2f} -> {after['summary']['app']['hit@1']:.2f} "
               f"(diff {d1:+.2f}, 95% CI {lo1:+.2f} to {hi1:+.2f})")

    out += ["", "## Index", "", "| | chunks | max chars | over CHUNK_SIZE |", "|---|---|---|---|"]
    for label, res in (("baseline", base), ("after", after)):
        cs = res["chunk_stats"].values()
        out.append(f"| {label} | {sum(c['chunks'] for c in cs)} | {max(c['max_chars'] for c in cs)} | "
                   f"{sum(c['over_limit'] for c in cs)} |")

    out += ["", "## Retrieval latency after fixes (query embedding excluded, cached)", "",
            "| Mode | p50 ms | p95 ms |", "|---|---|---|"]
    for m in ("bm25", "dense", "hybrid", "app"):
        s = after["summary"][m]
        out.append(f"| {m} | {s['latency_ms_p50']:.1f} | {s['latency_ms_p95']:.1f} |")

    out += ["", "## Per paper, app path Hit@5", "", "| Paper | n | baseline | after |", "|---|---|---|---|"]
    for p in sorted({r["paper"] for r in after["per_query"]}):
        rows_b = [r for r in base["per_query"] if r["paper"] == p]
        rows_a = [r for r in after["per_query"] if r["paper"] == p]
        hb = sum(r["app"]["rank"] is not None for r in rows_b) / len(rows_b)
        ha = sum(r["app"]["rank"] is not None for r in rows_a) / len(rows_a)
        out.append(f"| {p} | {len(rows_a)} | {hb:.2f} | {ha:.2f} |")

    misses = [r for r in after["per_query"] if r["app"]["rank"] is None]
    out += ["", f"## App-path misses after fixes ({len(misses)})", ""]
    for r in misses:
        top = r["app"]["top"][0].replace("\n", " ")[:160] if r["app"]["top"] else "(nothing returned)"
        out.append(f"- **{r['qid']}** ({r['paper']}) {r['question']}  \n  gold: `{r['gold_span']}`  \n  top-1: {top}")

    (HERE / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:20]))
    print(f"\nWrote {HERE / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
