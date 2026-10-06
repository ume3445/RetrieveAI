#!/usr/bin/env bash
# Reproduce evaluation/RESULTS.md: score the original pipeline and the fixed
# one with the same 100 queries and the same harness.
#
#   ./scripts/reproduce_eval.sh
#
# Needs OPENAI_API_KEY in .env and the five PDFs in papers/. First run embeds
# ~100 queries + ~1,000 re-chunked passages (well under 1 cent); every rerun
# after that is served from evaluation/.cache and costs nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${PYTHON:-python3}"
BASELINE_REF="${BASELINE_REF:-d89162d}"   # last commit before the eval and fixes

set -a; [ -f .env ] && . ./.env; set +a
CACHE="$ROOT/evaluation/.cache/embeddings.sqlite"

# 1. Baseline: original src/ at BASELINE_REF against a snapshot of the index it built.
if [ ! -d chroma_db_baseline ]; then
  echo "Snapshotting current index to chroma_db_baseline/ (built by the original chunker)"
  cp -R chroma_db chroma_db_baseline
fi
rm -rf .eval_worktree && git worktree prune
git worktree add --detach .eval_worktree "$BASELINE_REF" >/dev/null
rm -rf .eval_worktree/evaluation && cp -R evaluation .eval_worktree/evaluation
(cd .eval_worktree && CHROMA_PERSIST_DIR="$ROOT/chroma_db_baseline" "$PY" -m evaluation.run_eval \
  --label baseline --papers-dir "$ROOT/papers" --cache "$CACHE" --out "$ROOT/evaluation/results/baseline.json")
git worktree remove --force .eval_worktree

# 2. After: current src/, re-ingesting every paper with the fixed chunker into the app's index.
CHROMA_PERSIST_DIR="$ROOT/chroma_db" "$PY" -m evaluation.run_eval \
  --label after --reingest --papers-dir "$ROOT/papers" --cache "$CACHE"

# 3. Compare.
"$PY" -m evaluation.report
