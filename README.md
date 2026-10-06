# RetrieveAI

Ask questions about research papers with a RAG pipeline: PDF ingestion, hybrid retrieval (dense embeddings + BM25 fused with Reciprocal Rank Fusion), and cited answers from an LLM, orchestrated with LangGraph. Retrieval quality is measured on a 100-question benchmark built for this repo; see [Evaluation](#evaluation).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

The eval corpus is five arXiv papers placed in `papers/` (PDFs are gitignored): ResNet `1512.03385v1`, Transformer `1706.03762v7`, BERT `1810.04805v2`, RAG `2005.11401v4`, LoRA `2106.09685v2`.

## Usage

```bash
python cli.py ingest papers/1706.03762v7.pdf            # idempotent; --force to re-embed
python cli.py ask papers/1706.03762v7.pdf "How many attention heads does the base model use?"
python cli.py ask paper.pdf "..." --ingest --top-k 8
```

## How it works

```
src/ingestion.py   PDF -> clean text -> sentence-aware chunks -> OpenAI embeddings -> ChromaDB
src/retrieval.py   dense (HNSW + exact cosine rescore) and BM25 rankings, fused with RRF
src/generation.py  context prompt -> OpenAI chat with forced function call -> answer + citations
src/workflow.py    LangGraph graph: retrieve -> generate
```

**Ingestion.** pypdf output is NFKC-normalized (expands ligatures like `ﬁ`) and words hyphenated across line breaks are rejoined before chunking. Pages are split into sentences and packed greedily into chunks of at most `CHUNK_SIZE` characters, carrying the last one or two sentences into the next chunk as overlap when they fit. The size limit is hard: overlap is dropped before the limit is exceeded.

**Retrieval.** Dense retrieval pulls `3 x top_k` candidates from ChromaDB's HNSW index and re-ranks them by exact cosine similarity. BM25 scores every chunk in the paper and keeps its top `3 x top_k`. The two rankings are merged with Reciprocal Rank Fusion (k=60): rank-based, so BM25's unbounded scores and the tightly clustered cosine scores never need to share a scale, and a chunk only one retriever found still gets full credit for its rank.

**Generation.** The model is forced to call a `return_answer(answer, citations)` tool, so output is always structured JSON with page-level citations.

## Evaluation

```bash
./scripts/reproduce_eval.sh      # baseline vs. current, writes evaluation/RESULTS.md
pytest                           # unit tests: chunking limits, text cleaning, RRF, wiring, scoring
```

**Benchmark.** 100 questions over the five papers. `evaluation/build_queries.py` samples chunks with a fixed seed, proportional to each paper's chunk count, skipping boilerplate (reference lists, figure residue). Each sampled chunk got one natural-language question written by an LLM and a gold answer span copied verbatim from the chunk. The script asserts the annotated chunks are exactly the seeded sample, so questions can't be cherry-picked after seeing results.

**Scoring.** A retrieval counts as a hit if any returned chunk contains the gold span (ligature-, hyphenation-, and whitespace-insensitive). Because scoring uses the answer text rather than chunk IDs or page numbers, the same benchmark stays valid when chunking changes. Reported metrics are Hit@1/3/5 and MRR@5 at `top_k=5`, for BM25 alone, dense alone, hybrid, and the exact path the app hands to the LLM, plus the analytic expectation of random retrieval (Hit@5 = 0.04).

**Reproducibility.** The harness swaps in a disk-cached embeddings client, so the pipeline code runs unmodified, reruns are free and deterministic, and latency numbers exclude the API round trip. The same harness scores both the original commit (in a git worktree, against a snapshot of its index) and the current code, and `report.py` adds paired-bootstrap 95% confidence intervals for the change.

**Known limitation.** Questions were generated from the chunks that answer them, so they share more vocabulary with their source than real user questions would. That likely flatters BM25. Single-paper retrieval also makes the task easier than searching a whole library.

Results: [evaluation/RESULTS.md](evaluation/RESULTS.md)

## Configuration

All settings are read from `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | | Required |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Target overlap between chunks (best-effort) |
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store location |

Collections in ChromaDB are keyed by the PDF's absolute path, one collection per paper.
