# RetrieveAI

Ask questions against a PDF using a local RAG pipeline backed by ChromaDB and OpenAI.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your OPENAI_API_KEY
```

## Usage

```bash
# Ingest a PDF (idempotent; use --force to re-embed)
python cli.py ingest report.pdf

# Ask a question
python cli.py ask report.pdf "What are the key findings?"

# Ingest and ask in one step
python cli.py ask report.pdf "Who wrote this?" --ingest

# Override number of retrieved chunks
python cli.py ask report.pdf "..." --top-k 8
```

## Evaluation

```bash
python eval.py report.pdf
```

The default probes are generic (main topic, audience, conclusions). For domain-specific tests, create `eval_pairs.py`:

```python
QA_PAIRS = [
    {"question": "What year was the company founded?", "keywords": ["1994"]},
    {"question": "Who is the CEO?", "keywords": ["chief executive"]},
]
```

Each test passes if at least one keyword appears in the retrieved chunks and in the generated answer. Exits with code 1 if any test fails.

## Configuration

All settings are read from `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector store location |

## Architecture

```
src/ingestion.py   PDF -> chunks -> OpenAI embeddings -> ChromaDB
src/retrieval.py   query embed -> ChromaDB ANN -> exact cosine rerank
src/generation.py  context prompt -> OpenAI chat with function calling -> Answer
src/workflow.py    LangGraph graph: retrieve -> rerank -> generate
```

The rerank step in `workflow.py` is a score-floor filter on top of the cosine rerank already done in `retrieval.py` — it drops any chunk scoring below half the top result, which keeps the generation context from being diluted by marginal matches.

Generation uses `tool_choice: required` to force the model to call `return_answer(answer, citations)`, so the output is always structured JSON rather than free text that needs parsing.

Collections in ChromaDB are keyed by `md5(absolute_path)`, so the same PDF maps to the same collection regardless of working directory.
