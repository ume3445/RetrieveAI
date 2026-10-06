from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterator

import chromadb
from openai import OpenAI
from pypdf import PdfReader

from . import config

# Sentence boundary: punctuation + space(s) followed by a capital letter, or
# punctuation followed by newline(s). Deliberately simple — no nltk/spacy.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z])|(?<=[.!?])\n+")

# A lowercase word broken across a line with a hyphen ("op-\ntimal").
_HYPHEN_BREAK_RE = re.compile(r"(?<=[a-z])-\n(?=[a-z])")


def _clean_text(text: str) -> str:
    """
    Undo the two most common pypdf extraction artifacts before chunking:
    typographic ligatures ("ﬁ", "ﬂ") are expanded by NFKC, and words split
    across lines with a hyphen are rejoined. Left as-is, "classiﬁcation"
    tokenizes to ["classi", "cation"] and can never match a query for
    "classification" in BM25.
    """
    text = unicodedata.normalize("NFKC", text)
    return _HYPHEN_BREAK_RE.sub("", text)


def _pdf_pages(path: str) -> Iterator[tuple[int, str]]:
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = _clean_text(page.extract_text() or "")
        if text.strip():
            yield i + 1, text


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def _carry_over(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """Trailing 1-2 sentences to seed the next chunk, targeting ~overlap chars."""
    if not sentences or overlap <= 0:
        return [], 0
    carry = [sentences[-1]]
    carry_len = len(carry[0])
    if len(sentences) >= 2 and carry_len < overlap:
        carry.insert(0, sentences[-2])
        carry_len += 1 + len(sentences[-2])
    return carry, carry_len


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Greedily pack sentences into chunks up to `size` chars, never splitting a
    sentence across chunks. A single sentence longer than `size` (equations,
    tables) falls back to a hard character split. The last 1-2 sentences of
    each chunk carry over into the next, targeting ~`overlap` chars, but only
    when they fit: no chunk ever exceeds `size`.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if len(sentence) > size:
            if current:
                chunks.append(" ".join(current))
            start = 0
            while start < len(sentence):
                chunks.append(sentence[start : start + size])
                start += size - overlap
            current, current_len = [], 0
            continue

        added_len = len(sentence) + (1 if current else 0)
        if current and current_len + added_len > size:
            chunks.append(" ".join(current))
            current, current_len = _carry_over(current, overlap)
            # Overlap is best-effort, the size limit is not: drop carried
            # sentences (oldest first) until the incoming sentence fits.
            while current and current_len + 1 + len(sentence) > size:
                dropped = current.pop(0)
                current_len = current_len - len(dropped) - 1 if current else 0
            added_len = len(sentence) + (1 if current else 0)

        current.append(sentence)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return [c.strip() for c in chunks if c.strip()]


def _embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def _collection_name(pdf_path: str) -> str:
    # ChromaDB names: 3-63 chars, alphanumeric + hyphens
    safe = "".join(c if c.isalnum() else "-" for c in Path(pdf_path).stem)[:50]
    digest = hashlib.md5(pdf_path.encode()).hexdigest()[:8]
    return f"{safe}-{digest}"


def ingest(pdf_path: str, force: bool = False) -> tuple[chromadb.Collection, int]:
    """
    Chunk, embed, and store a PDF. Returns (collection, chunks_added).
    Returns 0 for chunks_added if the collection already exists and force=False.
    """
    pdf_path = str(Path(pdf_path).resolve())
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    chroma = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    name = _collection_name(pdf_path)

    if not force:
        existing = [c.name for c in chroma.list_collections()]
        if name in existing:
            return chroma.get_collection(name), 0

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    chunks: list[str] = []
    metadata: list[dict] = []
    for page_num, page_text in _pdf_pages(pdf_path):
        for chunk in _chunk_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunks.append(chunk)
            metadata.append({"source": pdf_path, "page": page_num})

    if not chunks:
        raise ValueError("No extractable text found in PDF.")

    # Batch to keep per-request payload small; API ceiling is much higher
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), 100):
        embeddings.extend(_embed(client, chunks[i : i + 100]))

    try:
        chroma.delete_collection(name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = chroma.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"{name}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
    )

    return collection, len(chunks)


def load_collection(pdf_path: str) -> chromadb.Collection:
    pdf_path = str(Path(pdf_path).resolve())
    chroma = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    return chroma.get_collection(_collection_name(pdf_path))
