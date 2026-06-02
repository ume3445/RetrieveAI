from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator

import chromadb
from openai import OpenAI
from pypdf import PdfReader

from . import config


def _pdf_pages(path: str) -> Iterator[tuple[int, str]]:
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            yield i + 1, text


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
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
