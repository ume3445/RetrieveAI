"""Drop-in stand-in for `openai.OpenAI` that caches embeddings on disk.

The eval swaps this in for the `OpenAI` name inside src.ingestion and
src.retrieval, so the pipeline code under test runs unmodified. Every vector
is fetched from the API once and replayed afterwards, which makes reruns free,
deterministic, and lets latency numbers exclude the network round trip.
"""
from __future__ import annotations

import hashlib
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI as _RealOpenAI


@dataclass
class _Item:
    embedding: list[float]


@dataclass
class _Response:
    data: list[_Item]


class EmbeddingCache:
    def __init__(self, path: str | Path, offline: bool = False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self.offline = offline
        self.api_calls = 0
        self.api_texts = 0
        self._real: _RealOpenAI | None = None
        self._api_key: str | None = None

    @staticmethod
    def _key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> list[list[float]]:
        keys = [self._key(model, t) for t in texts]
        found: dict[str, list[float]] = {}
        for k in set(keys):
            row = self.db.execute("SELECT vec FROM emb WHERE key = ?", (k,)).fetchone()
            if row:
                blob = row[0]
                found[k] = list(struct.unpack(f"{len(blob) // 4}f", blob))
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in found))
        if missing:
            if self.offline:
                raise RuntimeError(f"{len(missing)} embeddings not cached and --offline is set")
            if self._real is None:
                self._real = _RealOpenAI(api_key=self._api_key)
            for i in range(0, len(missing), 100):
                batch = missing[i : i + 100]
                resp = self._real.embeddings.create(model=model, input=batch)
                self.api_calls += 1
                self.api_texts += len(batch)
                for t, item in zip(batch, resp.data):
                    k = self._key(model, t)
                    found[k] = item.embedding
                    self.db.execute(
                        "INSERT OR REPLACE INTO emb VALUES (?, ?)",
                        (k, struct.pack(f"{len(item.embedding)}f", *item.embedding)),
                    )
            self.db.commit()
        return [found[k] for k in keys]

    def client_class(self):
        cache = self

        class _Embeddings:
            def create(self, model: str, input: list[str]):  # noqa: A002 (mirrors OpenAI signature)
                return _Response([_Item(v) for v in cache.get_many(model, list(input))])

        class CachedOpenAI:
            def __init__(self, api_key: str | None = None, **_):
                cache._api_key = cache._api_key or api_key
                self.embeddings = _Embeddings()

        return CachedOpenAI
