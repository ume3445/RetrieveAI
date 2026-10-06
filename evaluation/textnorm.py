"""Text normalization used only for *scoring* (span matching).

Deliberately independent of src/ so the same scorer works on every version of
the pipeline, including ones whose chunks still contain raw PDF artifacts.
"""
from __future__ import annotations

import re
import unicodedata

_HYPHEN_BREAK = re.compile(r"-[ \t]*\n\s*")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC (expands ligatures like 'ﬁ'), rejoin words hyphenated across line
    breaks, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text)
    text = _HYPHEN_BREAK.sub("", text)
    return _WS.sub(" ", text).strip().lower()


def contains_span(chunk_text: str, gold_span: str) -> bool:
    return normalize(gold_span) in normalize(chunk_text)
