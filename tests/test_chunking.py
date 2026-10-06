import random
from pathlib import Path

import pytest

from src.ingestion import _chunk_text, _pdf_pages

SIZE, OVERLAP = 512, 64


def _random_text(rng: random.Random, n_sentences: int) -> str:
    sentences = []
    for _ in range(n_sentences):
        words = [rng.choice(["alpha", "beta", "gamma", "residual", "attention", "x"]) for _ in range(rng.randint(1, 90))]
        sentences.append(" ".join(words).capitalize() + rng.choice([".", "?", "!"]))
    return rng.choice([" ", "\n"]).join(sentences)


@pytest.mark.parametrize("seed", range(200))
def test_no_chunk_exceeds_size(seed):
    rng = random.Random(seed)
    for chunk in _chunk_text(_random_text(rng, rng.randint(1, 40)), SIZE, OVERLAP):
        assert len(chunk) <= SIZE


def test_overlap_carries_last_sentence_when_it_fits():
    s = [f"Sentence number {i} is here." for i in range(40)]
    chunks = _chunk_text(" ".join(s), SIZE, OVERLAP)
    assert len(chunks) > 1
    last_sentence = chunks[0].split(". ")[-1]
    assert last_sentence in chunks[1]


def test_no_text_is_lost():
    rng = random.Random(7)
    text = _random_text(rng, 30)
    joined = " ".join(_chunk_text(text, SIZE, OVERLAP))
    for word in set(text.split()):
        assert word in joined


PAPERS = sorted(Path(__file__).resolve().parent.parent.joinpath("papers").glob("*.pdf"))


@pytest.mark.skipif(not PAPERS, reason="papers/ not present")
def test_real_papers_respect_size():
    for pdf in PAPERS:
        for _, page in _pdf_pages(str(pdf)):
            assert all(len(c) <= SIZE for c in _chunk_text(page, SIZE, OVERLAP))
