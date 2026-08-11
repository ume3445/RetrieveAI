from __future__ import annotations


def recall_at_k(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str], k: int) -> float:
    """1.0 if any relevant chunk_id appears in the top-k retrieved, else 0.0."""
    if not relevant_chunk_ids:
        return 0.0
    top_k = retrieved_chunk_ids[:k]
    return 1.0 if any(cid in relevant_chunk_ids for cid in top_k) else 0.0


def mean_reciprocal_rank(retrieved_chunk_ids: list[str], relevant_chunk_ids: set[str]) -> float:
    """1/rank of the first relevant chunk found, 0.0 if none found."""
    for rank, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0
