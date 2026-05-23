"""Reciprocal Rank Fusion."""

from __future__ import annotations


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using reciprocal rank fusion.

    Args:
        rankings: List of ranked (id, score) lists.
        k: RRF constant (default 60).

    Returns:
        Merged list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
