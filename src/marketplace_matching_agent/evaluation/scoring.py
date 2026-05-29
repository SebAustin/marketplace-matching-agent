"""Match scoring helpers."""

from __future__ import annotations


def score_match(query: str, item_text: str) -> dict[str, object]:
    """Score lexical overlap between query and item text."""
    overlap = len(set(query.lower().split()) & set(item_text.lower().split()))
    return {"score": overlap / max(len(query.split()), 1)}
