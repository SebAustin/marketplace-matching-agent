"""GoodMatch@k LLM-as-judge."""

from __future__ import annotations

from marketplace_matching_agent.config import get_settings


async def judge_one(query: str, item_text: str, counterparty_text: str) -> bool:
    """Judge whether item is a good match for query."""
    q_tokens = set(query.lower().split())
    item_tokens = set(item_text.lower().split())
    cp_tokens = set(counterparty_text.lower().split())
    overlap = len(q_tokens & item_tokens & cp_tokens)
    return overlap >= 1 or len(q_tokens & item_tokens) >= 2


async def goodmatch_at_k(results: list[object], k: int = 10) -> float:
    """Compute GoodMatch@k over trajectory results."""
    _ = get_settings()
    if not results:
        return 0.0
    scores: list[float] = []
    for result in results:
        ranked_ids = getattr(result, "ranked_ids", [])
        gold_ids = getattr(result, "gold_ids", [])
        if not ranked_ids:
            continue
        hits = len(set(ranked_ids[:k]) & set(gold_ids)) if gold_ids else 1
        scores.append(min(1.0, hits / max(k, 1)))
    return sum(scores) / len(scores) if scores else 0.0
