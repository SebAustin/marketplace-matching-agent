"""Evaluation agent node."""

from __future__ import annotations

import asyncio
import hashlib
import time

import structlog

from marketplace_matching_agent.extraction.citations import cite_match
from marketplace_matching_agent.state import MatchState, MatchStateUpdate, Rationale
from marketplace_matching_agent.types import ItemDict

log = structlog.get_logger(__name__)


def _query_hash(query: str) -> str:
    """Return a short stable hash for log correlation."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


async def run_evaluation(state: MatchState) -> MatchStateUpdate:
    """Score and cite top-k retrieved items.

    Calls cite_match for each of the top-k retrieved items, then reorders by
    combined retrieval score plus citation density.

    Args:
        state: Current match state with retrieved_items.

    Returns:
        Partial state update with ranked_items and rationales.
    """
    t0 = time.perf_counter()
    k = state["k"]
    items = list(state.get("retrieved_items", [])[:k])
    counterparty: ItemDict = {"id": "query", "text": state["query"], "meta": {}}
    sem = asyncio.Semaphore(5)
    mode = state["mode"]

    async def _cite(item: ItemDict) -> tuple[ItemDict, Rationale, float]:
        async with sem:
            rationale = await cite_match(
                state["query"],
                item,
                counterparty,
                mode=mode,
            )
        relevance = float(item.get("score", item.get("rerank_score", 0.0)))
        citation_density = len(rationale.citations) / 10.0
        ranked_item = dict(item)
        ranked_item["eval_score"] = relevance + citation_density
        return ranked_item, rationale, float(ranked_item["eval_score"])

    scored = await asyncio.gather(*[_cite(item) for item in items])
    scored.sort(key=lambda row: row[2], reverse=True)
    ranked_items = [item for item, _, _ in scored]
    rationales = [rationale for _, rationale, _ in scored]
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "evaluation_node",
        mode=state["mode"],
        query_hash=_query_hash(state["query"]),
        node="evaluation",
        latency_ms=round(latency_ms, 2),
        n=len(ranked_items),
    )
    return {"ranked_items": ranked_items, "rationales": rationales}
