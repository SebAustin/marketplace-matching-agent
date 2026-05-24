"""Evaluation agent node."""

from __future__ import annotations

import asyncio
import hashlib
import time

import structlog

from marketplace_matching_agent.extraction.citations import cite_match
from marketplace_matching_agent.state import MatchState, Rationale

log = structlog.get_logger(__name__)


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


async def run_evaluation(state: MatchState) -> MatchState:
    """Score and cite top retrieved items.

    Args:
        state: Current match state with retrieved_items.

    Returns:
        Updated state with ranked_items and rationales.
    """
    t0 = time.perf_counter()
    k = state.get("k", 5)
    items = state.get("retrieved_items", [])[:k]
    counterparty = {"id": "query", "text": state["query"], "meta": {}}
    sem = asyncio.Semaphore(5)

    async def _cite(item: dict[str, object]) -> tuple[dict[str, object], Rationale]:
        async with sem:
            if state["mode"] == "seeker":
                rationale = await cite_match(state["query"], item, counterparty)
            else:
                rationale = await cite_match(state["query"], item, counterparty)
            return item, rationale

    pairs = await asyncio.gather(*[_cite(item) for item in items])
    ranked: list[dict[str, object]] = []
    rationales: list[Rationale] = []
    for item, rationale in pairs:
        score = float(item.get("rerank_score", 0.0))
        citation_density = len(rationale.citations) / 10.0
        item_copy = dict(item)
        item_copy["eval_score"] = score + citation_density
        ranked.append(item_copy)
        rationales.append(rationale)
    ranked.sort(key=lambda x: float(x.get("eval_score", 0.0)), reverse=True)
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "evaluation_node",
        mode=state["mode"],
        query_hash=_query_hash(state["query"]),
        node="evaluation",
        latency_ms=round(latency_ms, 2),
    )
    return {**state, "ranked_items": ranked, "rationales": rationales}
