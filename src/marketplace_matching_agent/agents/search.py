"""Search agent node."""

from __future__ import annotations

import hashlib
import time

import structlog

from marketplace_matching_agent.retrieval.hybrid import hybrid_retrieve
from marketplace_matching_agent.state import MatchState

log = structlog.get_logger(__name__)


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


async def run_search(state: MatchState) -> MatchState:
    """Execute hybrid retrieval based on mode.

    Args:
        state: Current match state.

    Returns:
        Updated state with retrieved_items.
    """
    t0 = time.perf_counter()
    mode = state["mode"]
    tower = "jobs" if mode == "seeker" else "candidates"
    k = state.get("k", 5)
    items = await hybrid_retrieve(state["query"], tower, k=50, rerank_top_n=max(k, 10))
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "search_node",
        mode=mode,
        query_hash=_query_hash(state["query"]),
        node="search",
        latency_ms=round(latency_ms, 2),
        n=len(items),
    )
    return {**state, "retrieved_items": items}
