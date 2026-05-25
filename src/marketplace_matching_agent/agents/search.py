"""Search agent node."""

from __future__ import annotations

import hashlib
import time

import structlog

from marketplace_matching_agent.retrieval.hybrid import hybrid_retrieve
from marketplace_matching_agent.state import MatchState, MatchStateUpdate
from marketplace_matching_agent.types import ItemDict

log = structlog.get_logger(__name__)


def _query_hash(query: str) -> str:
    """Return a short stable hash for log correlation."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


def _tower_for_mode(mode: str) -> str:
    """Map seeker/recruiter mode to retrieval tower name."""
    return "jobs" if mode == "seeker" else "candidates"


def _normalize_item(raw: dict[str, object]) -> ItemDict:
    """Ensure retrieved records expose id, text, score, and meta."""
    record: ItemDict = dict(raw)
    score_val = raw.get("rerank_score", raw.get("score", 0.0))
    record["score"] = float(score_val) if isinstance(score_val, int | float) else 0.0
    return record


async def run_search(state: MatchState) -> MatchStateUpdate:
    """Execute hybrid retrieval based on mode.

    Mode routing selects the tower inside this node: seeker -> jobs,
    recruiter -> candidates. No conditional graph edges are used.

    Args:
        state: Current match state with mode, query, and k.

    Returns:
        Partial state update with retrieved_items.
    """
    t0 = time.perf_counter()
    mode = state["mode"]
    tower = _tower_for_mode(mode)
    k = state["k"]
    pool = await hybrid_retrieve(state["query"], tower, k=50, rerank_top_n=max(k, 10))
    items = [_normalize_item(item) for item in pool]
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "search_node",
        mode=mode,
        query_hash=_query_hash(state["query"]),
        node="search",
        latency_ms=round(latency_ms, 2),
        tower=tower,
        n=len(items),
    )
    return {"retrieved_items": items}
