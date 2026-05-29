"""Search agent node."""

from __future__ import annotations

import hashlib
import time
from typing import cast

import structlog

from marketplace_matching_agent.mcp_client import (
    append_audit_row_mcp,
    build_audit_row,
    get_registry,
)
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
    """Execute hybrid retrieval via job_search MCP tool."""
    t0 = time.perf_counter()
    mode = state["mode"]
    tower = _tower_for_mode(mode)
    k = state["k"]
    registry = await get_registry()
    payload = await registry.call(
        "search_jobs",
        query=state["query"],
        k=max(k, 10),
        tower=tower,
    )
    raw_results = payload.get("results", [])
    items = [_normalize_item(item) for item in cast(list[dict[str, object]], raw_results)]
    await append_audit_row_mcp(
        build_audit_row(
            mode=mode,
            query_hash=_query_hash(state["query"]),
            node="search",
            retrieved_doc_ids=[str(item.get("id", "")) for item in items],
        )
    )
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
