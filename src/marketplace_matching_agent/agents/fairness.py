"""Fairness agent node."""

from __future__ import annotations

import hashlib
import time
from typing import cast

import structlog

from marketplace_matching_agent.mcp_client import (
    MCPRegistry,
    append_audit_row_mcp,
    build_audit_row,
    get_registry,
)
from marketplace_matching_agent.state import FairnessReport, MatchState, MatchStateUpdate, Rationale
from marketplace_matching_agent.types import ItemDict

log = structlog.get_logger(__name__)


def _query_hash(query: str) -> str:
    """Return a short stable hash for log correlation."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


async def _sync_rationales(
    state: MatchState,
    ranked: list[ItemDict],
    k: int,
    registry: MCPRegistry,
) -> list[Rationale]:
    """Align rationales with final ranked list; cite any newly promoted items."""
    existing = {r.item_id: r for r in state.get("rationales", [])}
    rationales: list[Rationale] = []
    for item in ranked[:k]:
        item_id = str(item.get("id", ""))
        if item_id in existing:
            rationales.append(existing[item_id])
            continue
        payload = await registry.call(
            "cite_match",
            query=state["query"],
            candidate_text=str(item.get("text", "")),
            counterparty_text=state["query"],
            candidate_id=item_id,
            mode=state["mode"],
        )
        rationales.append(Rationale.model_validate(payload))
    return rationales


async def run_fairness(state: MatchState) -> MatchStateUpdate:
    """Audit ranked list via MCP; rebalance once if audit fails."""
    t0 = time.perf_counter()
    k = state["k"]
    ranked = list(state.get("ranked_items", []))
    rebalance_pool = list(state.get("retrieved_items", ranked))
    rationales = list(state.get("rationales", []))
    registry = await get_registry()

    report_payload = await registry.call("audit_ranked_list", ranked=ranked, k=k)
    report = FairnessReport.model_validate(report_payload)
    if not report.passed:
        rebalance_payload = await registry.call(
            "rebalance_detconstsort",
            ranked=rebalance_pool,
            k=k,
        )
        ranked = list(cast(list[ItemDict], rebalance_payload.get("ranked", [])))
        report = FairnessReport.model_validate(rebalance_payload.get("report", report_payload))
        report = report.model_copy(update={"rebalanced": True})
        rationales = await _sync_rationales(state, ranked, k, registry)

    final_ranked = ranked[:k]
    await append_audit_row_mcp(
        build_audit_row(
            mode=state["mode"],
            query_hash=_query_hash(state["query"]),
            node="fairness",
            retrieved_doc_ids=[str(i.get("id", "")) for i in state.get("retrieved_items", [])],
            rerank_scores={
                str(item.get("id", "")): float(item.get("rerank_score", item.get("score", 0.0)))
                for item in final_ranked
            },
            fairness_metrics=report.model_dump(),
            fairness_violation=not report.passed,
        )
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info(
        "fairness_node",
        mode=state["mode"],
        query_hash=_query_hash(state["query"]),
        node="fairness",
        latency_ms=round(latency_ms, 2),
        passed=report.passed,
        rebalanced=report.rebalanced,
    )
    return {
        "ranked_items": final_ranked,
        "rationales": rationales,
        "fairness_report": report,
        "audit_row_hash": "via_mcp",
    }
