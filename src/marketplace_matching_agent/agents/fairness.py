"""Fairness agent node."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

import structlog
from psycopg import AsyncConnection, OperationalError

from marketplace_matching_agent.audit.log import AuditRow, append
from marketplace_matching_agent.config import get_settings
from marketplace_matching_agent.extraction.citations import cite_match
from marketplace_matching_agent.fairness.audit import audit
from marketplace_matching_agent.fairness.detconstsort import rebalance
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
) -> list[Rationale]:
    """Align rationales with final ranked list; cite any newly promoted items."""
    existing = {r.item_id: r for r in state.get("rationales", [])}
    counterparty: ItemDict = {"id": "query", "text": state["query"], "meta": {}}
    rationales: list[Rationale] = []
    for item in ranked[:k]:
        item_id = str(item.get("id", ""))
        if item_id in existing:
            rationales.append(existing[item_id])
        else:
            rationales.append(
                await cite_match(state["query"], item, counterparty, mode=state["mode"])
            )
    return rationales


async def _append_audit(state: MatchState, report: FairnessReport) -> str:
    """Persist append-only audit row; return hash or offline sentinel."""
    settings = get_settings()
    ranked = state.get("ranked_items", [])
    rerank_scores = {
        str(item.get("id", "")): float(item.get("rerank_score", item.get("score", 0.0)))
        for item in ranked
    }
    row = AuditRow(
        mode=state["mode"],
        query_hash=_query_hash(state["query"]),
        prompt_version=settings.prompt_version,
        model_id=settings.model_id,
        retrieved_doc_ids=[str(i.get("id", "")) for i in state.get("retrieved_items", [])],
        rerank_scores=rerank_scores,
        fairness_metrics=json.loads(report.model_dump_json()),
        fairness_violation=not report.passed,
    )
    try:
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                async with await AsyncConnection.connect(settings.postgres_url) as conn:
                    return await append(conn, row)
            except (OSError, OperationalError) as exc:
                last_error = exc
                if attempt < 4:
                    await asyncio.sleep(0.5 * (attempt + 1))
        log.warning("audit_log_unavailable", error=str(last_error))
        return "offline"
    except OSError as exc:
        log.warning("audit_log_unavailable", error=str(exc))
        return "offline"


async def run_fairness(state: MatchState) -> MatchStateUpdate:
    """Audit ranked list; rebalance once from retrieved_items if audit fails."""
    t0 = time.perf_counter()
    k = state["k"]
    ranked = list(state.get("ranked_items", []))
    rebalance_pool = list(state.get("retrieved_items", ranked))
    rationales = list(state.get("rationales", []))
    report = audit(ranked, k)
    if not report.passed:
        ranked = rebalance(rebalance_pool, k)
        report = audit(ranked, k)
        report.rebalanced = True
        rationales = await _sync_rationales(state, ranked, k)
    final_ranked = ranked[:k]
    audit_hash = await _append_audit({**state, "ranked_items": final_ranked}, report)
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
        "audit_row_hash": audit_hash,
    }
