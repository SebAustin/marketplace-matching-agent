"""Fairness agent node."""

from __future__ import annotations

import hashlib
import json
import time

import structlog
from psycopg import AsyncConnection, OperationalError

from marketplace_matching_agent.audit.log import AuditRow, append
from marketplace_matching_agent.config import get_settings
from marketplace_matching_agent.fairness.audit import audit
from marketplace_matching_agent.fairness.detconstsort import rebalance
from marketplace_matching_agent.state import MatchState

log = structlog.get_logger(__name__)


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


async def _append_audit(state: MatchState, report: object) -> str:
    settings = get_settings()
    ranked = state.get("ranked_items", [])
    rerank_scores = {
        str(item.get("id", "")): float(item.get("rerank_score", 0.0)) for item in ranked
    }
    row = AuditRow(
        mode=state["mode"],
        query_hash=_query_hash(state["query"]),
        prompt_version=settings.prompt_version,
        model_id=settings.model_id,
        retrieved_doc_ids=[str(i.get("id", "")) for i in state.get("retrieved_items", [])],
        rerank_scores=rerank_scores,
        fairness_metrics=json.loads(report.model_dump_json())
        if hasattr(report, "model_dump_json")
        else {},
        fairness_violation=not getattr(report, "passed", True),
    )
    try:
        async with await AsyncConnection.connect(settings.postgres_url) as conn:
            return await append(conn, row)
    except (OSError, OperationalError):
        log.warning("audit_log_unavailable")
        return "offline"


async def run_fairness(state: MatchState) -> MatchState:
    """Audit ranked list and rebalance if needed.

    Args:
        state: Current match state with ranked_items.

    Returns:
        Updated state with fairness_report and possibly rebalanced ranked_items.
    """
    t0 = time.perf_counter()
    k = state.get("k", 5)
    ranked = list(state.get("ranked_items", []))
    report = audit(ranked, k)
    if not report.passed:
        ranked = rebalance(ranked, k)
        report = audit(ranked, k)
        report.rebalanced = True
    audit_hash = await _append_audit(state, report)
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
        **state,
        "ranked_items": ranked[:k],
        "fairness_report": report,
        "audit_row_hash": audit_hash,
    }
