"""Unit tests for MCP wrappers and helpers."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.evaluation.scoring import score_match
from marketplace_matching_agent.extraction.resume import extract_skills_from_text, parse_resume_text
from marketplace_matching_agent.mcp_client import MCPRegistry, parse_tool_result, reset_registry
from mcp_servers.audit_log.server import (
    AppendAuditRowInput,
    append_audit_row,
    query_audit_by_filter,
)
from mcp_servers.evaluator.server import CiteMatchInput, ScoreMatchInput, cite_match
from mcp_servers.evaluator.server import score_match as mcp_score
from mcp_servers.fairness_audit.server import (
    AuditRankedListInput,
    RebalanceInput,
    audit_ranked_list,
    rebalance_detconstsort,
)
from mcp_servers.job_search.server import (
    GetJobByIdInput,
    SearchJobsInput,
    get_job_by_id,
    search_jobs,
)
from mcp_servers.resume_parser.server import (
    ExtractSkillsInput,
    ParseResumeInput,
    extract_skills,
    parse_resume,
)


@pytest.mark.asyncio
async def test_resume_parser_tools() -> None:
    parsed = await parse_resume(ParseResumeInput(text="Python engineer\nAustin"))
    assert parsed["word_count"] == 3
    skills = await extract_skills(ExtractSkillsInput(text="Python LangGraph FastAPI"))
    assert "Python" in skills["skills"]


@pytest.mark.asyncio
async def test_evaluator_tools() -> None:
    scored = await mcp_score(ScoreMatchInput(query="python austin", item_text="python engineer austin"))
    assert scored["score"] > 0
    cited = await cite_match(
        CiteMatchInput(
            query="python",
            candidate_text="Brad Python Austin",
            counterparty_text="Need Python",
            candidate_id="c1",
            mode="recruiter",
        )
    )
    assert len(cited["citations"]) >= 3


@pytest.mark.asyncio
async def test_job_search_tools() -> None:
    results = await search_jobs(SearchJobsInput(query="python austin query 0", k=3, tower="jobs"))
    assert len(results["results"]) == 3
    job = await get_job_by_id(GetJobByIdInput(job_id="jobs_000"))
    assert job["job"] is None or isinstance(job["job"], dict)


@pytest.mark.asyncio
async def test_fairness_audit_tools() -> None:
    ranked = [
        {"id": "1", "meta": {"synthetic_group": "A"}, "rerank_score": 1.0},
        {"id": "2", "meta": {"synthetic_group": "A"}, "rerank_score": 0.9},
    ]
    report = await audit_ranked_list(AuditRankedListInput(ranked=ranked, k=2))
    assert "passed" in report
    rebalanced = await rebalance_detconstsort(
        RebalanceInput(
            ranked=[
                {"id": "1", "meta": {"synthetic_group": "A"}, "rerank_score": 1.0},
                {"id": "2", "meta": {"synthetic_group": "B"}, "rerank_score": 0.9},
            ],
            k=2,
        )
    )
    assert "ranked" in rebalanced


@pytest.mark.asyncio
async def test_audit_log_offline_fallback() -> None:
    from mcp_servers.audit_log.server import QueryAuditInput

    row_json = (
        '{"mode":"seeker","query_hash":"q","prompt_version":"v0.1.0","model_id":"m",'
        '"retrieved_doc_ids":[],"rerank_scores":{},"fairness_metrics":{},"fairness_violation":false}'
    )
    appended = await append_audit_row(AppendAuditRowInput(row_json=row_json))
    assert appended["row_hash"]
    queried = await query_audit_by_filter(QueryAuditInput(prompt_version=None))
    assert "rows" in queried


def test_src_helpers() -> None:
    assert parse_resume_text("one two three")["word_count"] == 3
    assert score_match("python", "python dev")["score"] == 1.0
    assert extract_skills_from_text("Python Go")["skills"]


def test_parse_tool_result_variants() -> None:
    assert parse_tool_result({"ok": True}) == {"ok": True}
    assert parse_tool_result('{"ok": true}') == {"ok": True}
    assert parse_tool_result([{"type": "text", "text": '{"x": 1}'}]) == {"x": 1}
    with pytest.raises(TypeError):
        parse_tool_result(42)


@pytest.mark.asyncio
async def test_mcp_registry_call() -> None:
    reset_registry()
    from unittest.mock import AsyncMock, MagicMock

    tool = MagicMock()
    tool.name = "score_match"
    tool.ainvoke = AsyncMock(return_value='{"score": 1.0}')
    client = MagicMock()
    client.get_tools = AsyncMock(return_value=[tool])
    registry = MCPRegistry(client)
    result = await registry.call("score_match", query="q", item_text="t")
    assert result["score"] == 1.0
