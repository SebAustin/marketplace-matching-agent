"""MCP server module tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_resume_parser_tools() -> None:
    from mcp_servers.resume_parser.server import extract_skills, parse_resume

    out = await parse_resume("Python LangGraph engineer in Austin")
    assert "sections" in out
    skills = await extract_skills("Python LangGraph FastAPI")
    assert "skills" in skills


@pytest.mark.asyncio
async def test_fairness_audit_server() -> None:
    from mcp_servers.fairness_audit.server import audit_ranked_list, rebalance_detconstsort

    ranked = [
        {"id": "1", "meta": {"synthetic_group": "A"}, "rerank_score": 1.0},
        {"id": "2", "meta": {"synthetic_group": "A"}, "rerank_score": 0.9},
    ]
    report = await audit_ranked_list(ranked, k=2)
    assert "passed" in report
    out = await rebalance_detconstsort(ranked, k=2)
    assert "ranked" in out


@pytest.mark.asyncio
async def test_evaluator_server() -> None:
    from mcp_servers.evaluator.server import cite_match_tool, score_match

    score = await score_match("python austin", "python engineer austin")
    assert score["score"] > 0
    rationale = await cite_match_tool("python", "Brad. Python. Austin.", "Need Python.", "c1")
    assert len(rationale["citations"]) >= 3


@pytest.mark.asyncio
async def test_job_search_server() -> None:
    from mcp_servers.job_search.server import get_job_by_id, search_jobs

    results = await search_jobs("python austin", k=5)
    assert len(results["results"]) == 5
    job = await get_job_by_id("jobs_000")
    assert "job" in job


def test_mcp_client_build() -> None:
    from marketplace_matching_agent.mcp_client import build_mcp_client

    assert build_mcp_client() is not None
