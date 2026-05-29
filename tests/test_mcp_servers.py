"""MCP server integration tests."""

from __future__ import annotations

import statistics
import time

import pytest

from marketplace_matching_agent.mcp_client import (
    build_mcp_client,
    parse_tool_result,
    reset_registry,
)

TOOL_CALLS: dict[str, dict[str, object]] = {
    "parse_resume": {"params": {"text": "Python LangGraph engineer in Austin"}},
    "extract_skills": {"params": {"text": "Python LangGraph FastAPI"}},
    "search_jobs": {"params": {"query": "python austin query 0", "k": 5, "tower": "jobs"}},
    "get_job_by_id": {"params": {"job_id": "jobs_000"}},
    "score_match": {"params": {"query": "python austin", "item_text": "python engineer austin"}},
    "cite_match": {
        "params": {
            "query": "python austin",
            "candidate_text": "Brad. Python. LangGraph. Austin.",
            "counterparty_text": "Need Python in Austin.",
            "candidate_id": "c1",
            "mode": "recruiter",
        }
    },
    "audit_ranked_list": {
        "params": {
            "ranked": [
                {"id": "1", "meta": {"synthetic_group": "A"}, "rerank_score": 1.0},
                {"id": "2", "meta": {"synthetic_group": "A"}, "rerank_score": 0.9},
            ],
            "k": 2,
        }
    },
    "rebalance_detconstsort": {
        "params": {
            "ranked": [
                {"id": "1", "meta": {"synthetic_group": "A"}, "rerank_score": 1.0},
                {"id": "2", "meta": {"synthetic_group": "B"}, "rerank_score": 0.9},
            ],
            "k": 2,
        }
    },
    "append_audit_row": {
        "params": {
            "row_json": (
                '{"mode":"seeker","query_hash":"abc","prompt_version":"v0.1.0",'
                '"model_id":"test","retrieved_doc_ids":[],"rerank_scores":{},'
                '"fairness_metrics":{"node":"test"},"fairness_violation":false}'
            )
        }
    },
    "query_audit_by_filter": {"params": {"prompt_version": None}},
}

SERVER_TOOL: dict[str, str] = {
    "resume_parser": "parse_resume",
    "job_search": "search_jobs",
    "evaluator": "cite_match",
    "fairness_audit": "audit_ranked_list",
    "audit_log": "append_audit_row",
}


@pytest.fixture(autouse=True)
def _reset_mcp_registry() -> None:
    reset_registry()
    yield
    reset_registry()


@pytest.mark.asyncio
async def test_mcp_servers_list_and_invoke_one_tool_per_server() -> None:
    client = build_mcp_client()
    tools = {tool.name: tool for tool in await client.get_tools()}
    assert set(SERVER_TOOL.values()).issubset(set(tools))

    for tool_name in SERVER_TOOL.values():
        payload = TOOL_CALLS[tool_name]
        result = parse_tool_result(await tools[tool_name].ainvoke(payload))
        assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_mcp_tool_latency_p95_under_two_seconds() -> None:
    client = build_mcp_client()
    tools = {tool.name: tool for tool in await client.get_tools()}
    latencies: list[float] = []
    probe_tools = ["parse_resume", "score_match", "search_jobs"]

    for _ in range(20):
        for tool_name in probe_tools:
            t0 = time.perf_counter()
            await tools[tool_name].ainvoke(TOOL_CALLS[tool_name])
            latencies.append(time.perf_counter() - t0)

    latencies.sort()
    p95 = statistics.quantiles(latencies, n=20)[-1]
    assert p95 < 2.0
