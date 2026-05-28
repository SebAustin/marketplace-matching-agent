"""Additional coverage tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from marketplace_matching_agent.agents.evaluation import run_evaluation
from marketplace_matching_agent.agents.fairness import run_fairness
from marketplace_matching_agent.agents.search import run_search
from marketplace_matching_agent.audit.log import (
    AuditRow,
    append,
    compute_row_hash,
    query,
    verify_chain,
)
from marketplace_matching_agent.extraction.citations import cite_match
from marketplace_matching_agent.fairness.audit import demographic_parity_gap, min_skew_at_k
from marketplace_matching_agent.graph import (
    build_supervisor,
    evaluation_node,
    fairness_node,
    search_node,
)
from marketplace_matching_agent.retrieval.bm25 import index_documents
from marketplace_matching_agent.retrieval.hybrid import hybrid_retrieve
from marketplace_matching_agent.retrieval.rerank import cohere_rerank
from marketplace_matching_agent.state import MatchState


def _item(i: int, group: str = "A") -> dict[str, object]:
    name = "Brad" if group == "A" else "Keisha"
    return {
        "id": f"jobs_{i:03d}",
        "text": f"{name} python austin engineer",
        "rerank_score": 1.0 - i * 0.01,
        "meta": {"synthetic": True, "synthetic_group": group},
    }


@pytest.mark.asyncio
async def test_agent_nodes_with_mocks() -> None:
    state: MatchState = {"mode": "seeker", "query": "python austin", "k": 5}
    with patch(
        "marketplace_matching_agent.agents.search.hybrid_retrieve",
        new=AsyncMock(return_value=[_item(i, "A" if i % 2 == 0 else "B") for i in range(5)]),
    ):
        state = {**state, **await run_search(state)}
    assert len(state["retrieved_items"]) == 5

    with patch(
        "marketplace_matching_agent.agents.evaluation.cite_match",
        new=AsyncMock(
            side_effect=lambda q, c, cp, **_: __import__(
                "marketplace_matching_agent.extraction.citations", fromlist=["_mock_rationale"]
            )._mock_rationale(q, c)
        ),
    ):
        state = {**state, **await run_evaluation(state)}
    assert len(state["ranked_items"]) == 5

    with patch(
        "marketplace_matching_agent.agents.fairness.append", new=AsyncMock(return_value="hash")
    ):
        state = {**state, **await run_fairness(state)}
    assert state["fairness_report"].passed is True or state["fairness_report"].rebalanced is True


@pytest.mark.asyncio
async def test_graph_node_wrappers() -> None:
    state: MatchState = {"mode": "recruiter", "query": "python", "k": 3}
    with patch(
        "marketplace_matching_agent.agents.search.hybrid_retrieve",
        new=AsyncMock(return_value=[_item(i) for i in range(3)]),
    ):
        state = {**state, **await search_node(state)}
    with patch(
        "marketplace_matching_agent.agents.evaluation.cite_match",
        new=AsyncMock(
            side_effect=lambda q, c, cp, **_: __import__(
                "marketplace_matching_agent.extraction.citations", fromlist=["_mock_rationale"]
            )._mock_rationale(q, c)
        ),
    ):
        state = {**state, **await evaluation_node(state)}
    with patch(
        "marketplace_matching_agent.agents.fairness.append", new=AsyncMock(return_value="h")
    ):
        state = {**state, **await fairness_node(state)}
    assert "fairness_report" in state


def test_fairness_metric_helpers() -> None:
    ranked = [_item(i, "A" if i < 8 else "B") for i in range(10)]
    assert min_skew_at_k(ranked, "synthetic_group", 10) <= 0.0
    assert demographic_parity_gap(ranked, "synthetic_group") >= 0.0


def test_index_documents() -> None:
    docs = [{"id": "x1", "text": "python engineer", "meta": {}}]
    index_documents("jobs", docs)


@pytest.mark.asyncio
async def test_hybrid_and_rerank_offline() -> None:
    items = await hybrid_retrieve("python austin", "jobs", k=50, rerank_top_n=5)
    assert len(items) == 5
    reranked = await cohere_rerank("python", [("a", "python austin")], top_n=1)
    assert len(reranked) == 1


@pytest.mark.asyncio
async def test_cite_match_mock() -> None:
    candidate = {"id": "1", "text": "Brad. Python. LangGraph. Postgres.", "meta": {}}
    cp = {"id": "2", "text": "Need Python.", "meta": {}}
    r = await cite_match("python", candidate, cp)
    assert len(r.citations) >= 3


def test_build_supervisor_compiles() -> None:
    g = build_supervisor()
    assert g is not None


@pytest.mark.asyncio
async def test_audit_append_mock_conn() -> None:
    row = AuditRow(
        mode="seeker",
        query_hash="q",
        prompt_version="v0.1.0",
        model_id="m",
        retrieved_doc_ids=["d"],
        rerank_scores={"d": 1.0},
        fairness_metrics={},
        fairness_violation=False,
    )
    assert compute_row_hash(row, None)

    class FakeCursor:
        async def __aenter__(self) -> FakeCursor:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def execute(self, *args: object, **kwargs: object) -> None:
            return None

        async def fetchone(self) -> None:
            return None

        async def fetchall(self) -> list[dict[str, object]]:
            return []

    class FakeConn:
        async def __aenter__(self) -> FakeConn:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        def cursor(self, *args: object, **kwargs: object) -> FakeCursor:
            return FakeCursor()

        async def commit(self) -> None:
            return None

    fake = FakeConn()
    h = await append(fake, row)  # type: ignore[arg-type]
    assert h
    assert await verify_chain(fake) is True  # type: ignore[arg-type]
    assert await query(fake) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_cli_main() -> None:
    from typer.testing import CliRunner

    from marketplace_matching_agent.cli import app

    runner = CliRunner()
    with patch("marketplace_matching_agent.cli.asyncio.run") as mock_run:
        mock_run.return_value = {"ranked_items": []}
        result = runner.invoke(app, ["--mode", "seeker", "--query", "python", "--k", "3"])
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_cohere_rerank_with_api_key() -> None:
    mock_result = type(
        "R", (), {"results": [type("X", (), {"index": 0, "relevance_score": 0.99})()]}
    )()
    with patch("marketplace_matching_agent.retrieval.rerank.get_settings") as mock_settings:
        mock_settings.return_value.cohere_api_key = "test-key"
        mock_settings.return_value.rerank_model = "rerank-v3.5"
        with patch(
            "marketplace_matching_agent.retrieval.rerank.cohere.AsyncClientV2"
        ) as mock_client:
            mock_client.return_value.rerank = AsyncMock(return_value=mock_result)
            out = await cohere_rerank("python", [("d1", "python text")], top_n=1)
    assert out[0][0] == "d1"


@pytest.mark.asyncio
async def test_append_audit_row_mcp() -> None:
    from mcp_servers.audit_log.server import append_audit_row

    row = AuditRow(
        mode="seeker",
        query_hash="q",
        prompt_version="v0.1.0",
        model_id="m",
        retrieved_doc_ids=["d"],
        rerank_scores={"d": 1.0},
        fairness_metrics={},
        fairness_violation=False,
    )
    with patch("mcp_servers.audit_log.server.append", new=AsyncMock(return_value="abc")):
        with patch("mcp_servers.audit_log.server.AsyncConnection.connect") as mock_conn:
            mock_conn.return_value.__aenter__ = AsyncMock(return_value=object())
            mock_conn.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await append_audit_row(row.model_dump_json())
    assert result["row_hash"] == "abc"
