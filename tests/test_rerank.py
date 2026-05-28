"""Rerank tests."""

from __future__ import annotations

import httpx
import pytest

from marketplace_matching_agent.retrieval.rerank import RERANK_MODEL, cohere_rerank
from tests.helpers.retrieval_mocks import mock_cohere_rerank_route, mock_retrieval_services


@pytest.mark.asyncio
async def test_cohere_rerank_offline() -> None:
    candidates = [("d1", "python austin backend"), ("d2", "java seattle")]
    out = await cohere_rerank("python austin", candidates, top_n=2)
    assert len(out) == 2
    assert out[0][0] == "d1"


@pytest.mark.asyncio
async def test_cohere_rerank_mocked_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    with mock_retrieval_services():
        candidates = [
            ("jobs_001", "Senior engineer with python experience in Austin."),
            ("jobs_002", "Senior engineer with java experience in Seattle."),
        ]
        out = await cohere_rerank("python backend austin query 0", candidates, top_n=1)
    assert len(out) == 1
    assert out[0][0] == "jobs_001"


@pytest.mark.asyncio
async def test_cohere_rerank_retries_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere")
    responses = [
        httpx.Response(429, json={"message": "rate limit"}),
        httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.99}]}),
    ]
    with mock_cohere_rerank_route(responses) as route:

        async def fast_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(
            "marketplace_matching_agent.retrieval.rerank.asyncio.sleep",
            fast_sleep,
        )
        out = await cohere_rerank("python", [("d1", "python austin")], top_n=1)

    assert route.call_count == 2
    assert out == [("d1", 0.99)]


def test_rerank_model_constant() -> None:
    assert RERANK_MODEL == "rerank-v3.5"
