"""Hybrid retrieval tests."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from evals.metrics import ndcg_at_k
from marketplace_matching_agent.retrieval.hybrid import hybrid_retrieve
from tests.helpers.retrieval_mocks import mock_retrieval_services

GOLD_PATH = Path("tests/fixtures/retrieval_gold.json")
FIXTURE_DOC_COUNT = 200


@pytest.mark.asyncio
async def test_hybrid_retrieve_returns_scored_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    with mock_retrieval_services():
        items = await hybrid_retrieve("python backend austin query 0", "jobs", k=50, rerank_top_n=10)
    assert len(items) == 10
    required = {"id", "text", "bm25_score", "dense_score", "rrf_score", "rerank_score", "meta"}
    assert required.issubset(items[0].keys())


@pytest.mark.asyncio
async def test_hybrid_ndcg_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    gold = json.loads(GOLD_PATH.read_text())
    ndcgs: list[float] = []
    latencies: list[float] = []

    with mock_retrieval_services():
        for entry in gold["queries"]:
            t0 = time.perf_counter()
            items = await asyncio.wait_for(
                hybrid_retrieve(
                    str(entry["query"]),
                    "jobs",
                    k=FIXTURE_DOC_COUNT,
                    rerank_top_n=10,
                ),
                timeout=1.2,
            )
            latencies.append(time.perf_counter() - t0)
            ranked_ids = [str(item["id"]) for item in items]
            ndcgs.append(ndcg_at_k(ranked_ids, entry["relevant_ids"], 10))

    mean_ndcg = sum(ndcgs) / len(ndcgs)
    assert mean_ndcg >= 0.78
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    assert p95 < 1.2
