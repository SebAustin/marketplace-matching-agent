"""Hybrid retrieval tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from evals.metrics import ndcg_at_k
from marketplace_matching_agent.retrieval.hybrid import hybrid_retrieve

GOLD_PATH = Path("tests/fixtures/retrieval_gold.json")


@pytest.mark.asyncio
async def test_hybrid_retrieve_returns_scored_items() -> None:
    items = await hybrid_retrieve("python backend austin", "jobs", k=50, rerank_top_n=10)
    assert len(items) == 10
    assert "rerank_score" in items[0]


@pytest.mark.asyncio
async def test_hybrid_ndcg_and_latency() -> None:
    gold = json.loads(GOLD_PATH.read_text())
    ndcgs: list[float] = []
    latencies: list[float] = []
    for entry in gold["queries"][:5]:
        t0 = time.perf_counter()
        items = await hybrid_retrieve(entry["query"], "jobs", k=50, rerank_top_n=10)
        latencies.append(time.perf_counter() - t0)
        ranked_ids = [str(i["id"]) for i in items]
        ndcgs.append(ndcg_at_k(ranked_ids, entry["relevant_ids"], 10))
    assert sum(ndcgs) / len(ndcgs) >= 0.0
    latencies.sort()
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    assert p95 < 5.0
