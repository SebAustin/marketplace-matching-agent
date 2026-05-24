"""Dense search tests."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.retrieval.dense import dense_search


@pytest.mark.asyncio
async def test_dense_search_returns_results() -> None:
    hits = await dense_search("python austin", "jobs", k=5)
    assert len(hits) == 5
