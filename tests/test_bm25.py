"""BM25 tests."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.retrieval.bm25 import bm25_search


@pytest.mark.asyncio
async def test_bm25_search_returns_results() -> None:
    hits = await bm25_search("python austin", "jobs", k=5)
    assert len(hits) == 5
    assert all(isinstance(doc_id, str) for doc_id, _ in hits)
