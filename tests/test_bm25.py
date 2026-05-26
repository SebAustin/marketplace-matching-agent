"""BM25 tests."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.retrieval.bm25 import _index_path, bm25_search


@pytest.mark.asyncio
async def test_bm25_search_returns_ranked_ids_and_scores() -> None:
    hits = await bm25_search("python austin", "jobs", k=5)
    assert len(hits) == 5
    assert all(isinstance(doc_id, str) and isinstance(score, float) for doc_id, score in hits)


@pytest.mark.asyncio
async def test_bm25_index_path_uses_tantivy_suffix() -> None:
    assert str(_index_path("jobs")).endswith("jobs.tantivy")
