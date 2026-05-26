"""Dense search tests."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.retrieval.dense import EMBED_DIM, _collection_name, dense_search
from tests.helpers.retrieval_mocks import mock_retrieval_services


@pytest.mark.asyncio
async def test_dense_search_returns_results_with_mocked_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage")
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.test:6333")
    with mock_retrieval_services():
        hits = await dense_search("python austin", "jobs", k=5)
    assert len(hits) == 5
    assert all(isinstance(doc_id, str) and isinstance(score, float) for doc_id, score in hits)


def test_dense_collection_name_and_dim() -> None:
    assert _collection_name("jobs") == "jobs_v1"
    assert EMBED_DIM == 256
