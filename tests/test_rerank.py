"""Rerank tests."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.retrieval.rerank import cohere_rerank


@pytest.mark.asyncio
async def test_cohere_rerank_offline() -> None:
    candidates = [("d1", "python austin"), ("d2", "java seattle")]
    out = await cohere_rerank("python austin", candidates, top_n=2)
    assert len(out) == 2
