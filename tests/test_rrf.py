"""RRF tests."""

from __future__ import annotations

from marketplace_matching_agent.retrieval.rrf import rrf_fuse


def test_rrf_fuse_merges_rankings_with_k60() -> None:
    first = [("d1", 1.0), ("d2", 0.5)]
    second = [("d2", 1.0), ("d3", 0.5)]
    fused = rrf_fuse([first, second], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "d2"
    assert set(ids) == {"d1", "d2", "d3"}


def test_rrf_uses_rank_only() -> None:
    first = [("b", 100.0), ("a", 1.0)]
    second = [("b", 0.01), ("c", 0.02)]
    fused = rrf_fuse([first, second], k=60)
    assert fused[0][0] == "b"
