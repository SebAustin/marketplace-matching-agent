"""RRF tests."""

from __future__ import annotations

from marketplace_matching_agent.retrieval.rrf import rrf_fuse


def test_rrf_fuse_merges_rankings() -> None:
    a = [("d1", 1.0), ("d2", 0.5)]
    b = [("d2", 1.0), ("d3", 0.5)]
    fused = rrf_fuse([a, b], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert "d2" in ids
    assert ids[0] == "d2"
