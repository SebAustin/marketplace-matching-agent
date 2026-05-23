"""Hybrid retrieval orchestrator."""

from __future__ import annotations

import asyncio

import structlog

from marketplace_matching_agent.retrieval.bm25 import bm25_search
from marketplace_matching_agent.retrieval.dense import dense_search, fetch_payloads
from marketplace_matching_agent.retrieval.rerank import cohere_rerank
from marketplace_matching_agent.retrieval.rrf import rrf_fuse

log = structlog.get_logger(__name__)


async def hybrid_retrieve(
    query: str,
    tower: str,
    k: int = 50,
    rerank_top_n: int = 10,
) -> list[dict[str, object]]:
    """Run BM25 + dense + RRF + Cohere rerank cascade.

    Args:
        query: User query.
        tower: jobs or candidates tower.
        k: Retrieval pool size.
        rerank_top_n: Final reranked results.

    Returns:
        List of dicts with id, text, scores, and meta.
    """
    bm25_hits, dense_hits = await asyncio.gather(
        bm25_search(query, tower, k),
        dense_search(query, tower, k),
    )
    bm25_map = dict(bm25_hits)
    dense_map = dict(dense_hits)
    fused = rrf_fuse([bm25_hits, dense_hits], k=60)
    fused_ids = [doc_id for doc_id, _score in fused[:k]]
    payloads = await fetch_payloads(tower, fused_ids)

    candidates: list[tuple[str, str]] = []
    for doc_id in fused_ids:
        payload = payloads.get(doc_id, {"id": doc_id, "text": doc_id, "meta": {}})
        candidates.append((doc_id, str(payload.get("text", doc_id))))

    reranked = await cohere_rerank(query, candidates, rerank_top_n)
    results: list[dict[str, object]] = []
    rrf_map = dict(fused)
    for doc_id, rerank_score in reranked:
        payload = payloads.get(doc_id, {"id": doc_id, "text": doc_id, "meta": {}})
        results.append(
            {
                "id": doc_id,
                "text": payload.get("text", doc_id),
                "bm25_score": bm25_map.get(doc_id, 0.0),
                "dense_score": dense_map.get(doc_id, 0.0),
                "rrf_score": rrf_map.get(doc_id, 0.0),
                "rerank_score": rerank_score,
                "meta": payload.get("meta", {}),
            }
        )
    log.info("hybrid_retrieve", tower=tower, query=query, n=len(results))
    return results
