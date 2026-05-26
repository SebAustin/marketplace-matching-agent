"""Cohere Rerank 3.5 integration."""

from __future__ import annotations

import asyncio

import cohere
import httpx
import structlog

from marketplace_matching_agent.config import get_settings

log = structlog.get_logger(__name__)
RERANK_MODEL = "rerank-v3.5"


def _lexical_score(query: str, text: str) -> float:
    """Offline relevance score from token overlap."""
    query_tokens = query.lower().split()
    words = set(text.lower().split())
    return float(sum(1 for token in query_tokens if token in words))


def _offline_rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_n: int,
) -> list[tuple[str, float]]:
    """Rank candidates lexically when Cohere is unavailable."""
    ranked = sorted(
        candidates,
        key=lambda item: (-_lexical_score(query, item[1]), item[0]),
    )
    return [
        (doc_id, _lexical_score(query, text)) for doc_id, text in ranked[:top_n]
    ]


async def cohere_rerank(
    query: str,
    candidates: list[tuple[str, str]],
    top_n: int,
) -> list[tuple[str, float]]:
    """Rerank candidate documents with Cohere Rerank 3.5.

    Args:
        query: Search query.
        candidates: List of (id, text) tuples.
        top_n: Number of results to return.

    Returns:
        Reranked (id, relevance_score) list.
    """
    settings = get_settings()
    if not candidates:
        return []
    limit = min(top_n, len(candidates))
    if not settings.cohere_api_key:
        return _offline_rerank(query, candidates, limit)

    client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
    documents = [text for _doc_id, text in candidates]
    id_by_index = [doc_id for doc_id, _text in candidates]

    for attempt in range(3):
        try:
            response = await client.rerank(
                model=RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=limit,
            )
            return [
                (id_by_index[result.index], float(result.relevance_score))
                for result in response.results
            ]
        except cohere.errors.TooManyRequestsError:
            wait = 2**attempt
            log.warning("cohere_rerank_429", wait_s=wait, attempt=attempt)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                wait = 2**attempt
                log.warning("cohere_rerank_429", wait_s=wait, attempt=attempt)
                await asyncio.sleep(wait)
                continue
            raise
    return _offline_rerank(query, candidates, limit)
