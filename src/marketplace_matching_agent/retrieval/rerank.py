"""Cohere Rerank 3.5 integration."""

from __future__ import annotations

import asyncio

import cohere
import httpx
import structlog

from marketplace_matching_agent.config import get_settings

log = structlog.get_logger(__name__)


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

    if not settings.cohere_api_key:
        return [(cid, 1.0 - i * 0.01) for i, (cid, _text) in enumerate(candidates[:top_n])]

    client = cohere.AsyncClientV2(api_key=settings.cohere_api_key)
    documents = [text for _cid, text in candidates]
    id_by_index = [cid for cid, _text in candidates]

    for attempt in range(3):
        try:
            response = await client.rerank(
                model=settings.rerank_model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            return [
                (id_by_index[r.index], float(r.relevance_score))
                for r in response.results
            ]
        except cohere.errors.TooManyRequestsError:
            wait = 2**attempt
            log.warning("cohere_rerank_429", wait_s=wait, attempt=attempt)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                wait = 2**attempt
                await asyncio.sleep(wait)
            else:
                raise
    return [(cid, 1.0 - i * 0.01) for i, (cid, _text) in enumerate(candidates[:top_n])]
