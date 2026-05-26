"""Qdrant dense vector search with Voyage embeddings."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import structlog
import voyageai
from httpx import ConnectError, HTTPError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException

from marketplace_matching_agent.config import get_settings

log = structlog.get_logger(__name__)
FIXTURE_ROOT = Path("tests/fixtures")
EMBED_MODEL = "voyage-3-large"
EMBED_DIM = 256


def _load_fixture_docs(tower: str) -> list[dict[str, object]]:
    """Load tower documents from fixtures."""
    fixture = FIXTURE_ROOT / f"{tower}_docs.json"
    if fixture.exists():
        return list(json.loads(fixture.read_text()))
    return []


def _collection_name(tower: str) -> str:
    """Return Qdrant collection name for a tower."""
    return f"{tower}_v1"


def _deterministic_vector(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Build a deterministic unit vector from text for offline/fixture mode."""
    seed = sum(ord(char) for char in text)
    raw = [((seed * (index + 1)) % 997) / 997.0 for index in range(dim)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def _lexical_overlap(query: str, text: str) -> float:
    """Score overlap between query tokens and document text."""
    query_tokens = query.lower().split()
    words = set(text.lower().split())
    return float(sum(1 for token in query_tokens if token in words))


def _offline_search(query: str, tower: str, k: int) -> list[tuple[str, float]]:
    """Fixture-backed dense fallback ranked by lexical overlap."""
    scored: list[tuple[str, float]] = []
    for doc in _load_fixture_docs(tower):
        overlap = _lexical_overlap(query, str(doc.get("text", "")))
        if overlap > 0:
            scored.append((str(doc["id"]), overlap))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:k]


def _embed_sync(query: str, api_key: str) -> list[float]:
    """Call Voyage embed API synchronously (run in worker thread)."""
    client = voyageai.Client(api_key=api_key)
    result = client.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="query",
        output_dimension=EMBED_DIM,
    )
    return list(result.embeddings[0])


async def _embed_query(query: str) -> list[float]:
    """Embed a query with voyage-3-large at Matryoshka dim 256."""
    settings = get_settings()
    if not settings.voyage_api_key:
        return _deterministic_vector(query, EMBED_DIM)
    return await asyncio.to_thread(_embed_sync, query, settings.voyage_api_key)


async def _ensure_collection(client: AsyncQdrantClient, tower: str) -> None:
    """Create and seed the tower collection when missing."""
    name = _collection_name(tower)
    collections = await client.get_collections()
    existing = {collection.name for collection in collections.collections}
    if name in existing:
        return
    await client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=EMBED_DIM, distance=qmodels.Distance.COSINE),
    )
    points: list[qmodels.PointStruct] = []
    for index, doc in enumerate(_load_fixture_docs(tower), start=1):
        text = str(doc.get("text", ""))
        points.append(
            qmodels.PointStruct(
                id=index,
                vector=_deterministic_vector(text, EMBED_DIM),
                payload={
                    "doc_id": doc["id"],
                    "text": text,
                    "meta": doc.get("meta", {}),
                },
            )
        )
    if points:
        await client.upsert(collection_name=name, points=points)


async def dense_search(query: str, tower: str, k: int = 50) -> list[tuple[str, float]]:
    """Search Qdrant with dense embeddings.

    Args:
        query: User query string.
        tower: Collection tower name.
        k: Maximum hits to return.

    Returns:
        Ranked list of (document id, cosine_score).
    """
    settings = get_settings()
    try:
        client = AsyncQdrantClient(url=settings.qdrant_url)
        await _ensure_collection(client, tower)
        vector = await _embed_query(query)
        hits = await client.search(
            collection_name=_collection_name(tower),
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        results = [
            (str((hit.payload or {}).get("doc_id", hit.id)), float(hit.score)) for hit in hits
        ]
        log.debug("dense_search", tower=tower, query=query, n=len(results))
        return results
    except (
        OSError,
        ConnectionError,
        ConnectError,
        HTTPError,
        ValueError,
        RuntimeError,
        ResponseHandlingException,
    ) as exc:
        log.warning("dense_search_offline_fallback", err=str(exc))
        return _offline_search(query, tower, k)


async def fetch_payloads(tower: str, doc_ids: list[str]) -> dict[str, dict[str, object]]:
    """Fetch document payloads from Qdrant or fixtures."""
    if not doc_ids:
        return {}
    wanted = set(doc_ids)
    lookup: dict[str, dict[str, object]] = {}
    settings = get_settings()
    try:
        client = AsyncQdrantClient(url=settings.qdrant_url)
        await _ensure_collection(client, tower)
        hits, _ = await client.scroll(
            collection_name=_collection_name(tower),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for point in hits:
            payload = point.payload or {}
            doc_id = str(payload.get("doc_id", point.id))
            if doc_id in wanted:
                lookup[doc_id] = {
                    "id": doc_id,
                    "text": str(payload.get("text", "")),
                    "meta": payload.get("meta", {}),
                }
    except (
        OSError,
        ConnectionError,
        ConnectError,
        HTTPError,
        ValueError,
        RuntimeError,
        ResponseHandlingException,
    ):
        for doc in _load_fixture_docs(tower):
            doc_id = str(doc["id"])
            if doc_id in wanted:
                lookup[doc_id] = {
                    "id": doc_id,
                    "text": str(doc.get("text", "")),
                    "meta": doc.get("meta", {}),
                }
    return lookup
