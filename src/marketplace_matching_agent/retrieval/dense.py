"""Qdrant dense vector search with Voyage embeddings."""

from __future__ import annotations

import json
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


def _load_fixture_docs(tower: str) -> list[dict[str, object]]:
    fixture = FIXTURE_ROOT / f"{tower}_docs.json"
    if fixture.exists():
        return json.loads(fixture.read_text())
    return []


def _deterministic_vector(doc_id: str, dim: int) -> list[float]:
    seed = sum(ord(c) for c in doc_id)
    return [((seed * (i + 1)) % 997) / 997.0 for i in range(dim)]


def _offline_search(query: str, tower: str, k: int) -> list[tuple[str, float]]:
    q_tokens = set(query.lower().split())
    scored: list[tuple[str, float]] = []
    for doc in _load_fixture_docs(tower):
        text = str(doc.get("text", "")).lower()
        overlap = len(q_tokens & set(text.split()))
        if overlap > 0:
            scored.append((str(doc["id"]), float(overlap)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


async def _embed_query(query: str) -> list[float]:
    settings = get_settings()
    if not settings.voyage_api_key:
        return _deterministic_vector(query, settings.embed_dim)
    client = voyageai.Client(api_key=settings.voyage_api_key)
    result = client.embed(
        texts=[query],
        model=settings.embed_model,
        input_type="query",
        output_dimension=settings.embed_dim,
    )
    return list(result.embeddings[0])


async def _ensure_collection(client: AsyncQdrantClient, tower: str) -> None:
    settings = get_settings()
    name = f"{tower}_v1"
    collections = await client.get_collections()
    existing = {c.name for c in collections.collections}
    if name not in existing:
        await client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=settings.embed_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        points: list[qmodels.PointStruct] = []
        for i, doc in enumerate(_load_fixture_docs(tower)):
            vec = _deterministic_vector(str(doc["id"]), settings.embed_dim)
            points.append(
                qmodels.PointStruct(
                    id=i + 1,
                    vector=vec,
                    payload={
                        "doc_id": doc["id"],
                        "text": doc["text"],
                        "meta": doc.get("meta", {}),
                    },
                )
            )
        if points:
            await client.upsert(collection_name=name, points=points)


async def dense_search(query: str, tower: str, k: int = 50) -> list[tuple[str, float]]:
    """Search Qdrant with dense embeddings."""
    settings = get_settings()
    try:
        client = AsyncQdrantClient(url=settings.qdrant_url)
        await _ensure_collection(client, tower)
        vector = await _embed_query(query)
        hits = await client.search(
            collection_name=f"{tower}_v1",
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        results: list[tuple[str, float]] = []
        for hit in hits:
            payload = hit.payload or {}
            doc_id = str(payload.get("doc_id", hit.id))
            results.append((doc_id, float(hit.score)))
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
    settings = get_settings()
    lookup: dict[str, dict[str, object]] = {}
    try:
        client = AsyncQdrantClient(url=settings.qdrant_url)
        await _ensure_collection(client, tower)
        hits, _ = await client.scroll(
            collection_name=f"{tower}_v1",
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for point in hits:
            payload = point.payload or {}
            doc_id = str(payload.get("doc_id", point.id))
            if doc_id in doc_ids:
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
            if doc_id in doc_ids:
                lookup[doc_id] = {
                    "id": doc_id,
                    "text": str(doc.get("text", "")),
                    "meta": doc.get("meta", {}),
                }
    return lookup
