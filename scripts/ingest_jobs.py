"""Ingest jobs into Qdrant and Tantivy."""

from __future__ import annotations

import json
from pathlib import Path

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from marketplace_matching_agent.config import get_settings
from marketplace_matching_agent.retrieval.bm25 import index_documents
from marketplace_matching_agent.retrieval.dense import _deterministic_vector, _ensure_collection

log = structlog.get_logger(__name__)


async def main() -> None:
    fixture = Path("tests/fixtures/jobs_docs.json")
    docs = json.loads(fixture.read_text())
    index_documents("jobs", docs)
    settings = get_settings()
    client = AsyncQdrantClient(url=settings.qdrant_url)
    await _ensure_collection(client, "jobs")
    points = []
    for i, doc in enumerate(docs):
        points.append(
            qmodels.PointStruct(
                id=i + 1,
                vector=_deterministic_vector(str(doc["id"]), settings.embed_dim),
                payload={"doc_id": doc["id"], "text": doc["text"], "meta": doc.get("meta", {})},
            )
        )
    await client.upsert(collection_name="jobs_v1", points=points)
    log.info("ingest_jobs_done", n=len(docs))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
