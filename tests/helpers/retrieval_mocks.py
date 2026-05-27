"""HTTP mocks for retrieval integration tests."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import respx

from marketplace_matching_agent.config import get_settings

FIXTURE_ROOT = Path("tests/fixtures")
GOLD_PATH = FIXTURE_ROOT / "retrieval_gold.json"
EMBED_DIM = 256
QDRANT_PORT = 6333


def _load_docs(tower: str) -> list[dict[str, object]]:
    return list(json.loads((FIXTURE_ROOT / f"{tower}_docs.json").read_text()))


def _load_gold_map() -> dict[str, list[str]]:
    gold = json.loads(GOLD_PATH.read_text())
    return {entry["query"]: list(entry["relevant_ids"]) for entry in gold["queries"]}


def _deterministic_vector(text: str, dim: int = EMBED_DIM) -> list[float]:
    seed = sum(ord(char) for char in text)
    raw = [((seed * (index + 1)) % 997) / 997.0 for index in range(dim)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class _QdrantMemory:
    """Minimal in-memory Qdrant store for respx-backed tests."""

    def __init__(self) -> None:
        self.collections: dict[str, list[dict[str, Any]]] = {}

    def ensure(self, name: str, tower: str) -> None:
        if name in self.collections:
            return
        points: list[dict[str, Any]] = []
        for index, doc in enumerate(_load_docs(tower), start=1):
            text = str(doc.get("text", ""))
            points.append(
                {
                    "id": index,
                    "vector": _deterministic_vector(text),
                    "payload": {
                        "doc_id": doc["id"],
                        "text": text,
                        "meta": doc.get("meta", {}),
                    },
                }
            )
        self.collections[name] = points

    def search(self, name: str, vector: list[float], limit: int) -> list[dict[str, Any]]:
        scored = [
            (point, _cosine(vector, point["vector"]))
            for point in self.collections.get(name, [])
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            {"id": point["id"], "score": score, "payload": point["payload"]}
            for point, score in scored[:limit]
        ]


_QDRANT = _QdrantMemory()
_TEXT_TO_DOC_ID: dict[str, str] = {}


def _text_to_doc_id(text: str, tower: str = "jobs") -> str:
    global _TEXT_TO_DOC_ID
    if not _TEXT_TO_DOC_ID:
        for doc in _load_docs(tower):
            _TEXT_TO_DOC_ID[str(doc["text"])] = str(doc["id"])
    return _TEXT_TO_DOC_ID.get(text, text)


def _cohere_score(query: str, doc_id: str, text: str, gold_map: dict[str, list[str]]) -> float:
    relevant = gold_map.get(query, [])
    if doc_id in relevant:
        return 1000.0 - float(relevant.index(doc_id))
    query_tokens = set(query.lower().split())
    return float(sum(1 for token in query_tokens if token in text.lower()))


def _request_method(request: httpx.Request) -> str:
    method = request.method
    if isinstance(method, bytes):
        return method.decode().upper()
    return str(method).upper()


def _handle_qdrant(request: httpx.Request) -> httpx.Response:
    path = request.url.path.rstrip("/")
    method = _request_method(request)
    if path.endswith("/collections") and method == "GET":
        return httpx.Response(200, json={"result": {"collections": []}})
    if "/collections/" in path and method == "PUT" and path.endswith("/points"):
        return httpx.Response(200, json={"result": {"status": "completed"}})
    if "/collections/" in path and method == "PUT":
        name = path.rstrip("/").split("/")[-1]
        tower = name.replace("_v1", "")
        _QDRANT.ensure(name, tower)
        return httpx.Response(200, json={"result": True})
    if path.endswith("/points/search") and method == "POST":
        name = path.split("/")[2]
        body = json.loads(request.content.decode())
        hits = _QDRANT.search(name, body["vector"], int(body.get("limit", 10)))
        return httpx.Response(
            200,
            json={
                "result": [
                    {
                        "id": hit["id"],
                        "version": 0,
                        "score": hit["score"],
                        "payload": hit["payload"],
                    }
                    for hit in hits
                ]
            },
        )
    if path.endswith("/points/scroll") and method == "POST":
        name = path.split("/")[2]
        tower = name.replace("_v1", "")
        _QDRANT.ensure(name, tower)
        points = _QDRANT.collections[name]
        return httpx.Response(
            200,
            json={
                "result": {
                    "points": [
                        {"id": point["id"], "version": 0, "payload": point["payload"]}
                        for point in points
                    ],
                    "next_page_offset": None,
                }
            },
        )
    return httpx.Response(404, json={"status": {"error": "unhandled qdrant route"}})


def _handle_cohere(request: httpx.Request, gold_map: dict[str, list[str]]) -> httpx.Response:
    body = json.loads(request.content.decode())
    query = str(body["query"])
    documents = [str(doc) for doc in body["documents"]]
    top_n = int(body["top_n"])
    ranked_docs = sorted(
        documents,
        key=lambda text: (-_cohere_score(query, _text_to_doc_id(text), text, gold_map), text),
    )
    results = []
    for text in ranked_docs[:top_n]:
        doc_id = _text_to_doc_id(text)
        results.append(
            {
                "index": documents.index(text),
                "relevance_score": _cohere_score(query, doc_id, text, gold_map),
            }
        )
    return httpx.Response(200, json={"results": results})


@contextmanager
def mock_retrieval_services(
    *,
    qdrant_host: str = "qdrant.test",
) -> Iterator[respx.MockRouter]:
    """Install respx routes for Qdrant/Cohere and patch Voyage embed."""
    get_settings.cache_clear()
    gold_map = _load_gold_map()

    def embed_side_effect(*_args: object, **kwargs: object) -> MagicMock:
        texts = kwargs.get("texts") or (_args[0] if _args else [""])
        text = str(texts[0])
        embedding = _deterministic_vector(text)
        result = MagicMock()
        result.embeddings = [embedding]
        return result

    def dispatch(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == qdrant_host:
            return _handle_qdrant(request)
        if host == "api.cohere.com" and request.url.path == "/v2/rerank":
            return _handle_cohere(request, gold_map)
        return httpx.Response(404, json={"status": {"error": f"unhandled host {host}"}})

    with patch("voyageai.Client") as mock_client:
        mock_client.return_value.embed.side_effect = embed_side_effect
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=dispatch)
            yield router


@contextmanager
def mock_cohere_rerank_route(
    side_effect: list[httpx.Response],
) -> Iterator[respx.Route]:
    """Install a respx route for Cohere rerank with a scripted response sequence."""
    get_settings.cache_clear()

    def dispatch(request: httpx.Request) -> httpx.Response:
        if (
            request.url.host == "api.cohere.com"
            and request.url.path == "/v2/rerank"
            and side_effect
        ):
            return side_effect.pop(0)
        return httpx.Response(500, json={"message": "unexpected cohere call"})

    with respx.mock(assert_all_called=False) as router:
        route = router.route(host="api.cohere.com").mock(side_effect=dispatch)
        yield route
