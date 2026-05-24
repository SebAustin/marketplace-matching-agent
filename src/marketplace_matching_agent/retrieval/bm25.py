"""Tantivy BM25 lexical search."""

from __future__ import annotations

import json
from pathlib import Path

import structlog
import tantivy

log = structlog.get_logger(__name__)
INDEX_ROOT = Path("data/indexes")
FIXTURE_ROOT = Path("tests/fixtures")


def _index_path(tower: str) -> Path:
    return INDEX_ROOT / f"{tower}.tantivy"


def _load_fixture_docs(tower: str) -> list[dict[str, object]]:
    fixture = FIXTURE_ROOT / f"{tower}_docs.json"
    if fixture.exists():
        return list(json.loads(fixture.read_text()))
    return []


def _ensure_index(tower: str) -> tantivy.Index:
    path = _index_path(tower)
    path.mkdir(parents=True, exist_ok=True)
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field("id", stored=True)
    schema_builder.add_text_field("text", stored=True, tokenizer_name="en_stem")
    schema_builder.add_text_field("meta_json", stored=True)
    schema = schema_builder.build()
    if (path / "meta.json").exists():
        index = tantivy.Index(schema, path=str(path))
        index.reload()
        return index
    index = tantivy.Index(schema, path=str(path))
    writer = index.writer()
    for doc in _load_fixture_docs(tower):
        writer.add_document(
            tantivy.Document(
                id=[str(doc["id"])],
                text=[str(doc["text"])],
                meta_json=[json.dumps(doc.get("meta", {}))],
            )
        )
    writer.commit()
    index.reload()
    return index


async def bm25_search(query: str, tower: str, k: int = 50) -> list[tuple[str, float]]:
    """Run BM25 search over a Tantivy index."""
    index = _ensure_index(tower)
    searcher = index.searcher()
    parser = index.parse_query(query, ["text"])
    hits = searcher.search(parser, k)
    results: list[tuple[str, float]] = []
    for score, doc_address in hits.hits:
        doc = searcher.doc(doc_address)
        doc_id = doc["id"][0]
        results.append((doc_id, float(score)))
    log.debug("bm25_search", tower=tower, query=query, n=len(results))
    return results


def index_documents(tower: str, documents: list[dict[str, object]]) -> None:
    """Index documents into Tantivy."""
    path = _index_path(tower)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    index = _ensure_index(tower)
    writer = index.writer()
    for doc in documents:
        writer.add_document(
            tantivy.Document(
                id=[str(doc["id"])],
                text=[str(doc["text"])],
                meta_json=[json.dumps(doc.get("meta", {}))],
            )
        )
    writer.commit()
    index.reload()
