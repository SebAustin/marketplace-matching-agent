"""Unit tests for audit log hash chain (no Postgres)."""

from __future__ import annotations

from pathlib import Path

from marketplace_matching_agent.audit.log import AuditRow, canonical_json, compute_row_hash


def test_canonical_json_sorted() -> None:
    a = canonical_json({"b": 1, "a": 2})
    b = canonical_json({"a": 2, "b": 1})
    assert a == b


def test_compute_row_hash_chain() -> None:
    row1 = AuditRow(
        mode="seeker",
        query_hash="h1",
        prompt_version="v0.1.0",
        model_id="m",
        retrieved_doc_ids=["d1"],
        rerank_scores={"d1": 0.9},
        fairness_metrics={"passed": True},
        fairness_violation=False,
    )
    h1 = compute_row_hash(row1, None)
    row1.row_hash = h1
    row2 = AuditRow(
        mode="seeker",
        query_hash="h2",
        prompt_version="v0.1.0",
        model_id="m",
        retrieved_doc_ids=["d2"],
        rerank_scores={"d2": 0.8},
        fairness_metrics={"passed": True},
        fairness_violation=False,
        prev_hash=h1,
    )
    h2 = compute_row_hash(row2, h1)
    row2.row_hash = h2
    assert h1 != h2


def test_synthetic_protected_assign() -> None:
    from marketplace_matching_agent.fairness.synthetic_protected import assign

    item = {"text": "Keisha Smith engineer", "meta": {"synthetic": True}}
    assert assign(item) == "B"
    item_a = {"text": "Brad Smith engineer", "meta": {"synthetic": True}}
    assert assign(item_a) == "A"


def test_synthetic_protected_raises_on_real_data() -> None:
    import pytest

    from marketplace_matching_agent.fairness.synthetic_protected import assign

    with pytest.raises(ValueError):
        assign({"text": "Real person", "meta": {}})


def test_types_aliases() -> None:
    from marketplace_matching_agent.types import ItemDict, RankedList

    item: ItemDict = {"id": "1"}
    assert item["id"] == "1"
    _ = RankedList


def test_eval_metrics() -> None:
    from evals.metrics import four_fifths_pass_rate, mrr, ndcg_at_k, recall_at_k

    assert ndcg_at_k(["a", "b"], ["b"], 2) > 0
    assert mrr(["x", "b"], ["b"]) == 0.5
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 2) == 0.5
    assert four_fifths_pass_rate([True, False]) == 0.5


def test_eval_cache() -> None:
    from evals.cache import SQLiteResponseCache

    cache = SQLiteResponseCache(path=Path(".cache/test_eval.sqlite"))
    cache.set("m", "prompt", ["doc"], {"ok": True})
    assert cache.get("m", "prompt", ["doc"]) is None
