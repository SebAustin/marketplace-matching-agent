"""DetConstSort tests."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from marketplace_matching_agent.fairness.detconstsort import _prefix_feasible, detconstsort


def _mk_item(i: int, group: str) -> dict[str, object]:
    return {
        "id": str(i),
        "rerank_score": 1.0 - i * 0.01,
        "meta": {"synthetic_group": group},
    }


def test_default_attr_fn_uses_synthetic_assign() -> None:
    ranked = [
        {
            "id": "1",
            "text": "Keisha engineer",
            "meta": {"synthetic": True, "synthetic_group": "B"},
        }
    ]
    out = detconstsort(ranked, None, {"A": 0.5, "B": 0.5}, k=1)
    assert len(out) == 1


def test_default_attr_fn_falls_back_to_advantaged() -> None:
    out = detconstsort([{"id": "1", "text": "plain", "meta": {}}], None, {"A": 1.0}, k=1)
    assert len(out) == 1


def test_prefix_feasible_rejects_insufficient_headroom() -> None:
    counts = {"A": 0, "B": 1}
    assert (
        _prefix_feasible(
            counts,
            "B",
            prefix_len=2,
            k=3,
            desired_distribution={"A": 0.9, "B": 0.1},
        )
        is False
    )
    assert (
        _prefix_feasible(counts, "B", prefix_len=2, k=2, desired_distribution={"A": 0.5, "B": 0.5})
        is False
    )


def test_detconstsort_length_and_no_duplicates() -> None:
    ranked = [_mk_item(i, "A" if i % 4 else "B") for i in range(30)]
    out = detconstsort(ranked, None, {"A": 0.5, "B": 0.5}, k=10)
    assert len(out) == 10
    assert len({item["id"] for item in out}) == 10


@given(st.data())
@settings(max_examples=50, deadline=None)
def test_detconstsort_property(data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=12, max_value=40))
    ranked = [_mk_item(i, "A" if i % 3 else "B") for i in range(n)]
    k = data.draw(st.integers(min_value=8, max_value=min(20, n)))
    out = detconstsort(ranked, None, {"A": 0.5, "B": 0.5}, k=k)
    out_ids = [str(item["id"]) for item in out]
    assert len(out) == k
    assert len(set(out_ids)) == k

    id_to_new_pos = {doc_id: index for index, doc_id in enumerate(out_ids)}
    quarter_cutoff = k // 4
    for original_index, item in enumerate(ranked):
        if original_index > quarter_cutoff:
            break
        doc_id = str(item["id"])
        assert doc_id in id_to_new_pos
        assert abs(id_to_new_pos[doc_id] - original_index) <= k
