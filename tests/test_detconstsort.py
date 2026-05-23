"""DetConstSort tests."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from marketplace_matching_agent.fairness.detconstsort import detconstsort


def _mk_item(i: int, group: str) -> dict[str, object]:
    return {
        "id": str(i),
        "rerank_score": 1.0 - i * 0.01,
        "meta": {"synthetic_group": group},
    }


def test_detconstsort_length_and_no_duplicates() -> None:
    ranked = [_mk_item(i, "A" if i % 4 else "B") for i in range(30)]
    out = detconstsort(ranked, None, {"A": 0.5, "B": 0.5}, k=10)
    assert len(out) == 10
    assert len({item["id"] for item in out}) == 10


@given(st.data())
@settings(max_examples=20, deadline=None)
def test_detconstsort_property(data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=10, max_value=40))
    ranked = [_mk_item(i, "A" if i % 3 else "B") for i in range(n)]
    k = data.draw(st.integers(min_value=5, max_value=10))
    out = detconstsort(ranked, None, {"A": 0.5, "B": 0.5}, k=k)
    assert len(out) == k
