"""Fairness audit tests."""

from __future__ import annotations

import math

import pytest

from marketplace_matching_agent.fairness.audit import (
    audit,
    demographic_parity_gap,
    four_fifths_impact_ratio,
    min_skew_at_k,
)
from marketplace_matching_agent.fairness.detconstsort import rebalance
from marketplace_matching_agent.fairness.synthetic_protected import assign


def _item(i: int, group: str) -> dict[str, object]:
    name = "Brad" if group == "A" else "Keisha"
    return {
        "id": f"c_{i}",
        "text": f"{name} engineer python",
        "rerank_score": 1.0 - i * 0.01,
        "meta": {"synthetic": True, "synthetic_group": group},
    }


def _mean_score(items: list[dict[str, object]]) -> float:
    return sum(float(item["rerank_score"]) for item in items) / len(items)


def test_impact_ratio_biased_fixture_fails() -> None:
    ranked = [_item(i, "A") for i in range(16)] + [_item(i + 16, "B") for i in range(16)]
    report = audit(ranked, k=20)
    assert report.impact_ratio == pytest.approx(0.25, rel=0.01)
    assert report.passed is False


def test_rebalance_then_passes_within_score_tolerance() -> None:
    ranked = [_item(i, "A") for i in range(16)] + [_item(i + 16, "B") for i in range(16)]
    original_top = ranked[:20]
    original_mean = _mean_score(original_top)
    rebalanced = rebalance(ranked, k=20)
    report = audit(rebalanced, k=20)
    assert report.passed is True
    assert _mean_score(rebalanced) >= original_mean * 0.95


def test_four_fifths_ratio() -> None:
    ratio = four_fifths_impact_ratio({"A": 16, "B": 1}, {"A": 80, "B": 20})
    assert ratio == pytest.approx(0.25, rel=0.01)


def test_min_skew_balanced_topk() -> None:
    ranked = [_item(i, "A" if i % 2 == 0 else "B") for i in range(20)]
    skew = min_skew_at_k(ranked, "synthetic_group", 10, {"A": 0.5, "B": 0.5})
    assert skew == pytest.approx(0.0, abs=0.05)


def test_demographic_parity_gap_empty() -> None:
    assert demographic_parity_gap([], "synthetic_group") == 0.0


def test_four_fifths_zero_advantaged_rate() -> None:
    assert four_fifths_impact_ratio({"A": 0, "B": 0}, {"A": 10, "B": 10}) == 0.0
    assert four_fifths_impact_ratio({"A": 0, "B": 3}, {"A": 10, "B": 10}) == pytest.approx(math.inf)


def test_min_skew_empty_or_missing_group() -> None:
    assert min_skew_at_k([], "synthetic_group", 5) == 0.0
    ranked = [_item(i, "A") for i in range(10)]
    skew = min_skew_at_k(ranked, "synthetic_group", 10, {"A": 0.5, "B": 0.5})
    assert skew == float("-inf")


def test_group_defaults_to_advantaged() -> None:
    ranked = [
        {"id": "1", "text": "plain", "meta": {}},
        {"id": "2", "text": "y", "meta": {"synthetic_group": "B"}},
    ]
    report = audit(ranked, k=2)
    assert report.impact_ratio == pytest.approx(1.0)


def test_assign_requires_synthetic_flag() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        assign({"id": "x", "text": "Brad", "meta": {}})


def test_group_from_meta_without_synthetic_flag() -> None:
    ranked = [
        {"id": "1", "text": "x", "meta": {"synthetic_group": "A"}},
        {"id": "2", "text": "y", "meta": {"synthetic_group": "B"}},
    ]
    report = audit(ranked, k=2)
    assert report.impact_ratio == pytest.approx(1.0)
