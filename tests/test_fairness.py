"""Fairness audit tests."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.fairness.audit import audit, four_fifths_impact_ratio
from marketplace_matching_agent.fairness.detconstsort import rebalance


def _item(i: int, group: str) -> dict[str, object]:
    name = "Brad" if group == "A" else "Keisha"
    return {
        "id": f"c_{i}",
        "text": f"{name} engineer python",
        "rerank_score": 1.0 - i * 0.01,
        "meta": {"synthetic": True, "synthetic_group": group},
    }


def test_impact_ratio_biased_fixture_fails() -> None:
    ranked = [_item(i, "A") for i in range(16)] + [_item(i + 16, "B") for i in range(16)]
    report = audit(ranked, k=10)
    assert report.impact_ratio < 0.80
    assert report.passed is False


def test_rebalance_then_passes() -> None:
    ranked = [_item(i, "A") for i in range(16)] + [_item(i + 16, "B") for i in range(16)]
    rebalanced = rebalance(ranked, k=10)
    report = audit(rebalanced, k=10)
    assert report.passed is True


def test_four_fifths_ratio() -> None:
    ratio = four_fifths_impact_ratio({"A": 16, "B": 1}, {"A": 80, "B": 20})
    assert ratio == pytest.approx(0.25, rel=0.01)
