"""Fairness audit metrics."""

from __future__ import annotations

import math

from marketplace_matching_agent.fairness.synthetic_protected import assign
from marketplace_matching_agent.state import FairnessReport

PROTECTED_ATTR = "synthetic_group"
ADVANTAGED_GROUP = "A"
DISADVANTAGED_GROUP = "B"


def _group(item: dict[str, object]) -> str:
    meta = item.get("meta", {})
    if isinstance(meta, dict) and meta.get("synthetic") is True:
        return assign(item)
    if isinstance(meta, dict) and "synthetic_group" in meta:
        return str(meta["synthetic_group"])
    return ADVANTAGED_GROUP


def four_fifths_impact_ratio(selections: dict[str, int], pool: dict[str, int]) -> float:
    """Compute EEOC 4/5ths adverse impact ratio for groups A and B.

    Returns (sel_rate_B / pool_B) / (sel_rate_A / pool_A).
    """
    sel_a = selections.get(ADVANTAGED_GROUP, 0)
    sel_b = selections.get(DISADVANTAGED_GROUP, 0)
    pool_a = max(pool.get(ADVANTAGED_GROUP, 1), 1)
    pool_b = max(pool.get(DISADVANTAGED_GROUP, 1), 1)
    rate_a = sel_a / pool_a
    rate_b = sel_b / pool_b
    if rate_a == 0:
        return 0.0 if rate_b == 0 else math.inf
    return rate_b / rate_a


def min_skew_at_k(
    ranked: list[dict[str, object]],
    protected_attr: str,
    k: int,
    desired_distribution: dict[str, float] | None = None,
) -> float:
    """Compute MinSkew@k: min_g log(actual_prop_g / desired_prop_g) in top-k."""
    _ = protected_attr
    dist = desired_distribution or {ADVANTAGED_GROUP: 0.5, DISADVANTAGED_GROUP: 0.5}
    top = ranked[:k]
    if not top or k <= 0:
        return 0.0
    counts: dict[str, int] = {}
    for item in top:
        group = _group(item)
        counts[group] = counts.get(group, 0) + 1
    skews: list[float] = []
    for group, desired_prop in dist.items():
        actual_prop = counts.get(group, 0) / k
        if actual_prop > 0 and desired_prop > 0:
            skews.append(math.log(actual_prop / desired_prop))
        elif actual_prop == 0 and desired_prop > 0:
            skews.append(float("-inf"))
    return min(skews) if skews else 0.0


def demographic_parity_gap(ranked: list[dict[str, object]], protected_attr: str) -> float:
    """Compute max-min selection rate across groups in top-k."""
    _ = protected_attr
    if not ranked:
        return 0.0
    counts: dict[str, int] = {}
    for item in ranked:
        group = _group(item)
        counts[group] = counts.get(group, 0) + 1
    total = sum(counts.values())
    rates = [count / total for count in counts.values()]
    return max(rates) - min(rates)


def audit(ranked: list[dict[str, object]], k: int) -> FairnessReport:
    """Run fairness audit on ranked list."""
    top = ranked[:k]
    pool_counts: dict[str, int] = {ADVANTAGED_GROUP: 0, DISADVANTAGED_GROUP: 0}
    sel_counts: dict[str, int] = {ADVANTAGED_GROUP: 0, DISADVANTAGED_GROUP: 0}
    for item in ranked:
        group = _group(item)
        pool_counts[group] = pool_counts.get(group, 0) + 1
    for item in top:
        group = _group(item)
        sel_counts[group] = sel_counts.get(group, 0) + 1

    impact = four_fifths_impact_ratio(sel_counts, pool_counts)
    desired = {ADVANTAGED_GROUP: 0.5, DISADVANTAGED_GROUP: 0.5}
    skew = min_skew_at_k(ranked, PROTECTED_ATTR, k, desired)
    gap = demographic_parity_gap(top, PROTECTED_ATTR)
    passed = impact >= 0.80 and abs(skew) <= 0.10
    return FairnessReport(
        impact_ratio=impact,
        min_skew_k=skew,
        demographic_parity_gap=gap,
        passed=passed,
        rebalanced=False,
        slice_breakdown={
            "impact_ratio": impact,
            "min_skew_k": skew,
            "demographic_parity_gap": gap,
        },
    )
