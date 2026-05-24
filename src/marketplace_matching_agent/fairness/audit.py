"""Fairness audit metrics."""

from __future__ import annotations

import math

from marketplace_matching_agent.fairness.synthetic_protected import assign
from marketplace_matching_agent.state import FairnessReport

PROTECTED_ATTR = "synthetic_group"


def _group(item: dict[str, object]) -> str:
    meta = item.get("meta", {})
    if isinstance(meta, dict) and meta.get("synthetic") is True:
        return assign(item)
    if isinstance(meta, dict) and "synthetic_group" in meta:
        return str(meta["synthetic_group"])
    return "A"


def four_fifths_impact_ratio(selections: dict[str, int], pool: dict[str, int]) -> float:
    """Compute EEOC 4/5ths adverse impact ratio.

    Args:
        selections: Selected counts per group.
        pool: Pool counts per group.

    Returns:
        Impact ratio (disadvantaged rate / advantaged rate).
    """
    groups = sorted(selections.keys())
    if len(groups) < 2:
        return 1.0
    a, b = groups[0], groups[1]
    sel_a = selections.get(a, 0)
    sel_b = selections.get(b, 0)
    pool_a = max(pool.get(a, 1), 1)
    pool_b = max(pool.get(b, 1), 1)
    rate_a = sel_a / pool_a
    rate_b = sel_b / pool_b
    if rate_a == 0:
        return 0.0 if rate_b == 0 else float("inf")
    disadvantaged = min(rate_a, rate_b)
    advantaged = max(rate_a, rate_b)
    return disadvantaged / advantaged if advantaged > 0 else 1.0


def min_skew_at_k(
    ranked: list[dict[str, object]],
    protected_attr: str,
    k: int,
    desired_distribution: dict[str, float] | None = None,
) -> float:
    """Compute MinSkew@k over ranked list.

    Args:
        ranked: Ranked items.
        protected_attr: Attribute key (unused; uses synthetic assign).
        k: Cutoff k.
        desired_distribution: Target group proportions (default 50/50).

    Returns:
        Minimum log skew across groups vs achievable top-k proportions.
    """
    _ = protected_attr
    dist = desired_distribution or {"A": 0.5, "B": 0.5}
    top = ranked[:k]
    if not top:
        return 0.0
    counts: dict[str, int] = {}
    for item in top:
        g = _group(item)
        counts[g] = counts.get(g, 0) + 1
    total = len(top)
    skews: list[float] = []
    for group, target_frac in dist.items():
        achievable = round(target_frac * k) / total if total > 0 else target_frac
        actual = counts.get(group, 0) / total
        if achievable > 0 and actual > 0:
            skews.append(math.log(actual / achievable))
        elif actual == 0 and achievable > 0:
            skews.append(-1.0)
    return min(skews) if skews else 0.0


def demographic_parity_gap(ranked: list[dict[str, object]], protected_attr: str) -> float:
    """Compute max-min selection rate gap across groups."""
    _ = protected_attr
    counts: dict[str, int] = {}
    for item in ranked:
        g = _group(item)
        counts[g] = counts.get(g, 0) + 1
    if not counts:
        return 0.0
    total = sum(counts.values())
    rates = [c / total for c in counts.values()]
    return max(rates) - min(rates)


def audit(ranked: list[dict[str, object]], k: int) -> FairnessReport:
    """Run fairness audit on ranked list.

    Args:
        ranked: Ranked candidate/job items.
        k: Evaluation cutoff.

    Returns:
        FairnessReport with pass/fail and slice metrics.
    """
    top = ranked[:k]
    pool_counts: dict[str, int] = {"A": 0, "B": 0}
    sel_counts: dict[str, int] = {"A": 0, "B": 0}
    for item in ranked:
        g = _group(item)
        pool_counts[g] = pool_counts.get(g, 0) + 1
    for item in top:
        g = _group(item)
        sel_counts[g] = sel_counts.get(g, 0) + 1

    impact = four_fifths_impact_ratio(sel_counts, pool_counts)
    desired = {"A": 0.5, "B": 0.5}
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
