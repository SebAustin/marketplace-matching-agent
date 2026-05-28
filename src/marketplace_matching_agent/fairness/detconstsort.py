"""DetConstSort fair re-ranking (Geyik et al., KDD 2019)."""

from __future__ import annotations

import math
from collections.abc import Callable

from marketplace_matching_agent.fairness.synthetic_protected import assign


def _default_attr_fn(item: dict[str, object]) -> str:
    meta = item.get("meta", {})
    if isinstance(meta, dict) and meta.get("synthetic") is True:
        return assign(item)
    if isinstance(meta, dict) and "synthetic_group" in meta:
        return str(meta["synthetic_group"])
    return "A"


def _prefix_feasible(
    counts: dict[str, int],
    group: str,
    prefix_len: int,
    k: int,
    desired_distribution: dict[str, float],
) -> bool:
    """Return True if adding `group` at prefix length `prefix_len` stays feasible."""
    trial = dict(counts)
    trial[group] = trial.get(group, 0) + 1
    remaining = k - prefix_len
    for attr, target_frac in desired_distribution.items():
        selected = trial.get(attr, 0)
        if selected > math.ceil(prefix_len * target_frac - 1e-9):
            return False
        min_required = math.floor(k * target_frac - 1e-9)
        if selected + remaining < min_required:
            return False
    return True


def detconstsort(
    ranked: list[dict[str, object]],
    attr_fn: Callable[[dict[str, object]], str] | None,
    desired_distribution: dict[str, float],
    k: int,
) -> list[dict[str, object]]:
    """Rebalance top-k list toward desired group proportions.

    Greedy DetConstSort: at each prefix, pick the highest-ranked feasible item.
    """
    fn = attr_fn or _default_attr_fn
    result: list[dict[str, object]] = []
    remaining = list(ranked)
    group_counts: dict[str, int] = {group: 0 for group in desired_distribution}

    for prefix_len in range(1, k + 1):
        picked_idx = 0
        for idx, item in enumerate(remaining):
            group = fn(item)
            if _prefix_feasible(group_counts, group, prefix_len, k, desired_distribution):
                picked_idx = idx
                break
        chosen = remaining.pop(picked_idx)
        chosen_group = fn(chosen)
        group_counts[chosen_group] = group_counts.get(chosen_group, 0) + 1
        result.append(chosen)

    return result


def rebalance(
    ranked: list[dict[str, object]],
    k: int,
    desired: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Convenience wrapper for fair rebalance with 50/50 default."""
    dist = desired or {"A": 0.5, "B": 0.5}
    return detconstsort(ranked, None, dist, k)
