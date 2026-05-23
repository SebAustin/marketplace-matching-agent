"""DetConstSort fair re-ranking (Geyik et al., KDD 2019)."""

from __future__ import annotations

from collections.abc import Callable

from marketplace_matching_agent.fairness.synthetic_protected import assign


def _default_attr_fn(item: dict[str, object]) -> str:
    meta = item.get("meta", {})
    if isinstance(meta, dict) and meta.get("synthetic") is True:
        return assign(item)
    if isinstance(meta, dict) and "synthetic_group" in meta:
        return str(meta["synthetic_group"])
    return "A"


def detconstsort(
    ranked: list[dict[str, object]],
    attr_fn: Callable[[dict[str, object]], str] | None,
    desired_distribution: dict[str, float],
    k: int,
) -> list[dict[str, object]]:
    """Rebalance top-k list toward desired group proportions.

    Args:
        ranked: Original relevance-ordered list.
        attr_fn: Attribute extractor; defaults to synthetic assign.
        desired_distribution: Target proportions per group.
        k: Output list length.

    Returns:
        Reordered list of length k.
    """
    fn = attr_fn or _default_attr_fn
    result: list[dict[str, object]] = []
    remaining = list(ranked)
    group_counts: dict[str, int] = {g: 0 for g in desired_distribution}

    while len(result) < k and remaining:
        prefix_len = len(result) + 1
        best_idx = 0
        best_penalty = float("inf")
        for idx, item in enumerate(remaining):
            g = fn(item)
            trial_counts = dict(group_counts)
            trial_counts[g] = trial_counts.get(g, 0) + 1
            penalty = 0.0
            for group, desired in desired_distribution.items():
                actual = trial_counts.get(group, 0) / prefix_len
                penalty += abs(actual - desired)
            penalty += idx * 0.001
            if penalty < best_penalty:
                best_penalty = penalty
                best_idx = idx
        chosen = remaining.pop(best_idx)
        g = fn(chosen)
        group_counts[g] = group_counts.get(g, 0) + 1
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
