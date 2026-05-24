"""Eval metrics."""

from __future__ import annotations

import math


def ndcg_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute nDCG@k."""
    if not gold_ids:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(gold_ids)) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def mrr(ranked_ids: list[str], gold_ids: list[str]) -> float:
    """Compute MRR."""
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Compute Recall@k."""
    if not gold_ids:
        return 0.0
    hits = len(set(ranked_ids[:k]) & set(gold_ids))
    return hits / len(gold_ids)


def four_fifths_pass_rate(reports: list[bool]) -> float:
    """Fraction of fairness audits that passed."""
    if not reports:
        return 0.0
    return sum(1 for r in reports if r) / len(reports)
