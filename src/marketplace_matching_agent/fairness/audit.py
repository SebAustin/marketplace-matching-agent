from __future__ import annotations

from marketplace_matching_agent.state import FairnessReport


def four_fifths_impact_ratio(selections: dict[str, int], pool: dict[str, int]) -> float:
    raise NotImplementedError


def min_skew_at_k(ranked: list[dict[str, object]], protected_attr: str, k: int) -> float:
    raise NotImplementedError


def demographic_parity_gap(ranked: list[dict[str, object]], protected_attr: str) -> float:
    raise NotImplementedError


def audit(ranked: list[dict[str, object]], k: int) -> FairnessReport:
    raise NotImplementedError
