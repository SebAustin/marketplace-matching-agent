"""Match state models and TypedDict."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field


class CitedSpan(BaseModel):
    """Character-offset citation span from Anthropic Citations API."""

    document_index: int
    start_char_index: int
    end_char_index: int
    cited_text: str


class Rationale(BaseModel):
    """Evidence-grounded match rationale."""

    item_id: str
    summary: str
    citations: list[CitedSpan] = Field(default_factory=list)


class FairnessReport(BaseModel):
    """Fairness audit metrics for a ranked list."""

    impact_ratio: float
    min_skew_k: float
    demographic_parity_gap: float
    passed: bool
    rebalanced: bool = False
    slice_breakdown: dict[str, float] = Field(default_factory=dict)


class MatchState(TypedDict):
    """LangGraph supervisor state.

    Attributes:
        mode: Seeker searches jobs; recruiter searches candidates.
        query: Natural-language match query.
        k: Final list size written by fairness_node.
        retrieved_items: Hybrid retrieval pool from search_node.
        ranked_items: Top-k list after evaluation and optional rebalance.
        rationales: Citation-grounded rationales aligned with ranked_items.
        fairness_report: Audit metrics from fairness_node.
        audit_row_hash: Append-only audit log chain tip (hex or offline).
        _cost_usd: Accumulated LLM spend for eval harness accounting.
    """

    mode: Literal["seeker", "recruiter"]
    query: str
    k: int
    retrieved_items: NotRequired[list[dict[str, object]]]
    ranked_items: NotRequired[list[dict[str, object]]]
    rationales: NotRequired[list[Rationale]]
    fairness_report: NotRequired[FairnessReport]
    audit_row_hash: NotRequired[str]
    _cost_usd: NotRequired[float]


class MatchStateUpdate(TypedDict, total=False):
    """Partial update returned by supervisor nodes."""

    retrieved_items: list[dict[str, object]]
    ranked_items: list[dict[str, object]]
    rationales: list[Rationale]
    fairness_report: FairnessReport
    audit_row_hash: str
    _cost_usd: float
