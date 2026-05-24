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
    """LangGraph supervisor state."""

    mode: Literal["seeker", "recruiter"]
    query: str
    k: int
    retrieved_items: NotRequired[list[dict[str, object]]]
    ranked_items: NotRequired[list[dict[str, object]]]
    rationales: NotRequired[list[Rationale]]
    fairness_report: NotRequired[FairnessReport]
    audit_row_hash: NotRequired[str]
    _cost_usd: NotRequired[float]
