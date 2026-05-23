from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel


class CitedSpan(BaseModel):
    document_index: int
    start_char_index: int
    end_char_index: int
    cited_text: str


class Rationale(BaseModel):
    item_id: str
    summary: str
    citations: list[CitedSpan]


class FairnessReport(BaseModel):
    impact_ratio: float
    min_skew_k: float
    demographic_parity_gap: float
    passed: bool
    rebalanced: bool
    slice_breakdown: dict[str, float]


class MatchState(TypedDict):
    mode: Literal["seeker", "recruiter"]
    query: str
    k: int
    retrieved_items: NotRequired[list[dict[str, object]]]
    ranked_items: NotRequired[list[dict[str, object]]]
    rationales: NotRequired[list[Rationale]]
    fairness_report: NotRequired[FairnessReport]
    audit_row_hash: NotRequired[str]
