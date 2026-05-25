"""State model tests."""

from __future__ import annotations

from typing import Literal, cast

from marketplace_matching_agent.state import (
    CitedSpan,
    FairnessReport,
    MatchState,
    MatchStateUpdate,
    Rationale,
)


def test_cited_span_model() -> None:
    span = CitedSpan(document_index=0, start_char_index=0, end_char_index=5, cited_text="hello")
    assert span.cited_text == "hello"


def test_rationale_model() -> None:
    citations = [
        CitedSpan(document_index=0, start_char_index=i, end_char_index=i + 3, cited_text=f"c{i}")
        for i in range(3)
    ]
    rationale = Rationale(item_id="x", summary="s", citations=citations)
    assert len(rationale.citations) == 3


def test_fairness_report_defaults() -> None:
    report = FairnessReport(
        impact_ratio=0.9,
        min_skew_k=-0.02,
        demographic_parity_gap=0.05,
        passed=True,
    )
    assert report.rebalanced is False


def test_match_state_required_keys() -> None:
    state: MatchState = {
        "mode": cast(Literal["seeker", "recruiter"], "seeker"),
        "query": "python backend austin",
        "k": 5,
    }
    assert state["mode"] == "seeker"
    assert state["k"] == 5


def test_match_state_update_is_partial() -> None:
    update: MatchStateUpdate = {
        "ranked_items": [{"id": "1", "text": "t", "score": 0.9, "meta": {}}],
    }
    assert update["ranked_items"][0]["score"] == 0.9
