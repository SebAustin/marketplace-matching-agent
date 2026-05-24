"""State model tests."""

from __future__ import annotations

from marketplace_matching_agent.state import CitedSpan, FairnessReport, Rationale


def test_cited_span_model() -> None:
    span = CitedSpan(document_index=0, start_char_index=0, end_char_index=5, cited_text="hello")
    assert span.cited_text == "hello"


def test_rationale_model() -> None:
    citations = [
        CitedSpan(document_index=0, start_char_index=i, end_char_index=i + 3, cited_text=f"c{i}")
        for i in range(3)
    ]
    r = Rationale(item_id="x", summary="s", citations=citations)
    assert len(r.citations) == 3


def test_fairness_report_defaults() -> None:
    fr = FairnessReport(
        impact_ratio=0.9,
        min_skew_k=-0.02,
        demographic_parity_gap=0.05,
        passed=True,
    )
    assert fr.rebalanced is False
