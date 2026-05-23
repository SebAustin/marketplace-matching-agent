from marketplace_matching_agent.state import CitedSpan, FairnessReport, MatchState


def test_cited_span_model() -> None:
    span = CitedSpan(
        document_index=0,
        start_char_index=1,
        end_char_index=5,
        cited_text="test",
    )
    assert span.cited_text == "test"


def test_fairness_report_model() -> None:
    report = FairnessReport(
        impact_ratio=1.0,
        min_skew_k=0.9,
        demographic_parity_gap=0.0,
        passed=True,
        rebalanced=False,
        slice_breakdown={"A": 0.5, "B": 0.5},
    )
    assert report.passed is True


def test_match_state_typed_dict() -> None:
    state: MatchState = {"mode": "seeker", "query": "python", "k": 5}
    assert state["k"] == 5
