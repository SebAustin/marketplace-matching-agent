import pytest

from marketplace_matching_agent.fairness import audit as fairness_audit


def test_four_fifths_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        fairness_audit.four_fifths_impact_ratio({"A": 1}, {"A": 10})


def test_min_skew_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        fairness_audit.min_skew_at_k([], "group", 5)


def test_demographic_parity_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        fairness_audit.demographic_parity_gap([], "group")


def test_audit_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        fairness_audit.audit([], 5)
