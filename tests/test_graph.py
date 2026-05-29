"""Graph integration tests."""

from __future__ import annotations

from typing import Literal, cast

import pytest

from marketplace_matching_agent.graph import build_supervisor
from marketplace_matching_agent.mcp_client import reset_registry


@pytest.fixture(autouse=True)
def _reset_mcp() -> None:
    reset_registry()
    yield
    reset_registry()


@pytest.mark.parametrize("mode", ["seeker", "recruiter"])
@pytest.mark.asyncio
async def test_supervisor_invoke(mode: str) -> None:
    graph = build_supervisor()
    out = await graph.ainvoke(
        cast(
            dict[str, object],
            {
                "mode": cast(Literal["seeker", "recruiter"], mode),
                "query": "python backend austin query 0",
                "k": 5,
            },
        )
    )
    ranked = out.get("ranked_items", [])
    assert len(ranked) == 5
    rationales = out.get("rationales", [])
    assert len(rationales) == 5
    for rationale, item in zip(rationales, ranked, strict=True):
        assert rationale.item_id == str(item["id"])
        assert len(rationale.citations) >= 3
    report = out["fairness_report"]
    assert report.passed is True or report.rebalanced is True
