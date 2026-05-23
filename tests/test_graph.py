"""Graph integration tests."""

from __future__ import annotations

from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import pytest

from marketplace_matching_agent.graph import build_supervisor


@pytest.mark.parametrize("mode", ["seeker", "recruiter"])
@pytest.mark.asyncio
async def test_supervisor_invoke(mode: str) -> None:
    with patch(
        "marketplace_matching_agent.agents.fairness.append",
        new=AsyncMock(return_value="offline"),
    ):
        graph = build_supervisor()
        out = await graph.ainvoke(
            cast(
                dict[str, object],
                {"mode": cast(Literal["seeker", "recruiter"], mode), "query": "python backend austin", "k": 5},
            )
        )
    ranked = out.get("ranked_items", [])
    assert len(ranked) == 5
    rationales = out.get("rationales", [])
    assert len(rationales) == 5
    for rationale in rationales:
        assert len(rationale.citations) >= 3
    report = out["fairness_report"]
    assert report.passed is True or report.rebalanced is True
