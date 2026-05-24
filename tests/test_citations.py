"""Citation tests."""

from __future__ import annotations

import pytest

from marketplace_matching_agent.extraction.citations import (
    cite_match,
)


@pytest.mark.asyncio
async def test_cite_match_mock_has_three_citations() -> None:
    candidate = {"id": "c1", "text": "Brad Smith. Python. LangGraph. Postgres. Austin.", "meta": {}}
    counterparty = {"id": "j1", "text": "Need Python backend in Austin.", "meta": {}}
    rationale = await cite_match("python austin", candidate, counterparty)
    assert len(rationale.citations) >= 3


@pytest.mark.asyncio
async def test_citation_contract_error_on_short_text() -> None:
    candidate = {"id": "c1", "text": "x", "meta": {}}
    counterparty = {"id": "j1", "text": "y", "meta": {}}
    rationale = await cite_match("query", candidate, counterparty)
    assert len(rationale.citations) >= 3
