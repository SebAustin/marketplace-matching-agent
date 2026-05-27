"""Citation tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from marketplace_matching_agent.config import get_settings
from marketplace_matching_agent.extraction.citations import (
    CitationContractError,
    cite_match,
)
from marketplace_matching_agent.state import CitedSpan

FIXTURE_ROOT = Path("tests/fixtures")
TWO_CITATIONS_PATH = FIXTURE_ROOT / "anthropic_messages_two_citations.json"
THREE_CITATIONS_PATH = FIXTURE_ROOT / "anthropic_messages_three_citations.json"

CANDIDATE = {
    "id": "jobs_000",
    "text": "Brad Smith. Python. LangGraph. Postgres. Austin.",
    "meta": {},
}
COUNTERPARTY = {
    "id": "jd_1",
    "text": "Need Python backend in Austin.",
    "meta": {},
}
QUERY = "python backend austin"


def _load_fixture(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


@pytest.mark.asyncio
async def test_citation_contract_error_when_fewer_than_three_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    get_settings.cache_clear()
    with respx.mock(assert_all_called=False) as router:
        router.route(host="api.anthropic.com").mock(
            return_value=httpx.Response(200, json=_load_fixture(TWO_CITATIONS_PATH))
        )
        with pytest.raises(CitationContractError, match="fewer than 3 citations"):
            await cite_match(QUERY, CANDIDATE, COUNTERPARTY, mode="recruiter")


@pytest.mark.asyncio
async def test_rationale_cited_spans_match_fixture_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    get_settings.cache_clear()
    fixture = _load_fixture(THREE_CITATIONS_PATH)
    with respx.mock(assert_all_called=False) as router:
        router.route(host="api.anthropic.com").mock(
            return_value=httpx.Response(200, json=fixture)
        )
        rationale = await cite_match(QUERY, CANDIDATE, COUNTERPARTY, mode="recruiter")

    expected = [
        CitedSpan(
            document_index=int(cite["document_index"]),
            start_char_index=int(cite["start_char_index"]),
            end_char_index=int(cite["end_char_index"]),
            cited_text=str(cite["cited_text"]),
        )
        for block in fixture["content"]
        if block["type"] == "text"
        for cite in block["citations"]
        if cite["type"] == "char_location"
    ]
    assert len(rationale.citations) == 3
    assert rationale.citations == expected
    assert [span.cited_text for span in rationale.citations] == [
        cite.cited_text for cite in expected
    ]
