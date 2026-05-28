"""Anthropic Citations API integration."""

from __future__ import annotations

from typing import Any, Literal, cast

from anthropic import AsyncAnthropic
from anthropic.types import Message, MessageParam

from marketplace_matching_agent.config import get_settings
from marketplace_matching_agent.state import CitedSpan, Rationale

CITATIONS_MODEL = "claude-sonnet-4-5-20251022"
MIN_CITATIONS = 3

SYSTEM_PROMPT = (
    "You are a match-evidence extractor. For every claim you make, "
    "you MUST attach a citation pointing to the supporting sentence in the "
    "supplied documents. Do not paraphrase cited material. "
    "If a claim cannot be cited, drop it."
)


class CitationContractError(Exception):
    """Raised when rationale has fewer than 3 citations."""


def _mock_rationale(query: str, candidate: dict[str, object]) -> Rationale:
    """Generate deterministic mock rationale for tests/offline mode."""
    text = str(candidate.get("text", ""))
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    citations: list[CitedSpan] = []
    for sentence in sentences[:5]:
        start = text.find(sentence)
        if start < 0:
            continue
        end = start + len(sentence)
        citations.append(
            CitedSpan(
                document_index=0,
                start_char_index=start,
                end_char_index=end,
                cited_text=sentence,
            )
        )
    while len(citations) < MIN_CITATIONS:
        snippet = text[:20] if text else query[:20]
        citations.append(
            CitedSpan(
                document_index=0,
                start_char_index=0,
                end_char_index=len(snippet),
                cited_text=snippet,
            )
        )
    return Rationale(
        item_id=str(candidate.get("id", "unknown")),
        summary=f"Match for query: {query}",
        citations=citations[:5],
    )


def _resolve_document_texts(
    candidate: dict[str, object],
    counterparty: dict[str, object],
    mode: Literal["seeker", "recruiter"],
) -> tuple[str, str]:
    """Map retrieved item and counterparty to resume/JD document text."""
    candidate_text = str(candidate.get("text", ""))
    counterparty_text = str(counterparty.get("text", ""))
    if mode == "seeker":
        return counterparty_text, candidate_text
    return candidate_text, counterparty_text


def _user_prompt(query: str) -> str:
    return (
        f"Question: {query}\n"
        "Return 3–5 short bullet reasons this match fits. "
        "Cite the exact resume and JD sentences that justify each bullet."
    )


def _build_messages(
    query: str,
    resume_text: str,
    jd_text: str,
) -> list[MessageParam]:
    content: list[dict[str, Any]] = [
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": resume_text,
            },
            "title": "Resume",
            "citations": {"enabled": True},
        },
        {
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": jd_text,
            },
            "title": "JobDescription",
            "citations": {"enabled": True},
        },
        {
            "type": "text",
            "text": _user_prompt(query),
        },
    ]
    return [cast(MessageParam, {"role": "user", "content": content})]


def _citation_field(citation: object, field: str) -> str | int:
    value = citation.get(field) if isinstance(citation, dict) else getattr(citation, field, None)
    if isinstance(value, str | int):
        return value
    raise TypeError(f"unexpected citation field {field!r}: {value!r}")


def _parse_rationale(response: Message, candidate: dict[str, object]) -> Rationale:
    """Parse Anthropic message blocks into a Rationale."""
    citations: list[CitedSpan] = []
    summary_parts: list[str] = []
    for block in response.content:
        if block.type != "text":
            continue
        summary_parts.append(block.text)
        block_citations = getattr(block, "citations", None) or []
        for cite in block_citations:
            if _citation_field(cite, "type") != "char_location":
                continue
            citations.append(
                CitedSpan(
                    document_index=int(_citation_field(cite, "document_index")),
                    start_char_index=int(_citation_field(cite, "start_char_index")),
                    end_char_index=int(_citation_field(cite, "end_char_index")),
                    cited_text=str(_citation_field(cite, "cited_text")),
                )
            )
    return Rationale(
        item_id=str(candidate.get("id", "unknown")),
        summary=" ".join(summary_parts).strip(),
        citations=citations,
    )


def _validate_rationale(rationale: Rationale) -> Rationale:
    if len(rationale.citations) < MIN_CITATIONS:
        raise CitationContractError("fewer than 3 citations")
    return rationale


async def cite_match(
    query: str,
    candidate: dict[str, object],
    counterparty: dict[str, object],
    *,
    mode: Literal["seeker", "recruiter"] = "recruiter",
) -> Rationale:
    """Produce citation-grounded match rationale.

    Args:
        query: User's natural-language query.
        candidate: Primary retrieved item (resume for recruiter, job for seeker).
        counterparty: Counterparty context (JD or seeker profile text).
        mode: Seeker swaps resume/JD document roles relative to recruiter mode.

    Returns:
        Rationale with >=3 CitedSpans.

    Raises:
        CitationContractError: If fewer than 3 citations returned.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        return _validate_rationale(_mock_rationale(query, candidate))

    resume_text, jd_text = _resolve_document_texts(candidate, counterparty, mode)
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    response = await client.messages.create(
        model=CITATIONS_MODEL,
        max_tokens=1024,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=_build_messages(query, resume_text, jd_text),
    )
    return _validate_rationale(_parse_rationale(response, candidate))
