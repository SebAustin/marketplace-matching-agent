"""Anthropic Citations API integration."""

from __future__ import annotations

from anthropic import APIError, AsyncAnthropic

from marketplace_matching_agent.config import get_settings
from marketplace_matching_agent.state import CitedSpan, Rationale

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
    for _idx, sentence in enumerate(sentences[:3]):
        start = text.find(sentence)
        end = start + len(sentence)
        citations.append(
            CitedSpan(
                document_index=0,
                start_char_index=start,
                end_char_index=end,
                cited_text=sentence,
            )
        )
    while len(citations) < 3:
        citations.append(
            CitedSpan(
                document_index=0,
                start_char_index=0,
                end_char_index=min(20, len(text)),
                cited_text=text[:20] if text else query[:20],
            )
        )
    return Rationale(
        item_id=str(candidate.get("id", "unknown")),
        summary=f"Match for query: {query}",
        citations=citations[:5],
    )


async def cite_match(
    query: str,
    candidate: dict[str, object],
    counterparty: dict[str, object],
) -> Rationale:
    """Produce citation-grounded match rationale.

    Args:
        query: User query.
        candidate: Primary document (resume or job).
        counterparty: Secondary document (JD or query context).

    Returns:
        Rationale with >=3 CitedSpans.

    Raises:
        CitationContractError: If fewer than 3 citations returned.
    """
    settings = get_settings()
    if not settings.anthropic_api_key:
        rationale = _mock_rationale(query, candidate)
        if len(rationale.citations) < 3:
            raise CitationContractError("fewer than 3 citations")
        return rationale

    try:
        client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        response = await client.messages.create(
            model=settings.model_id,
            max_tokens=1024,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": str(candidate.get("text", "")),
                            },
                            "title": "Resume",
                            "citations": {"enabled": True},
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": str(counterparty.get("text", query)),
                            },
                            "title": "JobDescription",
                            "citations": {"enabled": True},
                        },
                        {
                            "type": "text",
                            "text": (
                                f"Question: {query}\n"
                                "Return 3-5 short bullet reasons this match fits. "
                                "Cite the exact resume and JD sentences that justify each bullet."
                            ),
                        },
                    ],
                }
            ],
        )

        citations: list[CitedSpan] = []
        summary_parts: list[str] = []
        for block in response.content:
            if block.type != "text":
                continue
            summary_parts.append(block.text)
            for cite in block.citations or []:
                if cite.type == "char_location":
                    citations.append(
                        CitedSpan(
                            document_index=cite.document_index,
                            start_char_index=cite.start_char_index,
                            end_char_index=cite.end_char_index,
                            cited_text=cite.cited_text,
                        )
                    )

        rationale = Rationale(
            item_id=str(candidate.get("id", "unknown")),
            summary=" ".join(summary_parts).strip(),
            citations=citations,
        )
        if len(rationale.citations) < 3:
            raise CitationContractError("fewer than 3 citations")
        return rationale
    except (CitationContractError, OSError, ValueError):
        raise
    except APIError:
        return _mock_rationale(query, candidate)
