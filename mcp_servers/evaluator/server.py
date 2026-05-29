"""Evaluator MCP server."""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from marketplace_matching_agent.evaluation.scoring import score_match as score_match_fn
from marketplace_matching_agent.extraction.citations import cite_match as cite_match_fn
from mcp_servers.bootstrap import configure_stdio_logging

configure_stdio_logging()

mcp = FastMCP("evaluator")


class ScoreMatchInput(BaseModel):
    query: str = Field(description="User query")
    item_text: str = Field(description="Candidate or job text")


class CiteMatchInput(BaseModel):
    query: str = Field(description="User query")
    candidate_text: str = Field(description="Primary document text")
    counterparty_text: str = Field(description="Counterparty document text")
    candidate_id: str = Field(default="unknown", description="Primary document id")
    mode: Literal["seeker", "recruiter"] = Field(
        default="recruiter",
        description="Seeker swaps resume/JD roles",
    )


@mcp.tool()
async def score_match(params: ScoreMatchInput) -> dict[str, object]:
    """Score a match heuristically."""
    return score_match_fn(params.query, params.item_text)


@mcp.tool()
async def cite_match(params: CiteMatchInput) -> dict[str, object]:
    """Produce citation-grounded match rationale."""
    candidate = {"id": params.candidate_id, "text": params.candidate_text, "meta": {}}
    counterparty = {"id": "counterparty", "text": params.counterparty_text, "meta": {}}
    rationale = await cite_match_fn(
        params.query,
        candidate,
        counterparty,
        mode=params.mode,
    )
    return rationale.model_dump()


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
