"""Evaluator MCP server."""


from mcp.server.fastmcp import FastMCP

from marketplace_matching_agent.extraction.citations import cite_match

mcp = FastMCP("evaluator")


@mcp.tool()
async def score_match(query: str, item_text: str) -> dict[str, object]:
    """Score a match heuristically."""
    overlap = len(set(query.lower().split()) & set(item_text.lower().split()))
    return {"score": overlap / max(len(query.split()), 1)}


@mcp.tool()
async def cite_match_tool(
    query: str,
    candidate_text: str,
    counterparty_text: str,
    candidate_id: str = "unknown",
) -> dict[str, object]:
    """Produce citation-grounded rationale."""
    candidate = {"id": candidate_id, "text": candidate_text, "meta": {}}
    counterparty = {"id": "counterparty", "text": counterparty_text, "meta": {}}
    rationale = await cite_match(query, candidate, counterparty)
    return rationale.model_dump()


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
