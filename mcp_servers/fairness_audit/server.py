"""Fairness audit MCP server."""

from mcp.server.fastmcp import FastMCP

from marketplace_matching_agent.fairness.audit import audit
from marketplace_matching_agent.fairness.detconstsort import rebalance

mcp = FastMCP("fairness_audit")


@mcp.tool()
async def audit_ranked_list(ranked: list[dict[str, object]], k: int = 5) -> dict[str, object]:
    """Audit ranked list for fairness."""
    report = audit(ranked, k)
    return report.model_dump()


@mcp.tool()
async def rebalance_detconstsort(ranked: list[dict[str, object]], k: int = 5) -> dict[str, object]:
    """Rebalance list with DetConstSort."""
    rebalanced = rebalance(ranked, k)
    report = audit(rebalanced, k)
    return {"ranked": rebalanced, "report": report.model_dump()}


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
