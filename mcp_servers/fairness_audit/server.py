"""Fairness audit MCP server."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from marketplace_matching_agent.fairness.audit import audit
from marketplace_matching_agent.fairness.detconstsort import rebalance
from mcp_servers.bootstrap import configure_stdio_logging

configure_stdio_logging()

mcp = FastMCP("fairness_audit")


class AuditRankedListInput(BaseModel):
    ranked: list[dict[str, object]] = Field(description="Ranked items")
    k: int = Field(default=5, ge=1, description="Audit cutoff")


class RebalanceInput(BaseModel):
    ranked: list[dict[str, object]] = Field(description="Ranked items")
    k: int = Field(default=5, ge=1, description="Output list length")


@mcp.tool()
async def audit_ranked_list(params: AuditRankedListInput) -> dict[str, object]:
    """Audit ranked list for fairness."""
    report = audit(params.ranked, params.k)
    return report.model_dump()


@mcp.tool()
async def rebalance_detconstsort(params: RebalanceInput) -> dict[str, object]:
    """Rebalance list with DetConstSort."""
    rebalanced = rebalance(params.ranked, params.k)
    report = audit(rebalanced, params.k)
    return {"ranked": rebalanced, "report": report.model_dump()}


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
