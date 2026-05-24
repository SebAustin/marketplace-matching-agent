"""Audit log MCP server."""

import json

from mcp.server.fastmcp import FastMCP
from psycopg import AsyncConnection

from marketplace_matching_agent.audit.log import AuditRow, append, query
from marketplace_matching_agent.config import get_settings

mcp = FastMCP("audit_log")


@mcp.tool()
async def append_audit_row(row_json: str) -> dict[str, object]:
    """Append audit row to Postgres."""
    settings = get_settings()
    data = json.loads(row_json)
    row = AuditRow.model_validate(data)
    async with await AsyncConnection.connect(settings.postgres_url) as conn:
        row_hash = await append(conn, row)
    return {"row_hash": row_hash}


@mcp.tool()
async def query_audit_by_filter(prompt_version: str | None = None) -> dict[str, object]:
    """Query audit log rows."""
    settings = get_settings()
    async with await AsyncConnection.connect(settings.postgres_url) as conn:
        rows = await query(conn, prompt_version=prompt_version)
    return {"rows": [r.model_dump(mode="json") for r in rows]}


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
