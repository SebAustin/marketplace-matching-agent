"""Audit log MCP server."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP
from psycopg import AsyncConnection, OperationalError
from pydantic import BaseModel, Field

from marketplace_matching_agent.audit.log import AuditRow, append, query
from marketplace_matching_agent.config import get_settings
from mcp_servers.bootstrap import configure_stdio_logging

configure_stdio_logging()

mcp = FastMCP("audit_log")


class AppendAuditRowInput(BaseModel):
    row_json: str = Field(description="Serialized AuditRow JSON")


class QueryAuditInput(BaseModel):
    prompt_version: str | None = Field(default=None, description="Optional prompt version filter")


@mcp.tool()
async def append_audit_row(params: AppendAuditRowInput) -> dict[str, object]:
    """Append audit row to Postgres."""
    settings = get_settings()
    data = json.loads(params.row_json)
    row = AuditRow.model_validate(data)
    try:
        async with await AsyncConnection.connect(settings.postgres_url) as conn:
            row_hash = await append(conn, row)
        return {"row_hash": row_hash}
    except (OSError, OperationalError):
        return {"row_hash": "offline"}


@mcp.tool()
async def query_audit_by_filter(params: QueryAuditInput) -> dict[str, object]:
    """Query audit log rows."""
    settings = get_settings()
    try:
        async with await AsyncConnection.connect(settings.postgres_url) as conn:
            rows = await query(conn, prompt_version=params.prompt_version)
        return {"rows": [row.model_dump(mode="json") for row in rows]}
    except (OSError, OperationalError):
        return {"rows": []}


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
