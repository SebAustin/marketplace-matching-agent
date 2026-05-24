"""MCP client wiring via langchain-mcp-adapters."""

from __future__ import annotations

from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

ROOT = Path(__file__).resolve().parents[2]


def build_mcp_client() -> MultiServerMCPClient:
    """Create MultiServerMCPClient for all skill registry servers."""
    return MultiServerMCPClient(
        {
            "resume_parser": {
                "command": "python",
                "args": ["-m", "mcp_servers.resume_parser.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "job_search": {
                "command": "python",
                "args": ["-m", "mcp_servers.job_search.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "evaluator": {
                "command": "python",
                "args": ["-m", "mcp_servers.evaluator.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "fairness_audit": {
                "command": "python",
                "args": ["-m", "mcp_servers.fairness_audit.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "audit_log": {
                "command": "python",
                "args": ["-m", "mcp_servers.audit_log.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
        }
    )


async def get_mcp_tools() -> list[object]:
    """Load all MCP tools."""
    client = build_mcp_client()
    tools = await client.get_tools()
    return list(tools)
