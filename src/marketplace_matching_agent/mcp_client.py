"""MCP client wiring via langchain-mcp-adapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

from langchain_mcp_adapters.client import MultiServerMCPClient

from marketplace_matching_agent.audit.log import AuditRow
from marketplace_matching_agent.config import get_settings

ROOT = Path(__file__).resolve().parents[2]


def build_mcp_client() -> MultiServerMCPClient:
    """Create MultiServerMCPClient for all skill registry servers."""
    python = sys.executable
    return MultiServerMCPClient(
        {
            "resume_parser": {
                "command": python,
                "args": ["-m", "mcp_servers.resume_parser.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "job_search": {
                "command": python,
                "args": ["-m", "mcp_servers.job_search.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "evaluator": {
                "command": python,
                "args": ["-m", "mcp_servers.evaluator.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "fairness_audit": {
                "command": python,
                "args": ["-m", "mcp_servers.fairness_audit.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
            "audit_log": {
                "command": python,
                "args": ["-m", "mcp_servers.audit_log.server"],
                "transport": "stdio",
                "cwd": str(ROOT),
            },
        }
    )


def parse_tool_result(result: object) -> dict[str, object]:
    """Normalize LangChain MCP tool output to a dict."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return cast(dict[str, object], json.loads(result))
    if isinstance(result, list):
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", ""))
                return cast(dict[str, object], json.loads(text))
            if hasattr(block, "type") and block.type == "text":
                return cast(dict[str, object], json.loads(str(block.text)))
    raise TypeError(f"unexpected MCP tool result type: {type(result)!r}")


class MCPRegistry:
    """Cached MCP tool registry."""

    def __init__(self, client: MultiServerMCPClient) -> None:
        self._client = client
        self._tools: dict[str, Any] = {}

    async def load(self) -> None:
        if self._tools:
            return
        tools = await self._client.get_tools()
        self._tools = {tool.name: tool for tool in tools}

    async def call(self, tool_name: str, **kwargs: object) -> dict[str, object]:
        await self.load()
        tool = self._tools.get(tool_name)
        if tool is None:
            msg = f"MCP tool not found: {tool_name}"
            raise KeyError(msg)
        raw = await tool.ainvoke({"params": kwargs})
        return parse_tool_result(raw)


_registry: MCPRegistry | None = None


async def get_registry() -> MCPRegistry:
    """Return process-wide MCP registry."""
    global _registry
    if _registry is None:
        _registry = MCPRegistry(build_mcp_client())
    await _registry.load()
    return _registry


def reset_registry() -> None:
    """Clear cached MCP registry (for tests)."""
    global _registry
    _registry = None


async def get_mcp_tools() -> list[object]:
    """Load all MCP tools."""
    registry = await get_registry()
    return list(registry._tools.values())


def build_audit_row(
    *,
    mode: str,
    query_hash: str,
    node: str,
    retrieved_doc_ids: list[str] | None = None,
    rerank_scores: dict[str, float] | None = None,
    fairness_metrics: dict[str, object] | None = None,
    fairness_violation: bool = False,
) -> AuditRow:
    """Build an audit row for MCP append."""
    settings = get_settings()
    metrics = dict(fairness_metrics or {})
    metrics["node"] = node
    return AuditRow(
        mode=cast(Literal["seeker", "recruiter"], mode),
        query_hash=query_hash,
        prompt_version=settings.prompt_version,
        model_id=settings.model_id,
        retrieved_doc_ids=retrieved_doc_ids or [],
        rerank_scores=rerank_scores or {},
        fairness_metrics=metrics,
        fairness_violation=fairness_violation,
    )


async def append_audit_row_mcp(row: AuditRow) -> str:
    """Append audit row via audit_log MCP server."""
    registry = await get_registry()
    result = await registry.call("append_audit_row", row_json=row.model_dump_json())
    row_hash = result.get("row_hash", "offline")
    return str(row_hash)
