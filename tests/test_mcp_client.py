"""MCP client async tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from marketplace_matching_agent.mcp_client import get_mcp_tools


@pytest.mark.asyncio
async def test_get_mcp_tools() -> None:
    with patch(
        "marketplace_matching_agent.mcp_client.MultiServerMCPClient.get_tools",
        new=AsyncMock(return_value=[]),
    ):
        tools = await get_mcp_tools()
    assert tools == []
