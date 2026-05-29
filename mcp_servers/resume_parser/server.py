"""Resume parser MCP server."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from marketplace_matching_agent.extraction.resume import extract_skills_from_text, parse_resume_text
from mcp_servers.bootstrap import configure_stdio_logging

configure_stdio_logging()

mcp = FastMCP("resume_parser")


class ParseResumeInput(BaseModel):
    text: str = Field(description="Resume plain text")


class ExtractSkillsInput(BaseModel):
    text: str = Field(description="Resume or profile plain text")


@mcp.tool()
async def parse_resume(params: ParseResumeInput) -> dict[str, object]:
    """Parse resume into structured sections."""
    return parse_resume_text(params.text)


@mcp.tool()
async def extract_skills(params: ExtractSkillsInput) -> dict[str, object]:
    """Extract skill tokens from resume text."""
    return extract_skills_from_text(params.text)


@mcp.resource("health://status")
def health() -> str:
    """Health resource."""
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
