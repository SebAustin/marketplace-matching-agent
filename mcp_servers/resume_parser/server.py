"""Resume parser MCP server."""


import re

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("resume_parser")


@mcp.tool()
async def parse_resume(text: str) -> dict[str, object]:
    """Parse resume into structured sections."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {"sections": lines[:10], "word_count": len(text.split())}


@mcp.tool()
async def extract_skills(text: str) -> dict[str, object]:
    """Extract skill tokens from resume text."""
    tokens = set(re.findall(r"\b[A-Z][a-zA-Z+#.]+\b", text))
    skills = sorted(tokens)[:20]
    return {"skills": skills}


@mcp.resource("health://status")
def health() -> str:
    """Health resource."""
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
