"""Job search MCP server."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from marketplace_matching_agent.retrieval.hybrid import hybrid_retrieve
from mcp_servers.bootstrap import configure_stdio_logging

configure_stdio_logging()

mcp = FastMCP("job_search")


class SearchJobsInput(BaseModel):
    query: str = Field(description="Natural-language search query")
    k: int = Field(default=10, ge=1, le=50, description="Number of results")
    tower: str = Field(default="jobs", description="Retrieval tower: jobs or candidates")


class GetJobByIdInput(BaseModel):
    job_id: str = Field(description="Document identifier")


@mcp.tool()
async def search_jobs(params: SearchJobsInput) -> dict[str, object]:
    """Search jobs or candidates tower."""
    results = await hybrid_retrieve(
        params.query,
        params.tower,
        k=50,
        rerank_top_n=params.k,
    )
    return {"results": results}


@mcp.tool()
async def get_job_by_id(params: GetJobByIdInput) -> dict[str, object]:
    """Fetch a document by id from the jobs tower."""
    results = await hybrid_retrieve(params.job_id, "jobs", k=50, rerank_top_n=50)
    for item in results:
        if str(item.get("id")) == params.job_id:
            return {"job": item}
    return {"job": None}


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
