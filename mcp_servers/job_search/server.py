"""Job search MCP server."""

from mcp.server.fastmcp import FastMCP

from marketplace_matching_agent.retrieval.hybrid import hybrid_retrieve

mcp = FastMCP("job_search")


@mcp.tool()
async def search_jobs(query: str, k: int = 10) -> dict[str, object]:
    """Search jobs tower."""
    results = await hybrid_retrieve(query, "jobs", k=50, rerank_top_n=k)
    return {"results": results}


@mcp.tool()
async def get_job_by_id(job_id: str) -> dict[str, object]:
    """Fetch job by id from fixture/index."""
    results = await hybrid_retrieve(job_id, "jobs", k=50, rerank_top_n=50)
    for item in results:
        if str(item.get("id")) == job_id:
            return {"job": item}
    return {"job": None}


@mcp.resource("health://status")
def health() -> str:
    return '{"status":"ok","version":"0.1.0"}'


if __name__ == "__main__":
    mcp.run(transport="stdio")
