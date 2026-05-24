"""FastAPI server."""

from __future__ import annotations

from typing import Literal, cast

from fastapi import FastAPI
from pydantic import BaseModel

from marketplace_matching_agent.graph import build_supervisor
from marketplace_matching_agent.state import MatchState

app = FastAPI(title="marketplace-matching-agent", version="0.1.0")
_graph = build_supervisor()


class MatchRequest(BaseModel):
    mode: Literal["seeker", "recruiter"]
    query: str
    k: int = 5


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok", "version": "0.1.0"}


@app.post("/match")
async def match(req: MatchRequest) -> dict[str, object]:
    """Run match supervisor."""
    payload = cast(MatchState, {"mode": req.mode, "query": req.query, "k": req.k})
    result = await _graph.ainvoke(payload)
    return dict(result)
