"""CLI entrypoint."""

from __future__ import annotations

import asyncio
import json
from typing import Literal, cast

import typer

from marketplace_matching_agent.graph import build_supervisor

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    mode: str = typer.Option("seeker", "--mode", help="seeker or recruiter"),
    query: str = typer.Option(..., "--query", help="Natural language query"),
    k: int = typer.Option(5, "--k", help="Number of results"),
) -> None:
    """Run marketplace matching agent."""
    graph = build_supervisor()
    result = asyncio.run(
        graph.ainvoke({"mode": cast(Literal["seeker", "recruiter"], mode), "query": query, "k": k})
    )
    typer.echo(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    app()
