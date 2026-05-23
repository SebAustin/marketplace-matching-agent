"""CLI entrypoint."""

from __future__ import annotations

import typer

app = typer.Typer(help="Marketplace matching agent")


@app.callback(invoke_without_command=True)
def main() -> None:
    """Scaffold CLI; commands added in later milestones."""
    typer.echo("marketplace-matching-agent scaffold")
