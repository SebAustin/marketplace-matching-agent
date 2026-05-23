from typer.testing import CliRunner

from marketplace_matching_agent.cli import app


def test_cli_invokes() -> None:
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "scaffold" in result.stdout
