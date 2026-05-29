"""Shared MCP server bootstrap (stderr logging for stdio transport)."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_stdio_logging() -> None:
    """Route all logging to stderr so stdout stays JSON-RPC clean."""
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )
