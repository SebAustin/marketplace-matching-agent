"""Pytest configuration."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_indexes() -> None:
    index_root = Path("data/indexes")
    if index_root.exists():
        shutil.rmtree(index_root, ignore_errors=True)
    yield
    if index_root.exists():
        shutil.rmtree(index_root, ignore_errors=True)


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    monkeypatch.setenv("COHERE_API_KEY", "")
    monkeypatch.setenv("QDRANT_URL", os.environ.get("QDRANT_URL", "http://localhost:6333"))
