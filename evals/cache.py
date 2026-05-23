"""SQLite-backed response cache for eval runs."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path


class SQLiteResponseCache:
    """Cache keyed by model_id, prompt_hash, doc_hashes."""

    def __init__(self, path: Path | None = None) -> None:
        self.enabled = os.environ.get("EVAL_USE_CACHE", "0") == "1"
        self.path = path or Path(".cache/eval.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._conn.commit()

    def _key(self, model_id: str, prompt: str, docs: list[str]) -> str:
        doc_hashes = hashlib.sha256("".join(docs).encode()).hexdigest()
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        raw = f"{model_id}:{prompt_hash}:{doc_hashes}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model_id: str, prompt: str, docs: list[str]) -> dict[str, object] | None:
        if not self.enabled:
            return None
        key = self._key(model_id, prompt, docs)
        row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def set(self, model_id: str, prompt: str, docs: list[str], value: dict[str, object]) -> None:
        if not self.enabled:
            return
        key = self._key(model_id, prompt, docs)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self._conn.commit()
