"""Append-only tamper-evident audit log."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic import BaseModel


class AuditRow(BaseModel):
    """Single audit log row."""

    mode: Literal["seeker", "recruiter"]
    query_hash: str
    prompt_version: str
    model_id: str
    retrieved_doc_ids: list[str]
    rerank_scores: dict[str, float]
    fairness_metrics: dict[str, object]
    fairness_violation: bool
    human_override_flag: bool = False
    prev_hash: str | None = None
    row_hash: str | None = None
    ts: datetime | None = None
    id: int | None = None

    def without_hash(self) -> dict[str, object]:
        """Canonical payload excluding hash fields."""
        return self.model_dump(
            exclude={"prev_hash", "row_hash", "id", "ts"},
            mode="json",
        )


def canonical_json(payload: dict[str, object]) -> str:
    """Serialize payload deterministically."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_row_hash(row: AuditRow, prev_hash: str | None) -> str:
    """Compute SHA-256 hash for row."""
    payload = canonical_json(row.without_hash()) + (prev_hash or "")
    return hashlib.sha256(payload.encode()).hexdigest()


async def _last_hash(conn: AsyncConnection[dict[str, object]]) -> str | None:
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1")
        row = await cur.fetchone()
        return str(row["row_hash"]) if row else None


async def append(conn: AsyncConnection[dict[str, object]], row: AuditRow) -> str:
    """Append audit row with hash chain.

    Args:
        conn: Async psycopg connection.
        row: Audit row to append.

    Returns:
        row_hash of inserted row.
    """
    prev = await _last_hash(conn)
    row.prev_hash = prev
    row.row_hash = compute_row_hash(row, prev)
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO audit_log (
              mode, query_hash, prompt_version, model_id,
              retrieved_doc_ids, rerank_scores, fairness_metrics,
              fairness_violation, human_override_flag, prev_hash, row_hash
            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                row.mode,
                row.query_hash,
                row.prompt_version,
                row.model_id,
                json.dumps(row.retrieved_doc_ids),
                json.dumps(row.rerank_scores),
                json.dumps(row.fairness_metrics),
                row.fairness_violation,
                row.human_override_flag,
                row.prev_hash,
                row.row_hash,
            ),
        )
    await conn.commit()
    return row.row_hash or ""


async def verify_chain(conn: AsyncConnection[dict[str, object]]) -> bool:
    """Verify hash chain integrity."""
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute("SELECT * FROM audit_log ORDER BY id ASC")
        rows = await cur.fetchall()
    prev: str | None = None
    for raw in rows:
        row = AuditRow(
            mode=raw["mode"],
            query_hash=raw["query_hash"],
            prompt_version=raw["prompt_version"],
            model_id=raw["model_id"],
            retrieved_doc_ids=raw["retrieved_doc_ids"],
            rerank_scores=raw["rerank_scores"],
            fairness_metrics=raw["fairness_metrics"],
            fairness_violation=raw["fairness_violation"],
            human_override_flag=raw["human_override_flag"],
            prev_hash=raw["prev_hash"],
            row_hash=raw["row_hash"],
        )
        expected = compute_row_hash(row, prev)
        if row.row_hash != expected or row.prev_hash != prev:
            return False
        prev = row.row_hash
    return True


async def query(
    conn: AsyncConnection[dict[str, object]],
    *,
    prompt_version: str | None = None,
    model_id: str | None = None,
    fairness_violation: bool | None = None,
    since: datetime | None = None,
) -> list[AuditRow]:
    """Query audit log with optional filters."""
    clauses: list[str] = []
    params: list[object] = []
    if prompt_version:
        clauses.append("prompt_version = %s")
        params.append(prompt_version)
    if model_id:
        clauses.append("model_id = %s")
        params.append(model_id)
    if fairness_violation is not None:
        clauses.append("fairness_violation = %s")
        params.append(fairness_violation)
    if since:
        clauses.append("ts >= %s")
        params.append(since)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = "SELECT * FROM audit_log " + where + " ORDER BY id ASC"  # noqa: S608
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(sql, params)
        rows = await cur.fetchall()
    return [
        AuditRow(
            id=raw["id"],
            ts=raw["ts"],
            mode=raw["mode"],
            query_hash=raw["query_hash"],
            prompt_version=raw["prompt_version"],
            model_id=raw["model_id"],
            retrieved_doc_ids=raw["retrieved_doc_ids"],
            rerank_scores=raw["rerank_scores"],
            fairness_metrics=raw["fairness_metrics"],
            fairness_violation=raw["fairness_violation"],
            human_override_flag=raw["human_override_flag"],
            prev_hash=raw["prev_hash"],
            row_hash=raw["row_hash"],
        )
        for raw in rows
    ]
