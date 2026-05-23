"""Audit log tests."""

from __future__ import annotations

import json
import os

import pytest
from psycopg import AsyncConnection

from marketplace_matching_agent.audit.log import AuditRow, append, query, verify_chain

POSTGRES_URL = os.environ.get(
    "POSTGRES_URL", "postgresql://postgres:matchdev@localhost:5432/marketplace"
)


pytestmark = pytest.mark.skipif(
    os.environ.get("SKIP_AUDIT_TESTS") == "1",
    reason="Postgres not available",
)


async def _conn() -> AsyncConnection[dict[str, object]]:
    return await AsyncConnection.connect(POSTGRES_URL)


@pytest.mark.asyncio
async def test_append_and_verify_chain() -> None:
    try:
        conn = await _conn()
    except OSError:
        pytest.skip("postgres unavailable")
    async with conn:
        for i in range(3):
            row = AuditRow(
                mode="seeker",
                query_hash=f"hash{i}",
                prompt_version="v0.1.0",
                model_id="test",
                retrieved_doc_ids=[f"d{i}"],
                rerank_scores={f"d{i}": 0.9},
                fairness_metrics={"passed": True},
                fairness_violation=False,
            )
            await append(conn, row)
        assert await verify_chain(conn) is True


@pytest.mark.asyncio
async def test_query_by_prompt_version() -> None:
    try:
        conn = await _conn()
    except OSError:
        pytest.skip("postgres unavailable")
    async with conn:
        rows = await query(conn, prompt_version="v0.1.0")
        assert isinstance(rows, list)


def test_audit_row_without_hash() -> None:
    row = AuditRow(
        mode="recruiter",
        query_hash="abc",
        prompt_version="v0.1.0",
        model_id="m",
        retrieved_doc_ids=["d1"],
        rerank_scores={"d1": 1.0},
        fairness_metrics={"passed": True},
        fairness_violation=False,
    )
    payload = row.without_hash()
    assert "row_hash" not in payload
    json.dumps(payload)
