"""Eval harness for marketplace-matching-agent."""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import structlog
import typer
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evals.cache import SQLiteResponseCache  # noqa: E402
from evals.judge_goodmatch import goodmatch_at_k  # noqa: E402
from evals.metrics import (  # noqa: E402
    four_fifths_pass_rate,
    mrr,
    ndcg_at_k,
    recall_at_k,
)
from marketplace_matching_agent.graph import build_supervisor  # noqa: E402

app = typer.Typer(add_completion=False)
log = structlog.get_logger(__name__)

SEED = 42
EVAL_COST_CAP_USD = float(os.environ.get("EVAL_COST_CAP_USD", "5.00"))


class TrajectoryResult(BaseModel):
    id: str
    mode: str
    latency_ms: float
    ranked_ids: list[str]
    gold_ids: list[str]
    fairness_passed: bool
    fairness_impact_ratio: float
    cost_usd: float


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


async def _run_one(
    graph: object,
    traj: dict[str, object],
    cache: SQLiteResponseCache,
) -> TrajectoryResult:
    _ = cache
    t0 = time.perf_counter()
    out = await graph.ainvoke(  # type: ignore[attr-defined]
        {"mode": traj["mode"], "query": traj["query"], "k": traj["k"]},
        config={"configurable": {"cache": cache}},
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    ranked_ids = [str(item["id"]) for item in out.get("ranked_items", [])]
    fr = out["fairness_report"]
    return TrajectoryResult(
        id=str(traj["id"]),
        mode=str(traj["mode"]),
        latency_ms=dt_ms,
        ranked_ids=ranked_ids,
        gold_ids=[str(x) for x in traj.get("gold_relevant_ids", [])],
        fairness_passed=bool(fr.passed),
        fairness_impact_ratio=float(fr.impact_ratio),
        cost_usd=float(out.get("_cost_usd", 0.0)),
    )


async def _amain(mode: str, limit: int, output_dir: Path) -> None:
    random.seed(SEED)
    cache = SQLiteResponseCache(path=Path(".cache/eval.sqlite"))
    graph = build_supervisor()

    trajectories: list[dict[str, object]] = []
    root = Path("evals/trajectories")
    if mode in ("seeker", "both"):
        trajectories += _load_jsonl(root / "seeker_qs.jsonl")
    if mode in ("recruiter", "both"):
        trajectories += _load_jsonl(root / "recruiter_qs.jsonl")
    trajectories = trajectories[:limit] if limit else trajectories

    results: list[TrajectoryResult] = []
    running_cost = 0.0
    for traj in trajectories:
        if running_cost >= EVAL_COST_CAP_USD:
            log.warning("cost_cap_hit", running_cost=running_cost, cap=EVAL_COST_CAP_USD)
            break
        try:
            r = await _run_one(graph, traj, cache)
            running_cost += r.cost_usd
            results.append(r)
        except Exception as exc:  # noqa: BLE001
            log.error("trajectory_failed", id=traj.get("id"), err=str(exc))

    ndcg = [ndcg_at_k(r.ranked_ids, r.gold_ids, 10) for r in results if r.gold_ids]
    mrr_vals = [mrr(r.ranked_ids, r.gold_ids) for r in results if r.gold_ids]
    rec100 = [recall_at_k(r.ranked_ids, r.gold_ids, 100) for r in results if r.gold_ids]

    gm_at_k = await goodmatch_at_k(results, k=10)
    pass_rate = four_fifths_pass_rate([r.fairness_passed for r in results])

    latencies = sorted(r.latency_ms for r in results)
    p50 = median(latencies) if latencies else 0.0
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0

    summary = {
        "git_sha": _git_sha(),
        "seed": SEED,
        "n": len(results),
        "model_versions": {
            "judge": "claude-sonnet-4-5-20251022",
            "embed": "voyage-3-large",
            "rerank": "rerank-v3.5",
        },
        "ndcg_at_10": sum(ndcg) / len(ndcg) if ndcg else 0.0,
        "mrr": sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0.0,
        "recall_at_100": sum(rec100) / len(rec100) if rec100 else 0.0,
        "goodmatch_at_k": gm_at_k,
        "four_fifths_pass_rate": pass_rate,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "total_cost_usd": running_cost,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "trajectories.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in results)
    )
    log.info("eval_done", **summary)


@app.command()
def main(
    mode: str = typer.Option("both", "--mode"),
    limit: int = typer.Option(0, "--limit", help="0 = all"),
    output_dir: Path = typer.Option(None, "--output-dir"),
) -> None:
    out = output_dir or Path("evals/runs") / _git_sha()
    asyncio.run(_amain(mode, limit, out))


if __name__ == "__main__":
    app()
