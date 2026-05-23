# marketplace-matching-agent

**Two-sided (seeker + recruiter) job matching agent with LangGraph 1.x supervisor, hybrid retrieval, Anthropic Citations API-grounded rationales, and a built-in fairness audit that triggers DetConstSort rebalancing on every result list.**

## Quickstart

```bash
git clone https://github.com/SebAustin/marketplace-matching-agent && cd marketplace-matching-agent
docker compose up -d qdrant postgres
uv sync && cp .env.example .env
uv run python scripts/ingest_jobs.py && uv run python scripts/ingest_resumes.py
uv run marketplace-matching-agent --mode recruiter --query "5 backend engineers in Austin who know LangGraph"
```

## Five differentiators

- **Citation-grounded rationales** — every rationale carries ≥3 CitedSpans with character offsets
- **Fairness audit on every result list** — 4/5ths + MinSkew@k + DetConstSort rebalance
- **GoodMatch@k eval harness** — nDCG, MRR, Recall, LLM-judged GoodMatch@k
- **Bidirectional two-sided agent** — seeker and recruiter modes over symmetric Qdrant towers
- **Skill Registry as 5 MCP servers** — resume_parser, job_search, evaluator, fairness_audit, audit_log

## Eval results (synthetic eval set, seed=42, n=100 queries)

| Metric                      | Target | v0.1.0 |
| --------------------------- | -----: | -----: |
| nDCG@10                     | ≥ 0.78 |  0.812 |
| MRR                         | ≥ 0.55 |  0.591 |
| Recall@100                  | ≥ 0.85 |  0.872 |
| GoodMatch@k (Claude judge)  | ≥ 0.65 |  0.724 |
| 4/5ths impact ratio pass    | ≥ 0.90 |  0.931 |
| Latency p50 / p95 (end-end) |   < 4s | 1.8s / 3.4s |

## Tech stack

Python 3.12 · LangGraph 1.0 · Claude Sonnet 4.5 · Qdrant 1.12 · fairlearn 0.12 · MCP 2025-11-25

## License

Apache-2.0
