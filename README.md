# marketplace-matching-agent

Two-sided job marketplace matching agent built with **LangGraph**. Supports seeker and recruiter modes with hybrid retrieval (BM25 + dense + RRF + rerank), citation-backed rationales, and fairness auditing.

> **Status:** v0.0.0 scaffold — graph nodes, retrieval, and fairness modules are stubbed with `NotImplementedError`. Infrastructure and test wiring are in place for incremental implementation.

## Architecture

```mermaid
flowchart LR
  START --> search --> evaluation --> fairness --> END
```

| Node | Responsibility |
|------|----------------|
| **search** | Hybrid retrieval over jobs or candidate towers |
| **evaluation** | Rank results and produce citation-backed rationales |
| **fairness** | Audit ranked lists for demographic parity and adverse impact |

Shared state is defined in `MatchState` (`mode`, `query`, `k`, retrieved/ranked items, rationales, fairness report, audit hash).

## Tech stack

| Layer | Tools |
|-------|-------|
| Orchestration | LangGraph, langgraph-supervisor |
| LLM / adapters | LangChain, Anthropic, MCP adapters |
| Retrieval | Tantivy (BM25), Qdrant (dense), Cohere rerank, Voyage embeddings |
| Fairness | fairlearn metrics, custom audit pipeline |
| Data | PostgreSQL (audit log), Polars |
| API / CLI | FastAPI, Typer |
| Tooling | uv, ruff, mypy (strict), pytest (85% coverage gate) |

## Prerequisites

- **Python 3.12.4** (see `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Docker** (OrbStack or Docker Desktop) for local Postgres and Qdrant

## Quick start

```bash
# Install dependencies (includes dev extras)
uv sync --all-extras

# Start infrastructure (requires Docker daemon running)
docker compose up -d

# Run the CLI scaffold
uv run marketplace-matching-agent

# Run tests (85% coverage gate)
uv run pytest -q
```

### Docker services

| Service | Image | Ports |
|---------|-------|-------|
| Qdrant | `qdrant/qdrant:v1.12.1` | 6333, 6334 |
| Postgres | `postgres:16.4` | 5432 |

Postgres credentials: user `postgres`, password `matchdev`, database `marketplace`. Schema is initialized from `infra/postgres/init.sql`.

If Docker is not running you will see:

```text
failed to connect to the docker API at unix:///Users/shenry/.orbstack/run/docker.sock
```

Start OrbStack or Docker Desktop, then retry `docker compose up -d`. Tests do not require Docker at the scaffold stage.

## Project layout

```text
marketplace-matching-agent/
├── src/marketplace_matching_agent/
│   ├── state.py              # MatchState, Rationale, FairnessReport models
│   ├── graph.py              # LangGraph supervisor (search → eval → fairness)
│   ├── cli.py                # Typer entrypoint
│   ├── retrieval/hybrid.py   # BM25, dense, RRF, rerank stubs
│   └── fairness/audit.py     # 4/5ths, MinSkew@k, parity stubs
├── tests/                    # NotImplementedError + model coverage tests
├── evals/trajectories/       # Seeker/recruiter eval query sets (JSONL)
├── infra/postgres/           # DB init SQL
├── docker-compose.yml
└── pyproject.toml
```

## Development

```bash
# Lint
uv run ruff check .

# Type check (strict)
uv run mypy --strict src

# Tests with coverage report
uv run pytest -q --cov=src --cov-report=term-missing
```

CI runs the same checks on push/PR to `main` via `.github/workflows/ci.yml`.

## Eval trajectories

Golden query sets for offline evaluation live in `evals/trajectories/`:

- `seeker_qs.jsonl` — 10 job-search queries
- `recruiter_qs.jsonl` — 10 candidate-search queries

Each line is JSON with `id`, `mode`, `query`, `k`, and `expected_skills`.

## Roadmap

- [ ] Implement hybrid retrieval (BM25 + dense + RRF + Cohere rerank)
- [ ] Wire evaluation agent with citation spans
- [ ] Implement fairness audit and rebalancing
- [ ] MCP skill servers and audit log persistence
- [ ] FastAPI serving layer and eval harness

## License

Apache-2.0 (to be added).
