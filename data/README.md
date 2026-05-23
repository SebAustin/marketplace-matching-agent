# Data directory

- `jobs/` — LinkedIn job postings parquet (not committed)
- `resumes/` — Kaggle + synthetic balanced resumes
- `synthetic/` — Bertrand & Mullainathan name lists for fairness eval

Run ingest scripts after placing source data:

```bash
uv run python scripts/ingest_jobs.py
uv run python scripts/ingest_resumes.py
```
