-- Audit schema (mirrors infra/postgres/init.sql)
CREATE TABLE IF NOT EXISTS audit_log (
  id            BIGSERIAL PRIMARY KEY,
  ts            TIMESTAMPTZ NOT NULL DEFAULT now(),
  mode          TEXT NOT NULL CHECK (mode IN ('seeker','recruiter')),
  query_hash    TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  model_id      TEXT NOT NULL,
  retrieved_doc_ids JSONB NOT NULL,
  rerank_scores JSONB NOT NULL,
  fairness_metrics JSONB NOT NULL,
  fairness_violation BOOLEAN NOT NULL,
  human_override_flag BOOLEAN NOT NULL DEFAULT FALSE,
  prev_hash     TEXT,
  row_hash      TEXT NOT NULL UNIQUE
);
