-- Append-only audit log with hash chain
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

CREATE INDEX IF NOT EXISTS idx_audit_prompt_version ON audit_log(prompt_version);
CREATE INDEX IF NOT EXISTS idx_audit_model_id ON audit_log(model_id);
CREATE INDEX IF NOT EXISTS idx_audit_violation ON audit_log(fairness_violation)
  WHERE fairness_violation = TRUE;

CREATE OR REPLACE FUNCTION audit_log_no_update() RETURNS trigger AS $$
BEGIN
  RAISE EXCEPTION 'audit_log is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_no_update ON audit_log;
CREATE TRIGGER trg_audit_no_update BEFORE UPDATE OR DELETE ON audit_log
  FOR EACH ROW EXECUTE FUNCTION audit_log_no_update();
