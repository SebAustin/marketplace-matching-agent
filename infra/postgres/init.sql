-- Scaffold placeholder; expand in later milestones.
CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  mode TEXT NOT NULL CHECK (mode IN ('seeker', 'recruiter')),
  row_hash TEXT NOT NULL UNIQUE
);
