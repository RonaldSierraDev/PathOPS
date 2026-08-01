"""Postgres schema for prediction logging and the retraining feedback loop.

Three tables, matching the design doc's Week 2/4 plan:
- model_versions: which model version served a prediction
- predictions: every prediction the inference API makes
- feedback: corrections against a prediction, the source of truth a future
  retraining job pulls from instead of a static dataset
"""
import psycopg2

DEFAULT_DSN = "postgresql://pathml:pathml@localhost:5432/pathml"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    alias TEXT,
    mlflow_run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_name, version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    model_version_id INTEGER REFERENCES model_versions(id),
    input_hash TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version_id);

CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    corrected_label TEXT NOT NULL,
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON feedback(prediction_id);
"""


def init_schema(dsn: str = DEFAULT_DSN) -> None:
    """Create the schema if it doesn't already exist. Safe to call repeatedly."""
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        conn.commit()
