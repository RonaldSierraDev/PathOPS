"""Postgres schema for prediction logging and the retraining feedback loop.

Three tables, matching the design doc's Week 2/4 plan:
- model_versions: which model version served a prediction
- predictions: every prediction the inference API makes
- feedback: corrections against a prediction, the source of truth a future
  retraining job pulls from instead of a static dataset
"""
import psycopg2

DEFAULT_DSN = "postgresql://pathml:pathml@localhost:5432/pathml"

# Shared between the inference API (which writes predictions/feedback) and
# the training pipeline (which reads corrections back out for retraining).
LABELS = ("no_tumor", "tumor")
PREDICTION_IMAGES_PREFIX = "predictions"


def prediction_image_key(input_hash: str) -> str:
    """S3 key an uploaded image is stored under, keyed by its content hash."""
    return f"{PREDICTION_IMAGES_PREFIX}/{input_hash}.png"

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
    latency_ms DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predictions_model_version ON predictions(model_version_id);

-- Runs against a table that may already exist in a deployed environment
-- (this schema shipped once already without this column) -- ADD COLUMN IF
-- NOT EXISTS patches it in place instead of requiring a drop/recreate.
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS latency_ms DOUBLE PRECISION;

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


UPSERT_MODEL_VERSION_SQL = """
INSERT INTO model_versions (model_name, version, alias, mlflow_run_id)
VALUES (%s, %s, %s, %s)
ON CONFLICT (model_name, version)
DO UPDATE SET alias = EXCLUDED.alias, mlflow_run_id = EXCLUDED.mlflow_run_id
"""


def record_model_version(
    dsn: str, model_name: str, version: int, alias: str, mlflow_run_id: str
) -> None:
    """Record (or update) a model registry version's alias, e.g. after promoting it to 'production'.

    Upserts on (model_name, version), the table's unique key, so calling this
    again for the same version -- e.g. re-promoting or re-staging it -- just
    updates the alias rather than erroring or duplicating the row.
    """
    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(UPSERT_MODEL_VERSION_SQL, (model_name, version, alias, mlflow_run_id))
        conn.commit()
