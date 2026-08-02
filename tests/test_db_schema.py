import psycopg2
import pytest

from pathml.db.schema import DEFAULT_DSN, init_schema, record_model_version


def _postgres_available() -> bool:
    try:
        with psycopg2.connect(DEFAULT_DSN, connect_timeout=2):
            return True
    except psycopg2.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="local Postgres not running (docker compose -f docker/docker-compose.yml up -d)",
)


def test_init_schema_creates_expected_tables():
    init_schema(DEFAULT_DSN)

    with psycopg2.connect(DEFAULT_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = {row[0] for row in cur.fetchall()}

    assert {"model_versions", "predictions", "feedback"} <= tables


def test_init_schema_is_idempotent():
    init_schema(DEFAULT_DSN)
    init_schema(DEFAULT_DSN)  # must not raise on a second call


def test_record_model_version_inserts_row():
    init_schema(DEFAULT_DSN)
    model_name = "test-record-model-version-insert"

    record_model_version(DEFAULT_DSN, model_name, version=1, alias="staging", mlflow_run_id="run-abc")

    with psycopg2.connect(DEFAULT_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT alias, mlflow_run_id FROM model_versions WHERE model_name = %s AND version = %s",
            (model_name, 1),
        )
        row = cur.fetchone()

    assert row == ("staging", "run-abc")


def test_record_model_version_upserts_alias_on_conflict():
    init_schema(DEFAULT_DSN)
    model_name = "test-record-model-version-upsert"

    record_model_version(DEFAULT_DSN, model_name, version=1, alias="staging", mlflow_run_id="run-abc")
    record_model_version(DEFAULT_DSN, model_name, version=1, alias="production", mlflow_run_id="run-abc")

    with psycopg2.connect(DEFAULT_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT alias FROM model_versions WHERE model_name = %s AND version = %s",
            (model_name, 1),
        )
        rows = cur.fetchall()

    assert rows == [("production",)]  # one row, updated in place -- not a duplicate
