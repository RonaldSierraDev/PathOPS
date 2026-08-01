import psycopg2
import pytest

from pathml.db.schema import DEFAULT_DSN, init_schema


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
