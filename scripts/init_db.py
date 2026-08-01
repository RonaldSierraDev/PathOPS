#!/usr/bin/env python3
"""Create the Postgres schema (model_versions, predictions, feedback) if it doesn't exist."""
import argparse

from pathml.db.schema import DEFAULT_DSN, init_schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    args = parser.parse_args()

    init_schema(args.dsn)
    print(f"schema applied to {args.dsn}")


if __name__ == "__main__":
    main()
