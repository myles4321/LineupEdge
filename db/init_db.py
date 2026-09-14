"""
Create the 'edge' database and apply schema.sql.

Usage:
    python db/init_db.py
"""

import os
import sys
import pathlib
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

SCHEMA_PATH = pathlib.Path(__file__).parent / "schema.sql"


def _connect(dbname: str = "postgres") -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=dbname,
        user=DB_USER,
        password=DB_PASSWORD or None,
    )


def create_database() -> None:
    conn = _connect(dbname="postgres")
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)
        )
        if cur.fetchone():
            print(f"Database '{DB_NAME}' already exists — skipping creation.")
        else:
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
            print(f"Database '{DB_NAME}' created.")
    conn.close()


def apply_schema() -> None:
    schema_sql = SCHEMA_PATH.read_text()
    conn = _connect(dbname=DB_NAME)
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()
    conn.close()
    print("Schema applied successfully.")


def verify_tables() -> None:
    conn = _connect(dbname=DB_NAME)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
        )
        tables = [row[0] for row in cur.fetchall()]
    conn.close()
    print(f"Tables in '{DB_NAME}': {', '.join(tables)}")


if __name__ == "__main__":
    create_database()
    apply_schema()
    verify_tables()
