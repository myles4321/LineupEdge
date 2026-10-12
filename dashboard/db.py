"""Shared database connection for the Streamlit dashboard.

Uses dotenv directly — does not import config.py, which requires
API_FOOTBALL_KEY (not needed by the dashboard).
"""

import os
import pathlib
import functools

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")


@functools.cache
def _engine():
    user = os.environ.get("DB_USER", "")
    password = os.environ.get("DB_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "edge")
    auth = f"{user}:{password}@" if password else f"{user}@"
    return create_engine(f"postgresql+psycopg2://{auth}{host}:{port}/{name}")


def query(sql: str, **params) -> pd.DataFrame:
    with _engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)
