"""Central configuration — loaded once at startup from .env."""

import os
from dotenv import load_dotenv

load_dotenv()

# API-Football
API_FOOTBALL_KEY: str = os.environ["API_FOOTBALL_KEY"]
API_FOOTBALL_BASE_URL: str = os.getenv(
    "API_FOOTBALL_BASE_URL", "https://v3.football.api-sports.io"
)

# Database
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
DB_NAME: str = os.getenv("DB_NAME", "edge")
DB_USER: str = os.environ["DB_USER"]
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

DATABASE_URL: str = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# Target data
LEAGUE_ID: int = int(os.getenv("LEAGUE_ID", "39"))
SEASONS: list[int] = [
    int(s) for s in os.getenv("SEASONS", "2020,2021,2022,2023,2024").split(",")
]

# Rate limiting — stay safely under the 100 req/day free plan limit
API_DAILY_REQUEST_LIMIT: int = 95
