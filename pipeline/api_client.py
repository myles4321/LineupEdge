"""
Rate-limited API-Football client with checkpoint/resume.

Enforces the 95 requests/day budget (free plan cap = 100, we leave 5 headroom).
Progress is saved to pipeline/checkpoints/ so ingestion can be stopped and
resumed across multiple days without re-fetching already-stored data.

Usage:
    from pipeline.api_client import APIFootballClient
    client = APIFootballClient()
    data = client.get("/fixtures", params={"league": 39, "season": 2023})
"""

import json
import time
import logging
import pathlib
import datetime
from typing import Any

import requests

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import API_FOOTBALL_KEY, API_FOOTBALL_BASE_URL, API_DAILY_REQUEST_LIMIT

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = pathlib.Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

_RATE_STATE_FILE = CHECKPOINT_DIR / "_rate_state.json"


class DailyLimitReached(Exception):
    """Raised when the daily API request budget is exhausted."""


class APIFootballClient:
    """Thin wrapper around the API-Football v3 REST API.

    Tracks daily request counts in a JSON file so the limit persists across
    Python processes. Raises DailyLimitReached before sending a request that
    would exceed the budget.
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "x-apisports-key": API_FOOTBALL_KEY,
        })
        self._rate_state = self._load_rate_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict:
        """Make a GET request to endpoint (e.g. '/fixtures').

        Returns the full parsed JSON response body.
        Raises DailyLimitReached if the daily budget is exhausted.
        """
        self._check_and_increment()

        url = f"{API_FOOTBALL_BASE_URL}{endpoint}"
        response = self._session.get(url, params=params, timeout=30)

        if response.status_code == 429:
            logger.error("429 Too Many Requests — API-Football rate limit hit.")
            raise DailyLimitReached("API returned 429; daily limit exceeded upstream.")

        response.raise_for_status()
        return response.json()

    @property
    def requests_today(self) -> int:
        self._refresh_rate_state()
        return self._rate_state["count"]

    @property
    def requests_remaining(self) -> int:
        return max(0, API_DAILY_REQUEST_LIMIT - self.requests_today)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, key: str, value: Any) -> None:
        """Persist an arbitrary checkpoint value under a named key.

        Example:
            client.save_checkpoint("lineups_last_match_id", 12345)
        """
        path = CHECKPOINT_DIR / f"{key}.json"
        path.write_text(json.dumps({"value": value}))

    def load_checkpoint(self, key: str, default: Any = None) -> Any:
        """Load a previously saved checkpoint value."""
        path = CHECKPOINT_DIR / f"{key}.json"
        if not path.exists():
            return default
        return json.loads(path.read_text())["value"]

    # ------------------------------------------------------------------
    # Internal rate-state management
    # ------------------------------------------------------------------

    def _load_rate_state(self) -> dict:
        if _RATE_STATE_FILE.exists():
            state = json.loads(_RATE_STATE_FILE.read_text())
            if state.get("date") == str(datetime.date.today()):
                return state
        return self._reset_rate_state()

    def _refresh_rate_state(self) -> None:
        """Reload state from disk (another process may have updated it)."""
        self._rate_state = self._load_rate_state()

    def _reset_rate_state(self) -> dict:
        state = {"date": str(datetime.date.today()), "count": 0}
        _RATE_STATE_FILE.write_text(json.dumps(state))
        return state

    def _check_and_increment(self) -> None:
        self._refresh_rate_state()
        if self._rate_state["count"] >= API_DAILY_REQUEST_LIMIT:
            raise DailyLimitReached(
                f"Daily request budget of {API_DAILY_REQUEST_LIMIT} reached. "
                f"Resume tomorrow. Requests today: {self._rate_state['count']}"
            )
        self._rate_state["count"] += 1
        self._rate_state["date"] = str(datetime.date.today())
        _RATE_STATE_FILE.write_text(json.dumps(self._rate_state))
        logger.debug(
            "API request %d/%d today",
            self._rate_state["count"],
            API_DAILY_REQUEST_LIMIT,
        )
        # Small courtesy delay between requests (avoids burst triggering)
        time.sleep(0.5)
