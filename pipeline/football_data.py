"""
Loader for Football-Data.co.uk CSV files (historical odds + results).

Downloads free CSV files for EPL seasons and loads them into the odds table.
This costs zero API-Football requests.

Football-Data.co.uk EPL CSV URL pattern:
    https://www.football-data.co.uk/mmz4281/{season_short}/E0.csv
    e.g. season 2023/24 → season_short = "2324"

Usage:
    from pipeline.football_data import FootballDataLoader
    loader = FootballDataLoader(db_engine)
    loader.load_season(2023)
"""

import io
import logging
import pathlib
from typing import TYPE_CHECKING

import pandas as pd
import requests
from sqlalchemy import text
from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Raw CSV columns from Football-Data.co.uk used by Edge
_ODDS_COLUMNS = {
    "B365H": "home_odds",
    "B365D": "draw_odds",
    "B365A": "away_odds",
}
_RESULT_COLUMNS = {
    "HomeTeam": "home_team_name",
    "AwayTeam": "away_team_name",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result_raw",  # H / D / A
    "Date": "date_raw",
}

# Football-Data.co.uk uses abbreviated names that differ from API-Football.
# Maps FD name → API-Football name for reliable match lookups.
_FD_TO_API_NAME: dict[str, str] = {
    "Man City":       "Manchester City",
    "Man United":     "Manchester United",
    "Nott'm Forest":  "Nottingham Forest",
    "Sheffield Weds": "Sheffield Wednesday",
    "Sheffield Utd":  "Sheffield Utd",   # same in both, listed for clarity
    "Spurs":          "Tottenham",
    "West Brom":      "West Bromwich Albion",
    "QPR":            "Queens Park Rangers",
    "Huddersfield":   "Huddersfield Town",
    "Stoke":          "Stoke City",
    "Swansea":        "Swansea City",
    "Cardiff":        "Cardiff City",
    "Norwich":        "Norwich City",
    "Middlesbrough":  "Middlesbrough",
    "Blackburn":      "Blackburn Rovers",
    "Sunderland":     "Sunderland",
    "Wigan":          "Wigan Athletic",
    "Reading":        "Reading",
    "Birmingham":     "Birmingham City",
    "Coventry":       "Coventry City",
    "Watford":        "Watford",
    "Preston":        "Preston North End",
    "Ipswich":        "Ipswich",
    "Luton":          "Luton",
    "Burnley":        "Burnley",
    "Brentford":      "Brentford",
}


def _resolve_team_name(fd_name: str) -> str:
    """Return the API-Football equivalent of a Football-Data.co.uk team name."""
    return _FD_TO_API_NAME.get(fd_name, fd_name)


def _season_to_short(season: int) -> str:
    """Convert season year to Football-Data.co.uk short format.

    2023 → "2324" (covers 2023/24 season)
    """
    return f"{str(season)[2:]}{str(season + 1)[2:]}"


def _csv_url(season: int) -> str:
    return f"https://www.football-data.co.uk/mmz4281/{_season_to_short(season)}/E0.csv"


def _normalise_implied_probabilities(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Convert decimal odds to implied probabilities, removing the overround."""
    raw_home = 1 / home_odds
    raw_draw = 1 / draw_odds
    raw_away = 1 / away_odds
    total = raw_home + raw_draw + raw_away
    return raw_home / total, raw_draw / total, raw_away / total


class FootballDataLoader:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def load_season(self, season: int) -> int:
        """Download and load one EPL season from Football-Data.co.uk.

        Returns the number of odds rows inserted.
        Skips rows that are already in the database (upsert by match identity).
        """
        url = _csv_url(season)
        logger.info("Downloading Football-Data.co.uk CSV: %s", url)

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        logger.info("Loaded %d rows for season %d", len(df), season)

        inserted = self._upsert_odds(df, season)
        logger.info("Inserted/updated %d odds rows for season %d", inserted, season)
        return inserted

    def load_from_file(self, csv_path: pathlib.Path, season: int) -> int:
        """Load from a locally saved CSV (useful for offline development)."""
        df = pd.read_csv(csv_path)
        return self._upsert_odds(df, season)

    def _upsert_odds(self, df: pd.DataFrame, season: int) -> int:
        """Match CSV rows to matches table rows and insert odds."""
        inserted = 0

        with self._engine.connect() as conn:
            for _, row in df.iterrows():
                # Skip rows with missing odds
                if any(pd.isna(row.get(col)) for col in _ODDS_COLUMNS):
                    continue

                home_odds = float(row["B365H"])
                draw_odds = float(row["B365D"])
                away_odds = float(row["B365A"])

                implied_home, implied_draw, implied_away = (
                    _normalise_implied_probabilities(home_odds, draw_odds, away_odds)
                )

                # Resolve match_id by joining on team names + season.
                # Translate FD abbreviations to API-Football names before querying.
                api_home = _resolve_team_name(str(row["HomeTeam"]))
                api_away = _resolve_team_name(str(row["AwayTeam"]))

                match = conn.execute(
                    text("""
                        SELECT m.match_id
                        FROM matches m
                        JOIN teams ht ON ht.team_id = m.home_team_id
                        JOIN teams at ON at.team_id = m.away_team_id
                        WHERE ht.name ILIKE :home
                          AND at.name ILIKE :away
                          AND m.season = :season
                        LIMIT 1
                    """),
                    {
                        "home": f"%{api_home}%",
                        "away": f"%{api_away}%",
                        "season": season,
                    },
                ).fetchone()

                if match is None:
                    logger.debug(
                        "No match found for %s vs %s season %d — skipping",
                        row["HomeTeam"],
                        row["AwayTeam"],
                        season,
                    )
                    continue

                conn.execute(
                    text("""
                        INSERT INTO odds
                            (match_id, bookmaker, home_odds, draw_odds, away_odds,
                             implied_home, implied_draw, implied_away)
                        VALUES
                            (:match_id, 'B365', :home_odds, :draw_odds, :away_odds,
                             :implied_home, :implied_draw, :implied_away)
                        ON CONFLICT (match_id, bookmaker) DO UPDATE SET
                            home_odds    = EXCLUDED.home_odds,
                            draw_odds    = EXCLUDED.draw_odds,
                            away_odds    = EXCLUDED.away_odds,
                            implied_home = EXCLUDED.implied_home,
                            implied_draw = EXCLUDED.implied_draw,
                            implied_away = EXCLUDED.implied_away
                    """),
                    {
                        "match_id": match[0],
                        "home_odds": home_odds,
                        "draw_odds": draw_odds,
                        "away_odds": away_odds,
                        "implied_home": implied_home,
                        "implied_draw": implied_draw,
                        "implied_away": implied_away,
                    },
                )
                inserted += 1

            conn.commit()

        return inserted
