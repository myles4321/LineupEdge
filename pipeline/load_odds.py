"""
Football-Data.co.uk odds loader — Phase 1.

Downloads free historical odds CSVs and loads them into the odds table.
Costs zero API-Football requests.

Requires fixtures to already be ingested (odds are joined to matches by
team name + season, so matches must exist first).

Usage:
    # Load all configured seasons
    python pipeline/load_odds.py

    # Load specific seasons
    python pipeline/load_odds.py --seasons 2022 2023
"""

import argparse
import logging
import pathlib
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL, SEASONS
from pipeline.football_data import FootballDataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _count_completed_matches(engine, season: int) -> int:
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM matches WHERE season = :season AND status = 'FT'"),
            {"season": season},
        )
        return result.scalar() or 0


def load_all_seasons(seasons: list[int], engine) -> None:
    loader = FootballDataLoader(engine)

    for season in seasons:
        completed = _count_completed_matches(engine, season)
        if completed == 0:
            logger.warning(
                "Season %d has no completed matches in DB — run fixtures ingestion first. Skipping.",
                season,
            )
            continue

        logger.info(
            "Loading odds for season %d (%d completed matches in DB)...",
            season, completed,
        )
        try:
            inserted = loader.load_season(season)
            logger.info("Season %d: %d odds rows inserted/updated.", season, inserted)
        except Exception as exc:
            logger.error("Failed to load odds for season %d: %s", season, exc)


def report_odds_coverage(seasons: list[int], engine) -> None:
    """Print a coverage summary after loading."""
    logger.info("--- Odds coverage summary ---")
    with engine.connect() as conn:
        for season in seasons:
            total = conn.execute(
                text("SELECT COUNT(*) FROM matches WHERE season = :s AND status = 'FT'"),
                {"s": season},
            ).scalar() or 0

            with_odds = conn.execute(
                text("""
                    SELECT COUNT(DISTINCT o.match_id)
                    FROM odds o
                    JOIN matches m ON m.match_id = o.match_id
                    WHERE m.season = :s
                """),
                {"s": season},
            ).scalar() or 0

            pct = (with_odds / total * 100) if total > 0 else 0
            logger.info(
                "  Season %d: %d/%d matches have odds (%.1f%%)",
                season, with_odds, total, pct,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load historical bookmaker odds from Football-Data.co.uk"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=SEASONS,
        help="Seasons to load (default: all configured seasons)",
    )
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    load_all_seasons(args.seasons, engine)
    report_odds_coverage(args.seasons, engine)


if __name__ == "__main__":
    main()
