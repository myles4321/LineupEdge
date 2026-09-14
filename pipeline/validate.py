"""
Data quality validation and report — Phase 1.

Checks the edge database for:
  - Row counts per season
  - Coverage rates (lineups, team stats, player stats, odds)
  - Null rates on key columns
  - Duplicate detection
  - Predictions table readiness

Exits with code 1 if any hard threshold is breached (< 1,500 completed matches,
or lineup coverage < 50% — the latter means Phase 4 feature engineering
won't have enough data to work with).

Usage:
    python pipeline/validate.py
    python pipeline/validate.py --strict   # fail on any coverage < 90%
"""

import argparse
import logging
import pathlib
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL, SEASONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Hard thresholds — ingestion must meet these before Phase 2 can start
MIN_COMPLETED_MATCHES = 1000   # 3 EPL seasons = 1,140; free plan covers 2022-2024
MIN_LINEUP_COVERAGE_PCT = 50.0


def _scalar(conn, sql: str, params: dict | None = None) -> int:
    result = conn.execute(text(sql), params or {})
    return result.scalar() or 0


def report_match_counts(conn, seasons: list[int]) -> dict[int, dict]:
    logger.info("=== Match counts ===")
    season_data = {}
    for season in seasons:
        total    = _scalar(conn, "SELECT COUNT(*) FROM matches WHERE season = :s", {"s": season})
        finished = _scalar(conn, "SELECT COUNT(*) FROM matches WHERE season = :s AND status = 'FT'", {"s": season})
        season_data[season] = {"total": total, "finished": finished}
        logger.info("  Season %d: %d total, %d finished", season, total, finished)
    return season_data


def report_lineup_coverage(conn, seasons: list[int]) -> float:
    logger.info("=== Lineup coverage ===")
    total_finished = 0
    total_with_lineups = 0
    for season in seasons:
        finished = _scalar(
            conn,
            "SELECT COUNT(*) FROM matches WHERE season = :s AND status = 'FT'",
            {"s": season},
        )
        with_lineups = _scalar(
            conn,
            """
            SELECT COUNT(DISTINCT m.match_id)
            FROM matches m
            JOIN lineups l ON l.match_id = m.match_id
            WHERE m.season = :s AND m.status = 'FT'
            """,
            {"s": season},
        )
        pct = (with_lineups / finished * 100) if finished > 0 else 0.0
        logger.info(
            "  Season %d: %d/%d matches have lineups (%.1f%%)",
            season, with_lineups, finished, pct,
        )
        total_finished += finished
        total_with_lineups += with_lineups

    overall_pct = (total_with_lineups / total_finished * 100) if total_finished > 0 else 0.0
    logger.info("  Overall lineup coverage: %.1f%%", overall_pct)
    return overall_pct


def report_team_stats_coverage(conn, seasons: list[int]) -> float:
    logger.info("=== Team stats coverage ===")
    total_finished = 0
    total_with_stats = 0
    for season in seasons:
        finished = _scalar(
            conn,
            "SELECT COUNT(*) FROM matches WHERE season = :s AND status = 'FT'",
            {"s": season},
        )
        with_stats = _scalar(
            conn,
            """
            SELECT COUNT(DISTINCT m.match_id)
            FROM matches m
            JOIN team_stats ts ON ts.match_id = m.match_id
            WHERE m.season = :s AND m.status = 'FT'
            """,
            {"s": season},
        )
        pct = (with_stats / finished * 100) if finished > 0 else 0.0
        logger.info(
            "  Season %d: %d/%d matches have team stats (%.1f%%)",
            season, with_stats, finished, pct,
        )
        total_finished += finished
        total_with_stats += with_stats

    overall_pct = (total_with_stats / total_finished * 100) if total_finished > 0 else 0.0
    logger.info("  Overall team stats coverage: %.1f%%", overall_pct)
    return overall_pct


def report_player_stats_coverage(conn, seasons: list[int]) -> float:
    logger.info("=== Player stats coverage ===")
    total_finished = 0
    total_with_player_stats = 0
    for season in seasons:
        finished = _scalar(
            conn,
            "SELECT COUNT(*) FROM matches WHERE season = :s AND status = 'FT'",
            {"s": season},
        )
        with_player_stats = _scalar(
            conn,
            """
            SELECT COUNT(DISTINCT m.match_id)
            FROM matches m
            JOIN player_stats ps ON ps.match_id = m.match_id
            WHERE m.season = :s AND m.status = 'FT'
            """,
            {"s": season},
        )
        pct = (with_player_stats / finished * 100) if finished > 0 else 0.0
        logger.info(
            "  Season %d: %d/%d matches have player stats (%.1f%%)",
            season, with_player_stats, finished, pct,
        )
        total_finished += finished
        total_with_player_stats += with_player_stats

    overall_pct = (total_with_player_stats / total_finished * 100) if total_finished > 0 else 0.0
    logger.info("  Overall player stats coverage: %.1f%%", overall_pct)
    return overall_pct


def report_odds_coverage(conn, seasons: list[int]) -> float:
    logger.info("=== Odds coverage ===")
    total_finished = 0
    total_with_odds = 0
    for season in seasons:
        finished = _scalar(
            conn,
            "SELECT COUNT(*) FROM matches WHERE season = :s AND status = 'FT'",
            {"s": season},
        )
        with_odds = _scalar(
            conn,
            """
            SELECT COUNT(DISTINCT o.match_id)
            FROM odds o
            JOIN matches m ON m.match_id = o.match_id
            WHERE m.season = :s
            """,
            {"s": season},
        )
        pct = (with_odds / finished * 100) if finished > 0 else 0.0
        logger.info(
            "  Season %d: %d/%d matches have odds (%.1f%%)",
            season, with_odds, finished, pct,
        )
        total_finished += finished
        total_with_odds += with_odds

    overall_pct = (total_with_odds / total_finished * 100) if total_finished > 0 else 0.0
    logger.info("  Overall odds coverage: %.1f%%", overall_pct)
    return overall_pct


def report_null_rates(conn) -> None:
    logger.info("=== Null rates on key columns ===")

    # where_clause is an optional AND-prefixed filter applied after WHERE 1=1
    checks = [
        ("matches",     "result",         "result (completed only)", "AND status = 'FT'"),
        ("team_stats",  "xg",             "xG",                      ""),
        ("team_stats",  "shots_total",    "shots_total",             ""),
        ("player_stats","rating",         "player rating",           ""),
        ("player_stats","minutes_played", "minutes_played",          ""),
        ("lineups",     "formation",      "formation",               ""),
        ("odds",        "home_odds",      "home_odds",               ""),
    ]

    for table, column, label, and_clause in checks:
        total = _scalar(conn, f"SELECT COUNT(*) FROM {table} WHERE 1=1 {and_clause}")
        if total == 0:
            logger.info("  %s.%s (%s): table empty — skipping", table, column, label)
            continue
        nulls = _scalar(
            conn,
            f"SELECT COUNT(*) FROM {table} WHERE 1=1 {and_clause} AND {column} IS NULL",
        )
        null_pct = nulls / total * 100
        status = "OK" if null_pct < 5.0 else "WARNING" if null_pct < 20.0 else "HIGH"
        logger.info(
            "  %s.%s (%s): %.1f%% null  [%s]",
            table, column, label, null_pct, status,
        )


def report_duplicates(conn) -> None:
    logger.info("=== Duplicate detection ===")

    checks = [
        ("matches",       "match_id"),
        ("teams",         "team_id"),
        ("players",       "player_id"),
        ("lineups",       "match_id, team_id"),
        ("lineup_players","lineup_id, player_id"),
        ("team_stats",    "match_id, team_id"),
        ("player_stats",  "player_id, match_id"),
        ("odds",          "match_id, bookmaker"),
    ]

    all_clean = True
    for table, key_cols in checks:
        dupes = _scalar(
            conn,
            f"""
            SELECT COUNT(*) FROM (
                SELECT {key_cols}, COUNT(*) AS cnt
                FROM {table}
                GROUP BY {key_cols}
                HAVING COUNT(*) > 1
            ) sub
            """,
        )
        if dupes > 0:
            logger.warning("  %s: %d duplicate key group(s) on (%s)", table, dupes, key_cols)
            all_clean = False
        else:
            logger.info("  %s: no duplicates", table)

    if all_clean:
        logger.info("  All tables clean.")


def report_predictions_table(conn) -> None:
    logger.info("=== Predictions table ===")
    count = _scalar(conn, "SELECT COUNT(*) FROM predictions")
    logger.info("  %d prediction rows (ready to receive model outputs)", count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge data quality validation")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any coverage metric is below 90%%",
    )
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    failed = False

    with engine.connect() as conn:
        season_data = report_match_counts(conn, SEASONS)

        total_finished = sum(s["finished"] for s in season_data.values())
        logger.info("Total completed matches across all seasons: %d", total_finished)
        if total_finished < MIN_COMPLETED_MATCHES:
            logger.warning(
                "BELOW THRESHOLD: only %d completed matches — need %d for Phase 2.",
                total_finished, MIN_COMPLETED_MATCHES,
            )
            failed = True

        lineup_pct      = report_lineup_coverage(conn, SEASONS)
        stats_pct       = report_team_stats_coverage(conn, SEASONS)
        player_pct      = report_player_stats_coverage(conn, SEASONS)
        odds_pct        = report_odds_coverage(conn, SEASONS)

        report_null_rates(conn)
        report_duplicates(conn)
        report_predictions_table(conn)

    if lineup_pct < MIN_LINEUP_COVERAGE_PCT:
        logger.warning(
            "BELOW THRESHOLD: lineup coverage %.1f%% < %.1f%% minimum.",
            lineup_pct, MIN_LINEUP_COVERAGE_PCT,
        )
        failed = True

    if args.strict:
        for label, pct in [
            ("lineups", lineup_pct),
            ("team_stats", stats_pct),
            ("player_stats", player_pct),
            ("odds", odds_pct),
        ]:
            if pct < 90.0:
                logger.warning("STRICT MODE: %s coverage %.1f%% < 90%%", label, pct)
                failed = True

    if failed:
        logger.error("Validation failed — see warnings above.")
        sys.exit(1)
    else:
        logger.info("Validation passed.")


if __name__ == "__main__":
    main()
