"""
Main ingestion orchestrator — Phase 1.

Pulls historical EPL data from API-Football in priority order:
    1. Fixtures (results, dates)       — ~5 requests for 5 seasons
    2. Lineups per match               — ~1,900 requests over ~19 days
    3. Match statistics per match      — ~1,900 requests over ~19 days
    4. Player stats per match          — ~1,900 requests over ~19 days

Run daily. The script resumes from where it stopped using checkpoints.
Stops automatically when the daily budget is reached.

Usage:
    python pipeline/ingest.py --step fixtures
    python pipeline/ingest.py --step lineups
    python pipeline/ingest.py --step stats
    python pipeline/ingest.py --step player_stats
"""

import argparse
import logging
import pathlib
import re
import sys
from typing import Any

from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL, LEAGUE_ID, SEASONS
from pipeline.api_client import APIFootballClient, DailyLimitReached


def _parse_matchday(raw: str | None) -> int | None:
    """Extract a round number from API-Football's matchday string.

    API returns strings like "Regular Season - 12" or "Premier League - 5".
    We extract the trailing integer and store that.
    Returns None if the string is absent or contains no number.
    """
    if raw is None:
        return None
    match = re.search(r"(\d+)$", str(raw).strip())
    return int(match.group(1)) if match else None


def _parse_stat(stats: dict[str, Any], key: str) -> float | None:
    """Extract a numeric value from an API-Football statistics dict.

    Handles percentage strings (e.g. "84%"), plain numbers, and nulls.
    """
    val = stats.get(key)
    if val is None:
        return None
    if isinstance(val, str) and val.endswith("%"):
        try:
            return float(val.rstrip("%"))
        except ValueError:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def ingest_fixtures(client: APIFootballClient, engine) -> None:
    """Pull all fixtures for configured seasons. ~1 request per season."""
    for season in SEASONS:
        checkpoint_key = f"fixtures_season_{season}"
        if client.load_checkpoint(checkpoint_key):
            logger.info("Season %d fixtures already ingested — skipping.", season)
            continue

        logger.info("Fetching fixtures for season %d...", season)
        try:
            data = client.get("/fixtures", params={"league": LEAGUE_ID, "season": season})
        except DailyLimitReached:
            logger.warning("Daily limit reached during fixture ingestion.")
            return

        fixtures = data.get("response", [])
        logger.info("Got %d fixtures for season %d", len(fixtures), season)

        with engine.connect() as conn:
            for fix in fixtures:
                teams = fix["teams"]
                goals = fix["goals"]
                score = fix["score"]["fulltime"]

                # Upsert team
                for side in ("home", "away"):
                    conn.execute(
                        text("""
                            INSERT INTO teams (team_id, name)
                            VALUES (:team_id, :name)
                            ON CONFLICT (team_id) DO NOTHING
                        """),
                        {"team_id": teams[side]["id"], "name": teams[side]["name"]},
                    )

                # Determine result
                h, a = score.get("home"), score.get("away")
                result = None
                if h is not None and a is not None:
                    result = "H" if h > a else ("A" if a > h else "D")

                conn.execute(
                    text("""
                        INSERT INTO matches
                            (match_id, date, season, matchday,
                             home_team_id, away_team_id,
                             home_goals, away_goals, result, status)
                        VALUES
                            (:match_id, :date, :season, :matchday,
                             :home_team_id, :away_team_id,
                             :home_goals, :away_goals, :result, :status)
                        ON CONFLICT (match_id) DO UPDATE SET
                            home_goals = EXCLUDED.home_goals,
                            away_goals = EXCLUDED.away_goals,
                            result     = EXCLUDED.result,
                            status     = EXCLUDED.status
                    """),
                    {
                        "match_id":     fix["fixture"]["id"],
                        "date":         fix["fixture"]["date"],
                        "season":       season,
                        "matchday":     _parse_matchday(fix["league"].get("round")),
                        "home_team_id": teams["home"]["id"],
                        "away_team_id": teams["away"]["id"],
                        "home_goals":   goals.get("home"),
                        "away_goals":   goals.get("away"),
                        "result":       result,
                        "status":       fix["fixture"]["status"]["short"],
                    },
                )

            conn.commit()

        if fixtures:
            client.save_checkpoint(checkpoint_key, True)
            logger.info("Season %d fixtures committed. Requests remaining today: %d",
                        season, client.requests_remaining)
        else:
            logger.warning(
                "Season %d returned 0 fixtures — free plan may not cover this season. "
                "Not checkpointing so it will be retried.",
                season,
            )


def ingest_lineups(client: APIFootballClient, engine) -> None:
    """Pull confirmed lineups for every completed match. ~1 request per match."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT m.match_id FROM matches m
                LEFT JOIN lineups l ON l.match_id = m.match_id AND l.team_id = m.home_team_id
                WHERE m.status = 'FT' AND l.lineup_id IS NULL
                ORDER BY m.date
            """)
        ).fetchall()

    match_ids = [r[0] for r in rows]
    logger.info("%d matches need lineups.", len(match_ids))

    for match_id in match_ids:
        if client.requests_remaining == 0:
            logger.warning("Daily limit reached — stopping lineup ingestion. Resume tomorrow.")
            return

        try:
            data = client.get("/fixtures/lineups", params={"fixture": match_id})
        except DailyLimitReached:
            logger.warning("Daily limit reached during lineup ingestion.")
            return

        lineups = data.get("response", [])
        if not lineups:
            logger.debug("No lineup data for match %d — skipping.", match_id)
            continue

        with engine.connect() as conn:
            for lineup in lineups:
                team_id = lineup["team"]["id"]

                # Upsert lineup row
                result = conn.execute(
                    text("""
                        INSERT INTO lineups (match_id, team_id, formation)
                        VALUES (:match_id, :team_id, :formation)
                        ON CONFLICT (match_id, team_id) DO UPDATE SET formation = EXCLUDED.formation
                        RETURNING lineup_id
                    """),
                    {
                        "match_id":  match_id,
                        "team_id":   team_id,
                        "formation": lineup.get("formation"),
                    },
                )
                lineup_id = result.fetchone()[0]

                for player in lineup.get("startXI", []):
                    p = player["player"]
                    conn.execute(
                        text("""
                            INSERT INTO players (player_id, name, position)
                            VALUES (:player_id, :name, :position)
                            ON CONFLICT (player_id) DO NOTHING
                        """),
                        {"player_id": p["id"], "name": p["name"], "position": p.get("pos")},
                    )
                    conn.execute(
                        text("""
                            INSERT INTO lineup_players (lineup_id, player_id, is_starter, jersey_number, position)
                            VALUES (:lineup_id, :player_id, TRUE, :number, :position)
                            ON CONFLICT (lineup_id, player_id) DO NOTHING
                        """),
                        {
                            "lineup_id": lineup_id,
                            "player_id": p["id"],
                            "number":    p.get("number"),
                            "position":  p.get("pos"),
                        },
                    )

                for player in lineup.get("substitutes", []):
                    p = player["player"]
                    conn.execute(
                        text("""
                            INSERT INTO players (player_id, name, position)
                            VALUES (:player_id, :name, :position)
                            ON CONFLICT (player_id) DO NOTHING
                        """),
                        {"player_id": p["id"], "name": p["name"], "position": p.get("pos")},
                    )
                    conn.execute(
                        text("""
                            INSERT INTO lineup_players (lineup_id, player_id, is_starter, jersey_number, position)
                            VALUES (:lineup_id, :player_id, FALSE, :number, :position)
                            ON CONFLICT (lineup_id, player_id) DO NOTHING
                        """),
                        {
                            "lineup_id": lineup_id,
                            "player_id": p["id"],
                            "number":    p.get("number"),
                            "position":  p.get("pos"),
                        },
                    )

            conn.commit()

        logger.info(
            "Lineups stored for match %d. Requests remaining today: %d",
            match_id, client.requests_remaining,
        )


def ingest_match_stats(client: APIFootballClient, engine) -> None:
    """Pull team-level statistics for every completed match."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT m.match_id FROM matches m
                LEFT JOIN team_stats ts ON ts.match_id = m.match_id
                WHERE m.status = 'FT' AND ts.id IS NULL
                ORDER BY m.date
            """)
        ).fetchall()

    match_ids = [r[0] for r in rows]
    logger.info("%d matches need statistics.", len(match_ids))

    for match_id in match_ids:
        if client.requests_remaining == 0:
            logger.warning("Daily limit reached — stopping stats ingestion. Resume tomorrow.")
            return

        try:
            data = client.get("/fixtures/statistics", params={"fixture": match_id})
        except DailyLimitReached:
            return

        stats_list = data.get("response", [])
        if not stats_list:
            continue

        with engine.connect() as conn:
            for team_stats in stats_list:
                team_id = team_stats["team"]["id"]
                stats = {s["type"]: s["value"] for s in team_stats["statistics"]}

                conn.execute(
                    text("""
                        INSERT INTO team_stats
                            (match_id, team_id, xg, shots_total, shots_on_target,
                             possession_pct, passes_total, pass_accuracy,
                             corners, fouls, yellow_cards, red_cards)
                        VALUES
                            (:match_id, :team_id, :xg, :shots_total, :shots_on_target,
                             :possession_pct, :passes_total, :pass_accuracy,
                             :corners, :fouls, :yellow_cards, :red_cards)
                        ON CONFLICT (match_id, team_id) DO NOTHING
                    """),
                    {
                        "match_id":        match_id,
                        "team_id":         team_id,
                        "xg":              _parse_stat(stats, "expected_goals"),
                        "shots_total":     _parse_stat(stats, "Total Shots"),
                        "shots_on_target": _parse_stat(stats, "Shots on Goal"),
                        "possession_pct":  _parse_stat(stats, "Ball Possession"),
                        "passes_total":    _parse_stat(stats, "Total passes"),
                        "pass_accuracy":   _parse_stat(stats, "Passes %"),
                        "corners":         _parse_stat(stats, "Corner Kicks"),
                        "fouls":           _parse_stat(stats, "Fouls"),
                        "yellow_cards":    _parse_stat(stats, "Yellow Cards"),
                        "red_cards":       _parse_stat(stats, "Red Cards"),
                    },
                )
            conn.commit()

        logger.info(
            "Stats stored for match %d. Requests remaining today: %d",
            match_id, client.requests_remaining,
        )


def ingest_player_stats(client: APIFootballClient, engine) -> None:
    """Pull individual player statistics for every completed match.

    Uses /fixtures/players endpoint — returns ratings, minutes, goals,
    assists, shots, passes, and tackles per player per match.
    ~1 request per match, ~1,900 requests total for 5 EPL seasons.
    """
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT DISTINCT m.match_id FROM matches m
                LEFT JOIN player_stats ps ON ps.match_id = m.match_id
                WHERE m.status = 'FT' AND ps.id IS NULL
                ORDER BY m.match_id
            """)
        ).fetchall()

    match_ids = [r[0] for r in rows]
    logger.info("%d matches need player stats.", len(match_ids))

    for match_id in match_ids:
        if client.requests_remaining == 0:
            logger.warning("Daily limit reached — stopping player stats ingestion. Resume tomorrow.")
            return

        try:
            data = client.get("/fixtures/players", params={"fixture": match_id})
        except DailyLimitReached:
            logger.warning("Daily limit reached during player stats ingestion.")
            return

        team_entries = data.get("response", [])
        if not team_entries:
            logger.debug("No player stats for match %d — skipping.", match_id)
            continue

        with engine.connect() as conn:
            for team_entry in team_entries:
                team_id = team_entry["team"]["id"]

                for player_entry in team_entry.get("players", []):
                    player = player_entry["player"]
                    player_id = player["id"]

                    # API returns a list of statistics — we only need index 0
                    raw = player_entry.get("statistics", [{}])[0]
                    games   = raw.get("games", {})
                    goals   = raw.get("goals", {})
                    shots   = raw.get("shots", {})
                    passes  = raw.get("passes", {})
                    tackles = raw.get("tackles", {})

                    # Upsert the player record in case they weren't in a lineup
                    conn.execute(
                        text("""
                            INSERT INTO players (player_id, name, position)
                            VALUES (:player_id, :name, :position)
                            ON CONFLICT (player_id) DO NOTHING
                        """),
                        {
                            "player_id": player_id,
                            "name":      player.get("name", ""),
                            "position":  player.get("pos"),
                        },
                    )

                    rating_raw = games.get("rating")
                    try:
                        rating = float(rating_raw) if rating_raw is not None else None
                    except (TypeError, ValueError):
                        rating = None

                    conn.execute(
                        text("""
                            INSERT INTO player_stats
                                (player_id, match_id, team_id, minutes_played, rating,
                                 goals, assists, shots_total, shots_on_target,
                                 passes_total, pass_accuracy, tackles)
                            VALUES
                                (:player_id, :match_id, :team_id, :minutes_played, :rating,
                                 :goals, :assists, :shots_total, :shots_on_target,
                                 :passes_total, :pass_accuracy, :tackles)
                            ON CONFLICT (player_id, match_id) DO NOTHING
                        """),
                        {
                            "player_id":      player_id,
                            "match_id":       match_id,
                            "team_id":        team_id,
                            "minutes_played": games.get("minutes"),
                            "rating":         rating,
                            "goals":          goals.get("total") or 0,
                            "assists":        goals.get("assists") or 0,
                            "shots_total":    shots.get("total") or 0,
                            "shots_on_target": shots.get("on") or 0,
                            "passes_total":   passes.get("total") or 0,
                            "pass_accuracy":  _parse_stat(passes, "accuracy"),
                            "tackles":        tackles.get("total") or 0,
                        },
                    )

            conn.commit()

        logger.info(
            "Player stats stored for match %d. Requests remaining today: %d",
            match_id, client.requests_remaining,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge data ingestion pipeline")
    parser.add_argument(
        "--step",
        choices=["fixtures", "lineups", "stats", "player_stats"],
        required=True,
        help="Which ingestion step to run",
    )
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    client = APIFootballClient()

    logger.info(
        "Starting ingestion step '%s'. Requests remaining today: %d",
        args.step, client.requests_remaining,
    )

    if args.step == "fixtures":
        ingest_fixtures(client, engine)
    elif args.step == "lineups":
        ingest_lineups(client, engine)
    elif args.step == "stats":
        ingest_match_stats(client, engine)
    elif args.step == "player_stats":
        ingest_player_stats(client, engine)

    logger.info("Done. Requests used today: %d/%d", client.requests_today, 95)


if __name__ == "__main__":
    main()
