"""
Lineup watcher — detects confirmed lineups and triggers Tier 2 recomputation.

Two modes:

  replay  : Process all historical matches with ingested lineups in
             chronological order, logging each lineup-release event.
             Use this to:
             (a) Validate the Tier 2 pipeline end-to-end on historical data.
             (b) Confirm lineup coverage before running export_tier2.py.
             (c) Simulate the matchday trigger sequence for a given season.

  live    : Poll API-Football every 5 minutes for newly confirmed lineups.
             Stores each lineup in the database as it is released, then logs
             a Tier 2 recomputation trigger. Run on matchday as a background
             process, started ~2 hours before kickoff.

Usage:
    python pipeline/lineup_watcher.py replay [--season 2024]
    python pipeline/lineup_watcher.py live   [--gameweek 12]

After replay or live confirms all lineups are stored:
    python features/export_tier2.py
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Live mode: poll this often (seconds) when within _PRE_MATCH_WINDOW_HOURS of kickoff
_POLL_INTERVAL_SECONDS = 300
# Start polling when kickoff is this many hours away
_PRE_MATCH_WINDOW_HOURS = 2


# ── Database helpers ───────────────────────────────────────────────────────────

def _matches_with_lineups(engine, season: int | None = None) -> pd.DataFrame:
    """Return completed matches that have lineup data in the database."""
    season_clause = "AND m.season = :season" if season else ""
    sql = text(f"""
        SELECT DISTINCT
            m.match_id, m.date, m.season, m.matchday,
            m.home_team_id, m.away_team_id,
            t_home.name AS home_team,
            t_away.name AS away_team
        FROM matches m
        JOIN lineups l       ON l.match_id   = m.match_id
        JOIN teams   t_home  ON t_home.team_id = m.home_team_id
        JOIN teams   t_away  ON t_away.team_id = m.away_team_id
        WHERE m.status = 'FT'
          {season_clause}
        ORDER BY m.date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"season": season} if season else {})
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _matches_missing_lineups(engine, season: int | None = None) -> pd.DataFrame:
    """Return completed matches that have no lineup stored yet."""
    season_clause = "AND m.season = :season" if season else ""
    sql = text(f"""
        SELECT m.match_id, m.date, m.season, m.matchday,
               m.home_team_id, m.away_team_id
        FROM matches m
        LEFT JOIN lineups l
               ON l.match_id = m.match_id AND l.team_id = m.home_team_id
        WHERE m.status = 'FT'
          AND l.lineup_id IS NULL
          {season_clause}
        ORDER BY m.date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"season": season} if season else {})
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _upcoming_matches(engine, gameweek: int | None = None) -> pd.DataFrame:
    """Return scheduled (non-finished) matches for live polling."""
    gw_clause = "AND m.matchday = :gameweek" if gameweek else ""
    sql = text(f"""
        SELECT
            m.match_id, m.date, m.matchday,
            m.home_team_id, m.away_team_id,
            t_home.name AS home_team,
            t_away.name AS away_team
        FROM matches m
        JOIN teams t_home ON t_home.team_id = m.home_team_id
        JOIN teams t_away ON t_away.team_id = m.away_team_id
        WHERE m.status != 'FT'
          {gw_clause}
        ORDER BY m.date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"gameweek": gameweek} if gameweek else {})
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _store_lineup_response(
    engine,
    match_id: int,
    lineup_response: list[dict],
) -> None:
    """Persist a single match's lineup API response into the database.

    Handles the lineup, lineup_players, and players tables. Safe to call
    multiple times — all inserts are ON CONFLICT DO NOTHING / DO UPDATE.
    """
    with engine.connect() as conn:
        for lineup in lineup_response:
            team_id = lineup["team"]["id"]

            row = conn.execute(
                text("""
                    INSERT INTO lineups (match_id, team_id, formation)
                    VALUES (:match_id, :team_id, :formation)
                    ON CONFLICT (match_id, team_id)
                    DO UPDATE SET formation = EXCLUDED.formation
                    RETURNING lineup_id
                """),
                {"match_id": match_id, "team_id": team_id,
                 "formation": lineup.get("formation")},
            )
            lineup_id = row.fetchone()[0]

            for is_starter, players in [
                (True,  lineup.get("startXI", [])),
                (False, lineup.get("substitutes", [])),
            ]:
                for entry in players:
                    p = entry["player"]
                    conn.execute(
                        text("""
                            INSERT INTO players (player_id, name, position)
                            VALUES (:player_id, :name, :position)
                            ON CONFLICT (player_id) DO NOTHING
                        """),
                        {"player_id": p["id"], "name": p["name"],
                         "position": p.get("pos")},
                    )
                    conn.execute(
                        text("""
                            INSERT INTO lineup_players
                                (lineup_id, player_id, is_starter, jersey_number, position)
                            VALUES
                                (:lineup_id, :player_id, :is_starter, :number, :position)
                            ON CONFLICT (lineup_id, player_id) DO NOTHING
                        """),
                        {"lineup_id": lineup_id, "player_id": p["id"],
                         "is_starter": is_starter, "number": p.get("number"),
                         "position": p.get("pos")},
                    )
        conn.commit()


# ── Replay mode ────────────────────────────────────────────────────────────────

def run_replay(engine, season: int | None = None) -> None:
    """Simulate the lineup watcher replaying all historical lineup release events.

    Processes matches in chronological order. Logs each confirmed lineup as a
    Tier 2 trigger event, showing what the live watcher would have done.
    """
    logger.info("=== LINEUP WATCHER — REPLAY MODE ===")
    if season:
        logger.info("Season filter: %d", season)

    matches_with = _matches_with_lineups(engine, season=season)
    if matches_with.empty:
        logger.warning(
            "No matches with lineup data found in the database.\n"
            "  Run:  python pipeline/ingest.py --step lineups  to ingest lineup data."
        )
        return

    logger.info(
        "Replaying %d lineup-confirmed events (season filter: %s)...",
        len(matches_with), season or "all",
    )

    for _, match in matches_with.iterrows():
        logger.info(
            "[REPLAY] Lineup confirmed — match_id=%-8d  %s vs %s  (%s)",
            int(match["match_id"]),
            match.get("home_team", match["home_team_id"]),
            match.get("away_team", match["away_team_id"]),
            pd.Timestamp(match["date"]).strftime("%Y-%m-%d"),
        )

    # Coverage summary
    matches_missing = _matches_missing_lineups(engine, season=season)
    total = len(matches_with) + len(matches_missing)
    pct   = 100 * len(matches_with) / total if total > 0 else 0.0

    print("\n" + "=" * 65)
    print("REPLAY SUMMARY")
    print("=" * 65)
    print(f"  Lineup events replayed : {len(matches_with)}")
    print(f"  Matches missing lineup : {len(matches_missing)}")
    print(f"  Coverage               : {len(matches_with)}/{total} ({pct:.1f}%)")

    if matches_missing.empty:
        print("\n  All completed matches have lineup data.")
        print("  Ready for Tier 2 export:")
        print("    python features/export_tier2.py")
    else:
        print(f"\n  {len(matches_missing)} matches still need lineups.")
        print("  Continue ingestion:")
        print("    python pipeline/ingest.py --step lineups")
    print("=" * 65 + "\n")


# ── Live mode ──────────────────────────────────────────────────────────────────

def run_live(engine, gameweek: int | None = None) -> None:
    """Poll API-Football every _POLL_INTERVAL_SECONDS for confirmed lineups.

    Designed to be started 2–3 hours before the first kickoff of a gameweek.
    Runs until interrupted (Ctrl+C) or all lineups for the session are confirmed.
    Requires a valid API key in .env.
    """
    from pipeline.api_client import APIFootballClient, DailyLimitReached

    logger.info("=== LINEUP WATCHER — LIVE MODE ===")
    if gameweek:
        logger.info("Monitoring gameweek %d", gameweek)

    upcoming = _upcoming_matches(engine, gameweek=gameweek)
    if upcoming.empty:
        logger.warning(
            "No upcoming matches found. Check the fixtures table or specify "
            "--gameweek to filter to the correct round."
        )
        return

    logger.info("Monitoring %d upcoming matches:", len(upcoming))
    for _, m in upcoming.iterrows():
        logger.info(
            "  match_id=%-8d  %-22s vs %-22s  kickoff=%s UTC",
            int(m["match_id"]),
            m["home_team"],
            m["away_team"],
            pd.Timestamp(m["date"]).strftime("%Y-%m-%d %H:%M"),
        )

    try:
        client = APIFootballClient()
    except KeyError:
        logger.error(
            "API_FOOTBALL_KEY not set in .env — live mode requires a valid API key. "
            "Copy .env.example to .env and fill in your key."
        )
        sys.exit(1)

    confirmed: set[int] = set()
    logger.info(
        "Polling every %d s. Press Ctrl+C to stop.",
        _POLL_INTERVAL_SECONDS,
    )

    while True:
        now = datetime.now(timezone.utc)

        for _, match in upcoming.iterrows():
            match_id = int(match["match_id"])
            kickoff  = pd.Timestamp(match["date"])

            hours_to_kickoff = (kickoff - pd.Timestamp(now)).total_seconds() / 3600
            if hours_to_kickoff > _PRE_MATCH_WINDOW_HOURS or hours_to_kickoff < -1.0:
                continue
            if match_id in confirmed:
                continue

            # Check if lineup is already stored
            with engine.connect() as conn:
                stored_count = conn.execute(
                    text("SELECT COUNT(*) FROM lineups WHERE match_id = :mid"),
                    {"mid": match_id},
                ).scalar()
            if stored_count and stored_count > 0:
                logger.info("[LIVE] Lineup already stored — match_id=%d", match_id)
                confirmed.add(match_id)
                continue

            if client.requests_remaining == 0:
                logger.warning(
                    "Daily API limit reached — live polling paused. "
                    "Resume tomorrow or check rate state in pipeline/checkpoints/."
                )
                return

            try:
                data = client.get("/fixtures/lineups", params={"fixture": match_id})
            except DailyLimitReached:
                logger.warning("Daily API limit reached during polling.")
                return

            lineup_response = data.get("response", [])
            if not lineup_response:
                logger.debug("No lineup yet for match_id=%d (%.0f min to kickoff)",
                             match_id, hours_to_kickoff * 60)
                continue

            # Lineup confirmed — store it and emit Tier 2 trigger
            _store_lineup_response(engine, match_id, lineup_response)
            confirmed.add(match_id)
            logger.info(
                "[LIVE] Lineup confirmed — match_id=%-8d  %s vs %s  "
                "(%.0f min to kickoff)",
                match_id,
                match["home_team"],
                match["away_team"],
                max(0.0, hours_to_kickoff * 60),
            )
            logger.info(
                "[LIVE] Tier 2 trigger — recompute features now:\n"
                "    python features/export_tier2.py"
            )

        if len(confirmed) >= len(upcoming):
            logger.info("All %d lineups confirmed. Exiting live mode.", len(upcoming))
            break

        logger.debug("Sleeping %d s...", _POLL_INTERVAL_SECONDS)
        time.sleep(_POLL_INTERVAL_SECONDS)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lineup watcher — replay historical events or poll live API"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    replay_parser = sub.add_parser(
        "replay", help="Replay historical lineup-release events"
    )
    replay_parser.add_argument(
        "--season", type=int, default=None,
        help="Filter to a specific season (e.g. 2024)",
    )

    live_parser = sub.add_parser(
        "live", help="Poll API-Football for live lineup releases on matchday"
    )
    live_parser.add_argument(
        "--gameweek", type=int, default=None,
        help="Monitor only matches from this gameweek",
    )

    args = parser.parse_args()
    engine = create_engine(DATABASE_URL)

    if args.mode == "replay":
        run_replay(engine, season=args.season)
    elif args.mode == "live":
        run_live(engine, gameweek=args.gameweek)


if __name__ == "__main__":
    main()
