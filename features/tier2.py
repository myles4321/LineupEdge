"""
Tier 2 feature engineering — post-lineup features.

All features are computed using only information available AFTER the confirmed
lineup is released (roughly 1 hour before kickoff) but BEFORE the match
result is known. Temporal ordering is strictly enforced:

  - Formation win rate  : uses only prior matches (date < match_date)
  - Lineup deviation    : based on starter frequency in last 20 prior matches
  - Lineup strength     : based on each starter's ratings from prior matches only
  - Fatigue             : minutes played in [match_date − 14d, match_date)
  - Formation change    : compares to formation in team's most recent prior match

The confirmed starters for the current match ARE used (that is the Tier 2
trigger event), but the current match's results and player statistics are
never touched.

Features produced (per match row):
  Formation          : home_formation, away_formation         (string, e.g. "4-3-3")
  Formation win rate : home/away_formation_win_rate           (0–1 or NaN)
  Formation change   : home/away_formation_change             (0/1 or NaN)
  Lineup strength    : home/away_lineup_strength, lineup_strength_diff
  Key availability   : home/away_key_player_available         (0–1 or NaN)
  Lineup deviation   : home/away_lineup_deviation             (0–1 or NaN)
  Fatigue            : home/away_avg_fatigue, fatigue_diff

Usage:
    from features.tier2 import compute_tier2_features
    from sqlalchemy import create_engine
    import pandas as pd
    tier1_df = pd.read_csv("data/processed/tier1_features.csv", parse_dates=["date"])
    df = compute_tier2_features(create_engine(DATABASE_URL), tier1_df)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── Module-level constants ─────────────────────────────────────────────────────

# Matches in the rolling window used to identify a team's modal starting XI
_MODAL_XI_WINDOW = 20

# Prior matches per player used to estimate their current form rating
_STRENGTH_RATING_WINDOW = 3

# Days before the match used to measure cumulative fatigue
_FATIGUE_DAYS = 14

# Tier 2 numeric model-input columns (categorical formation strings are excluded —
# they are written to CSV as metadata but not passed to the model directly)
TIER2_FEATURE_COLS: list[str] = [
    "home_formation_win_rate",
    "away_formation_win_rate",
    "home_formation_change",
    "away_formation_change",
    "home_lineup_strength",
    "away_lineup_strength",
    "lineup_strength_diff",
    "home_key_player_available",
    "away_key_player_available",
    "home_lineup_deviation",
    "away_lineup_deviation",
    "home_avg_fatigue",
    "away_avg_fatigue",
    "fatigue_diff",
]


# ── Database loading ───────────────────────────────────────────────────────────

def _load_lineup_metadata(engine: Engine) -> pd.DataFrame:
    """Load one row per (match, team): match context, date, and formation.

    Returns columns: lineup_id, match_id, team_id, formation, date,
    result, home_team_id, away_team_id.
    Sorted by date ascending.
    """
    sql = text("""
        SELECT
            l.lineup_id,
            l.match_id,
            l.team_id,
            l.formation,
            m.date,
            m.result,
            m.home_team_id,
            m.away_team_id
        FROM lineups l
        JOIN matches m ON m.match_id = l.match_id
        WHERE m.status = 'FT'
          AND m.result IS NOT NULL
        ORDER BY m.date ASC, l.match_id ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _load_lineup_starters(engine: Engine) -> pd.DataFrame:
    """Load one row per starting player across all completed matches.

    Returns columns: player_id, match_id, team_id.
    Only rows where is_starter = TRUE are included.
    """
    sql = text("""
        SELECT lp.player_id, l.match_id, l.team_id
        FROM lineup_players lp
        JOIN lineups l ON l.lineup_id = lp.lineup_id
        JOIN matches m ON m.match_id = l.match_id
        WHERE lp.is_starter = TRUE
          AND m.status = 'FT'
    """)
    with engine.connect() as conn:
        return pd.read_sql(sql, conn)


def _load_player_stats(engine: Engine) -> pd.DataFrame:
    """Load per-player match statistics with match dates.

    Returns columns: player_id, match_id, team_id, minutes_played, rating, date.
    Sorted by date ascending.
    """
    sql = text("""
        SELECT
            ps.player_id,
            ps.match_id,
            ps.team_id,
            ps.minutes_played,
            ps.rating,
            m.date
        FROM player_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        WHERE m.status = 'FT'
        ORDER BY m.date ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# ── History builders ───────────────────────────────────────────────────────────

def _build_starters_lookup(
    lineup_starters: pd.DataFrame,
) -> dict[tuple[int, int], frozenset]:
    """Return (match_id, team_id) → frozenset of starting player_ids."""
    if lineup_starters.empty:
        return {}
    lookup: dict[tuple[int, int], frozenset] = {}
    for (match_id, team_id), group in lineup_starters.groupby(["match_id", "team_id"]):
        lookup[(int(match_id), int(team_id))] = frozenset(
            int(p) for p in group["player_id"]
        )
    return lookup


def _build_team_lineup_histories(
    lineup_meta: pd.DataFrame,
    starters_lookup: dict[tuple[int, int], frozenset],
) -> dict[int, pd.DataFrame]:
    """Return dict[team_id → DataFrame] of per-match lineup records.

    Each row contains: date, match_id, formation, starters (frozenset of
    player_ids), team_won (1 = this team won, 0 = draw or loss).
    Sorted by date ascending.
    """
    records: dict[int, list[dict]] = {}

    for _, row in lineup_meta.iterrows():
        team_id  = int(row["team_id"])
        match_id = int(row["match_id"])
        is_home  = team_id == int(row["home_team_id"])
        result   = row["result"]
        team_won = 1 if (
            (is_home and result == "H") or (not is_home and result == "A")
        ) else 0

        if team_id not in records:
            records[team_id] = []
        records[team_id].append({
            "date":      row["date"],
            "match_id":  match_id,
            "formation": row["formation"],
            "starters":  starters_lookup.get((match_id, team_id), frozenset()),
            "team_won":  team_won,
        })

    return {
        team_id: pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        for team_id, rows in records.items()
    }


def _build_player_stats_histories(
    player_stats: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Return dict[player_id → DataFrame] of per-match statistics.

    Each row contains: date, match_id, minutes_played, rating.
    Sorted by date ascending.
    """
    if player_stats.empty:
        return {}
    result: dict[int, pd.DataFrame] = {}
    for player_id, group in player_stats.groupby("player_id"):
        result[int(player_id)] = (
            group[["date", "match_id", "minutes_played", "rating"]]
            .sort_values("date")
            .reset_index(drop=True)
        )
    return result


# ── Individual feature computers ─────────────────────────────────────────────

def _formation_win_rate(
    team_hist: pd.DataFrame,
    formation: Optional[str],
    before_date: pd.Timestamp,
) -> float:
    """Win rate across all prior matches where this team used this formation.

    Returns NaN if the formation is unknown or no prior matches used it.
    """
    if team_hist.empty or pd.isna(formation):
        return np.nan
    prior = team_hist[team_hist["date"] < before_date]
    with_formation = prior[prior["formation"] == formation]
    if len(with_formation) == 0:
        return np.nan
    return float(with_formation["team_won"].mean())


def _formation_change(
    team_hist: pd.DataFrame,
    current_formation: Optional[str],
    before_date: pd.Timestamp,
) -> float:
    """1.0 if the current formation differs from the team's last prior match, else 0.0.

    Returns NaN if the team has no prior lineup history or the prior
    formation is unknown.
    """
    if team_hist.empty or pd.isna(current_formation):
        return np.nan
    prior = team_hist[team_hist["date"] < before_date]
    if prior.empty:
        return np.nan
    last_formation = prior.iloc[-1]["formation"]
    if pd.isna(last_formation):
        return np.nan
    return 1.0 if last_formation != current_formation else 0.0


def _lineup_deviation(
    team_hist: pd.DataFrame,
    current_starters: frozenset,
    before_date: pd.Timestamp,
) -> float:
    """Fraction of the team's modal XI absent from today's confirmed lineup.

    Modal XI = the 11 players with the highest starter frequency in the last
    _MODAL_XI_WINDOW prior matches. Deviation 0.0 means the full modal XI
    is starting; 1.0 means none of the modal XI are starting.

    Returns NaN if the team has no prior lineup history.
    """
    if team_hist.empty or not current_starters:
        return np.nan

    prior = team_hist[team_hist["date"] < before_date].tail(_MODAL_XI_WINDOW)
    if prior.empty:
        return np.nan

    all_prior_starters: list[int] = []
    for starter_set in prior["starters"]:
        all_prior_starters.extend(starter_set)

    if not all_prior_starters:
        return np.nan

    counts = Counter(all_prior_starters)
    modal_xi = frozenset(pid for pid, _ in counts.most_common(11))
    absent_count = len(modal_xi - current_starters)
    return absent_count / len(modal_xi)


def _lineup_strength(
    current_starters: frozenset,
    player_stats_hist: dict[int, pd.DataFrame],
    before_date: pd.Timestamp,
) -> float:
    """Mean of each starter's average rating over their last _STRENGTH_RATING_WINDOW
    matches before before_date.

    Players without prior rating data are excluded from the mean.
    Returns NaN if no starters have any prior rating data.
    """
    if not current_starters:
        return np.nan

    player_avg_ratings: list[float] = []
    for player_id in current_starters:
        hist = player_stats_hist.get(player_id)
        if hist is None or hist.empty:
            continue
        recent = hist[hist["date"] < before_date].tail(_STRENGTH_RATING_WINDOW)
        valid_ratings = recent["rating"].dropna()
        if len(valid_ratings) > 0:
            player_avg_ratings.append(float(valid_ratings.mean()))

    return float(np.mean(player_avg_ratings)) if player_avg_ratings else np.nan


def _avg_fatigue(
    current_starters: frozenset,
    player_stats_hist: dict[int, pd.DataFrame],
    before_date: pd.Timestamp,
) -> float:
    """Mean total minutes played by starters in the _FATIGUE_DAYS-day window
    ending at before_date (exclusive).

    Players with no stats data contribute 0 minutes (unknown load = assumed
    rested). Returns NaN if the starters set is empty.
    """
    if not current_starters:
        return np.nan

    window_start = before_date - pd.Timedelta(days=_FATIGUE_DAYS)
    fatigue_by_player: list[float] = []

    for player_id in current_starters:
        hist = player_stats_hist.get(player_id)
        if hist is None or hist.empty:
            fatigue_by_player.append(0.0)
            continue
        in_window = hist[
            (hist["date"] >= window_start) & (hist["date"] < before_date)
        ]
        total_minutes = float(in_window["minutes_played"].fillna(0).sum())
        fatigue_by_player.append(total_minutes)

    return float(np.mean(fatigue_by_player)) if fatigue_by_player else np.nan


# ── Main entry point ───────────────────────────────────────────────────────────

def compute_tier2_features(
    engine: Engine,
    tier1_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all Tier 2 post-lineup features and append them to tier1_df.

    Returns a copy of tier1_df extended with:
    - 2 categorical columns: home_formation, away_formation (metadata only)
    - 14 numeric columns: see TIER2_FEATURE_COLS

    When lineup data is unavailable (empty tables), all Tier 2 columns are
    added as NaN and a clear warning describes which ingestion step to run.

    Temporal invariant: for any match on date D, only matches with date < D
    are used to compute formation win rates, lineup deviation, and change
    features. The confirmed starters for match D are used (that is the Tier 2
    trigger event), but the match's result and post-match stats are never
    referenced.
    """
    all_tier2_cols = ["home_formation", "away_formation"] + TIER2_FEATURE_COLS
    result_df = tier1_df.copy()

    # ── Load lineup metadata ──────────────────────────────────────────────────
    logger.info("Loading lineup metadata from database...")
    lineup_meta = _load_lineup_metadata(engine)

    if lineup_meta.empty:
        logger.warning(
            "No lineup data found in the database. "
            "Run:  python pipeline/ingest.py --step lineups  "
            "then re-run this export. All Tier 2 columns set to NaN."
        )
        for col in all_tier2_cols:
            result_df[col] = np.nan
        return result_df

    logger.info("Loading lineup starters...")
    lineup_starters = _load_lineup_starters(engine)

    logger.info("Loading player statistics...")
    player_stats = _load_player_stats(engine)
    if player_stats.empty:
        logger.warning(
            "No player statistics found. Lineup strength and fatigue will be NaN. "
            "Run:  python pipeline/ingest.py --step player_stats  to ingest."
        )

    # ── Build lookup structures (built once, reused per match) ────────────────
    logger.info("Building starters lookup...")
    starters_lookup = _build_starters_lookup(lineup_starters)

    logger.info("Building team lineup histories...")
    team_lineup_hist = _build_team_lineup_histories(lineup_meta, starters_lookup)

    logger.info("Building player statistics histories...")
    player_stats_hist = _build_player_stats_histories(player_stats)

    # (match_id, team_id) → formation string for the current match
    match_formation: dict[tuple[int, int], Optional[str]] = {}
    for _, row in lineup_meta.iterrows():
        match_formation[(int(row["match_id"]), int(row["team_id"]))] = row["formation"]

    # ── Per-match feature computation ─────────────────────────────────────────
    logger.info("Computing per-match Tier 2 features (%d matches)...", len(result_df))
    tier2_rows: list[dict] = []
    missing_lineup_count = 0

    for _, match in result_df.iterrows():
        match_id   = int(match["match_id"])
        match_date = pd.Timestamp(match["date"])
        if match_date.tzinfo is None:
            match_date = match_date.tz_localize("UTC")
        home_id = int(match["home_team_id"])
        away_id = int(match["away_team_id"])

        home_key = (match_id, home_id)
        away_key = (match_id, away_id)

        home_formation = match_formation.get(home_key)
        away_formation = match_formation.get(away_key)
        home_starters  = starters_lookup.get(home_key, frozenset())
        away_starters  = starters_lookup.get(away_key, frozenset())

        if home_formation is None or away_formation is None:
            missing_lineup_count += 1

        home_hist = team_lineup_hist.get(home_id, pd.DataFrame())
        away_hist = team_lineup_hist.get(away_id, pd.DataFrame())

        # Formation features (prior matches only — date < match_date)
        home_fwr = _formation_win_rate(home_hist, home_formation, match_date)
        away_fwr = _formation_win_rate(away_hist, away_formation, match_date)
        home_fc  = _formation_change(home_hist, home_formation, match_date)
        away_fc  = _formation_change(away_hist, away_formation, match_date)

        # Lineup deviation: fraction of modal XI absent from today's lineup
        home_dev = _lineup_deviation(home_hist, home_starters, match_date)
        away_dev = _lineup_deviation(away_hist, away_starters, match_date)
        # Key player available = complement of deviation (fraction of modal XI present)
        home_kpa = (1.0 - home_dev) if not np.isnan(home_dev) else np.nan
        away_kpa = (1.0 - away_dev) if not np.isnan(away_dev) else np.nan

        # Lineup strength (based on starters' ratings from prior matches)
        home_str = _lineup_strength(home_starters, player_stats_hist, match_date)
        away_str = _lineup_strength(away_starters, player_stats_hist, match_date)
        str_diff = (
            (home_str - away_str)
            if not (np.isnan(home_str) or np.isnan(away_str))
            else np.nan
        )

        # Fatigue (minutes in the 14-day window before match_date)
        home_fat = _avg_fatigue(home_starters, player_stats_hist, match_date)
        away_fat = _avg_fatigue(away_starters, player_stats_hist, match_date)
        fat_diff = (
            (home_fat - away_fat)
            if not (np.isnan(home_fat) or np.isnan(away_fat))
            else np.nan
        )

        tier2_rows.append({
            "match_id":                  match_id,
            "home_formation":            home_formation,
            "away_formation":            away_formation,
            "home_formation_win_rate":   home_fwr,
            "away_formation_win_rate":   away_fwr,
            "home_formation_change":     home_fc,
            "away_formation_change":     away_fc,
            "home_lineup_strength":      home_str,
            "away_lineup_strength":      away_str,
            "lineup_strength_diff":      str_diff,
            "home_key_player_available": home_kpa,
            "away_key_player_available": away_kpa,
            "home_lineup_deviation":     home_dev,
            "away_lineup_deviation":     away_dev,
            "home_avg_fatigue":          home_fat,
            "away_avg_fatigue":          away_fat,
            "fatigue_diff":              fat_diff,
        })

    if missing_lineup_count > 0:
        logger.warning(
            "%d/%d matches are missing lineup data — Tier 2 features will be NaN for those rows.",
            missing_lineup_count, len(result_df),
        )

    tier2_feature_df = pd.DataFrame(tier2_rows)
    result_df = result_df.merge(tier2_feature_df, on="match_id", how="left")

    lineup_coverage = len(result_df) - missing_lineup_count
    logger.info(
        "Tier 2 feature matrix: %d rows × %d columns  (lineup coverage: %d/%d = %.1f%%)",
        len(result_df), len(result_df.columns),
        lineup_coverage, len(result_df),
        100 * lineup_coverage / len(result_df) if len(result_df) > 0 else 0.0,
    )
    return result_df
