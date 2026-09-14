"""
Tier 1 feature engineering — pre-lineup features.

All features are computed using only information available BEFORE the match
kicks off. Strict temporal ordering is enforced: for any match on date D,
only matches with date < D are used as inputs.

Features produced (per match row):
  Match metadata      : match_id, date, season, matchday, home_team_id, away_team_id
  Rolling form        : {home|away}_form{5|10}_{win|draw|loss}_rate
  xG rolling averages : {home|away}_xg5_{for|against}_avg  (NaN if stats not ingested)
  ELO ratings         : {home|away}_elo  (1500 baseline, K=20, updated chronologically)
  Venue splits        : home_home_win_rate, away_away_win_rate  (last 10 venue matches)
  Rest days           : {home|away}_rest_days  (days since last match; NaN = first of season)
  H2H history         : h2h_home_wins, h2h_draws, h2h_away_wins, h2h_total  (last 5 meetings)
  League table        : {home|away}_{points|position|goal_diff}  (season-to-date before match)
  Points trajectory   : {home|away}_ppg_last5  (points per game in last 5 matches)
  Target              : result ('H'/'D'/'A'), result_label (1/0/-1)

Usage:
    from features.tier1 import compute_tier1_features
    from sqlalchemy import create_engine
    df = compute_tier1_features(create_engine(DATABASE_URL))
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── ELO constants ────────────────────────────────────────────────────────────
_ELO_K = 20
_ELO_INITIAL = 1500.0
_ELO_SCALE = 400.0


# ── Database loading ──────────────────────────────────────────────────────────

def _load_matches(engine: Engine) -> pd.DataFrame:
    """Load all completed matches with team stats joined, sorted oldest-first."""
    sql = text("""
        SELECT
            m.match_id,
            m.date                          AS date,
            m.season,
            m.matchday,
            m.home_team_id,
            m.away_team_id,
            m.home_goals,
            m.away_goals,
            m.result,
            ts_h.xg                         AS home_xg,
            ts_a.xg                         AS away_xg
        FROM matches m
        LEFT JOIN team_stats ts_h
               ON ts_h.match_id = m.match_id AND ts_h.team_id = m.home_team_id
        LEFT JOIN team_stats ts_a
               ON ts_a.match_id = m.match_id AND ts_a.team_id = m.away_team_id
        WHERE m.status = 'FT'
          AND m.result IS NOT NULL
        ORDER BY m.date ASC, m.match_id ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)

    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


# ── ELO computation ───────────────────────────────────────────────────────────

def _expected_score(own_rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - own_rating) / _ELO_SCALE))


def _compute_elo_series(matches: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return two Series (home_elo, away_elo) with ELO BEFORE each match.

    Processes matches in chronological order. Each team starts at 1500.
    Ratings are updated after each match using the standard ELO formula.
    """
    ratings: dict[int, float] = {}
    home_elo_before: list[float] = []
    away_elo_before: list[float] = []

    for _, row in matches.iterrows():
        home_id = int(row["home_team_id"])
        away_id = int(row["away_team_id"])

        home_r = ratings.get(home_id, _ELO_INITIAL)
        away_r = ratings.get(away_id, _ELO_INITIAL)

        home_elo_before.append(home_r)
        away_elo_before.append(away_r)

        # Update ratings based on result
        result = row["result"]
        home_actual = 1.0 if result == "H" else (0.5 if result == "D" else 0.0)
        away_actual = 1.0 - home_actual

        home_expected = _expected_score(home_r, away_r)
        away_expected = _expected_score(away_r, home_r)

        ratings[home_id] = home_r + _ELO_K * (home_actual - home_expected)
        ratings[away_id] = away_r + _ELO_K * (away_actual - away_expected)

    return (
        pd.Series(home_elo_before, index=matches.index, name="home_elo"),
        pd.Series(away_elo_before, index=matches.index, name="away_elo"),
    )


# ── Per-team history helpers ───────────────────────────────────────────────────

def _build_team_histories(matches: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return a dict mapping team_id → all their matches (as both home and away).

    Each row has: date, match_id, team_goals, opponent_goals, result_for_team,
    is_home, xg_for, xg_against, season, home_team_id, away_team_id.
    Sorted by date ascending.
    """
    records: list[dict] = []

    for _, row in matches.iterrows():
        h_id = int(row["home_team_id"])
        a_id = int(row["away_team_id"])

        home_result = row["result"]  # 'H', 'D', 'A'
        away_result = "H" if home_result == "A" else ("A" if home_result == "H" else "D")

        records.append({
            "team_id":        h_id,
            "opponent_id":    a_id,
            "date":           row["date"],
            "match_id":       row["match_id"],
            "season":         row["season"],
            "team_goals":     row["home_goals"],
            "opp_goals":      row["away_goals"],
            "team_won":       1 if home_result == "H" else 0,
            "team_drew":      1 if home_result == "D" else 0,
            "team_lost":      1 if home_result == "A" else 0,
            "is_home":        True,
            "xg_for":         row["home_xg"],
            "xg_against":     row["away_xg"],
            "home_team_id":   h_id,
            "away_team_id":   a_id,
        })
        records.append({
            "team_id":        a_id,
            "opponent_id":    h_id,
            "date":           row["date"],
            "match_id":       row["match_id"],
            "season":         row["season"],
            "team_goals":     row["away_goals"],
            "opp_goals":      row["home_goals"],
            "team_won":       1 if away_result == "H" else 0,
            "team_drew":      1 if away_result == "D" else 0,
            "team_lost":      1 if away_result == "A" else 0,
            "is_home":        False,
            "xg_for":         row["away_xg"],
            "xg_against":     row["home_xg"],
            "home_team_id":   h_id,
            "away_team_id":   a_id,
        })

    history_df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)

    team_histories: dict[int, pd.DataFrame] = {}
    for team_id, group in history_df.groupby("team_id"):
        team_histories[int(team_id)] = group.reset_index(drop=True)

    return team_histories


def _prior_matches(
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    n: Optional[int] = None,
) -> pd.DataFrame:
    """Return rows from history that occurred strictly before before_date."""
    prior = history[history["date"] < before_date]
    if n is not None:
        prior = prior.tail(n)
    return prior


def _prior_home_matches(
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    n: Optional[int] = None,
) -> pd.DataFrame:
    prior = history[(history["date"] < before_date) & (history["is_home"])]
    if n is not None:
        prior = prior.tail(n)
    return prior


def _prior_away_matches(
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    n: Optional[int] = None,
) -> pd.DataFrame:
    prior = history[(history["date"] < before_date) & (~history["is_home"])]
    if n is not None:
        prior = prior.tail(n)
    return prior


# ── Individual feature computers ─────────────────────────────────────────────

def _rolling_form(prior: pd.DataFrame) -> dict[str, float]:
    """Win/draw/loss rates from a slice of prior matches."""
    n = len(prior)
    if n == 0:
        return {"win_rate": np.nan, "draw_rate": np.nan, "loss_rate": np.nan, "count": 0}
    return {
        "win_rate":  prior["team_won"].sum() / n,
        "draw_rate": prior["team_drew"].sum() / n,
        "loss_rate": prior["team_lost"].sum() / n,
        "count":     n,
    }


def _xg_averages(prior: pd.DataFrame) -> dict[str, float]:
    """Average xG for and against from a slice of prior matches."""
    xg_for = prior["xg_for"].dropna()
    xg_against = prior["xg_against"].dropna()
    return {
        "xg_for_avg":     float(xg_for.mean()) if len(xg_for) > 0 else np.nan,
        "xg_against_avg": float(xg_against.mean()) if len(xg_against) > 0 else np.nan,
    }


def _rest_days(history: pd.DataFrame, before_date: pd.Timestamp) -> float:
    """Days since the team's last match before before_date. NaN if no prior match."""
    prior = history[history["date"] < before_date]
    if prior.empty:
        return np.nan
    last_match_date = prior["date"].max()
    delta = before_date - last_match_date
    return float(delta.days)


def _season_table_stats(
    before_date: pd.Timestamp,
    season: int,
    all_team_histories: dict[int, pd.DataFrame],
    team_id: int,
) -> dict[str, float]:
    """League table position and points for this team before before_date."""
    # Compute points for ALL teams in the season up to before_date
    team_points: dict[int, int] = {}
    team_gd: dict[int, int] = {}

    for tid, th in all_team_histories.items():
        season_prior = th[
            (th["date"] < before_date) & (th["season"] == season)
        ]
        if season_prior.empty:
            continue
        points = season_prior["team_won"].sum() * 3 + season_prior["team_drew"].sum()
        gd = int(season_prior["team_goals"].sum()) - int(season_prior["opp_goals"].sum())
        team_points[tid] = int(points)
        team_gd[tid] = gd

    this_points = team_points.get(team_id, 0)
    this_gd = team_gd.get(team_id, 0)

    # Sort teams by points desc, then GD desc to get position
    sorted_teams = sorted(
        team_points.keys(),
        key=lambda t: (team_points[t], team_gd.get(t, 0)),
        reverse=True,
    )
    # Return NaN when no prior season matches — avoids confusing position=20 with
    # "genuinely last place". NaN is consistent with how other cold-start features
    # handle the absence of prior data.
    position = sorted_teams.index(team_id) + 1 if team_id in sorted_teams else np.nan

    return {
        "points":   float(this_points),
        "position": float(position),
        "goal_diff": float(this_gd),
    }


def _points_per_game_last5(prior: pd.DataFrame) -> float:
    """Points per game over the last 5 matches."""
    last5 = prior.tail(5)
    if last5.empty:
        return np.nan
    points = last5["team_won"].sum() * 3 + last5["team_drew"].sum()
    return float(points / len(last5))


def _h2h_stats(
    home_history: pd.DataFrame,
    away_team_id: int,
    before_date: pd.Timestamp,
    n: int = 5,
) -> dict[str, float]:
    """Head-to-head record: last N meetings between these two teams."""
    h2h = home_history[
        (home_history["date"] < before_date) &
        (home_history["opponent_id"] == away_team_id)
    ].tail(n)

    total = len(h2h)
    if total == 0:
        return {
            "h2h_home_wins": np.nan,
            "h2h_draws":     np.nan,
            "h2h_away_wins": np.nan,
            "h2h_total":     0,
        }

    # From home team's perspective in the history
    home_wins = int(h2h["team_won"].sum())
    draws     = int(h2h["team_drew"].sum())
    away_wins = total - home_wins - draws

    return {
        "h2h_home_wins": float(home_wins) / total,
        "h2h_draws":     float(draws) / total,
        "h2h_away_wins": float(away_wins) / total,
        "h2h_total":     float(total),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_tier1_features(engine: Engine) -> pd.DataFrame:
    """Compute all Tier 1 pre-lineup features for every completed match.

    Returns a DataFrame where each row represents one match with all
    home-team and away-team pre-match features. Safe to use directly as
    input to scikit-learn / XGBoost / df-analyze.

    All features use only information available before the match date —
    run leakage_audit.verify() to confirm temporal correctness.
    """
    logger.info("Loading matches from database...")
    matches = _load_matches(engine)
    logger.info("Loaded %d matches.", len(matches))

    logger.info("Computing ELO ratings...")
    home_elo, away_elo = _compute_elo_series(matches)

    logger.info("Building per-team match histories...")
    team_histories = _build_team_histories(matches)

    logger.info("Computing per-match features (this may take a minute)...")
    rows: list[dict] = []

    for idx, match in matches.iterrows():
        match_date  = match["date"]
        season      = int(match["season"])
        home_id     = int(match["home_team_id"])
        away_id     = int(match["away_team_id"])

        home_hist = team_histories.get(home_id, pd.DataFrame())
        away_hist = team_histories.get(away_id, pd.DataFrame())

        # ── Rolling form ──────────────────────────────────────────────────
        # Filter once; tail(n) slices are O(1) views on the already-filtered result.
        home_prior_all  = _prior_matches(home_hist, match_date)
        away_prior_all  = _prior_matches(away_hist, match_date)

        home_form5  = _rolling_form(home_prior_all.tail(5))
        home_form10 = _rolling_form(home_prior_all.tail(10))
        away_form5  = _rolling_form(away_prior_all.tail(5))
        away_form10 = _rolling_form(away_prior_all.tail(10))

        # ── xG averages ───────────────────────────────────────────────────
        home_xg = _xg_averages(home_prior_all.tail(5))
        away_xg = _xg_averages(away_prior_all.tail(5))

        # ── Venue splits ──────────────────────────────────────────────────
        home_home_prior = _prior_home_matches(home_hist, match_date, 10)
        away_away_prior = _prior_away_matches(away_hist, match_date, 10)
        home_venue_form = _rolling_form(home_home_prior)
        away_venue_form = _rolling_form(away_away_prior)

        # ── Rest days ─────────────────────────────────────────────────────
        home_rest = _rest_days(home_hist, match_date)
        away_rest = _rest_days(away_hist, match_date)

        # ── League table ──────────────────────────────────────────────────
        home_table = _season_table_stats(match_date, season, team_histories, home_id)
        away_table = _season_table_stats(match_date, season, team_histories, away_id)

        # ── Points trajectory ─────────────────────────────────────────────
        home_ppg5 = _points_per_game_last5(home_prior_all)
        away_ppg5 = _points_per_game_last5(away_prior_all)

        # ── Head-to-head ──────────────────────────────────────────────────
        h2h = _h2h_stats(home_hist, away_id, match_date)

        # ── Target variable ───────────────────────────────────────────────
        result = match["result"]
        result_label = 1 if result == "H" else (0 if result == "D" else -1)

        rows.append({
            # Metadata (not used as model features)
            "match_id":        match["match_id"],
            "date":            match_date,
            "season":          season,
            "matchday":        match["matchday"],
            "home_team_id":    home_id,
            "away_team_id":    away_id,

            # ELO (computed before this match).
            # home_elo/away_elo are indexed by matches.index, so .loc[idx] is always
            # correct — even if the DataFrame has non-sequential integer labels.
            "home_elo":        home_elo.loc[idx],
            "away_elo":        away_elo.loc[idx],
            "elo_diff":        home_elo.loc[idx] - away_elo.loc[idx],

            # Rolling form — last 5
            "home_form5_win_rate":  home_form5["win_rate"],
            "home_form5_draw_rate": home_form5["draw_rate"],
            "home_form5_loss_rate": home_form5["loss_rate"],
            "away_form5_win_rate":  away_form5["win_rate"],
            "away_form5_draw_rate": away_form5["draw_rate"],
            "away_form5_loss_rate": away_form5["loss_rate"],

            # Rolling form — last 10
            "home_form10_win_rate":  home_form10["win_rate"],
            "home_form10_draw_rate": home_form10["draw_rate"],
            "home_form10_loss_rate": home_form10["loss_rate"],
            "away_form10_win_rate":  away_form10["win_rate"],
            "away_form10_draw_rate": away_form10["draw_rate"],
            "away_form10_loss_rate": away_form10["loss_rate"],

            # xG rolling averages (NaN until team_stats ingested)
            "home_xg5_for_avg":     home_xg["xg_for_avg"],
            "home_xg5_against_avg": home_xg["xg_against_avg"],
            "away_xg5_for_avg":     away_xg["xg_for_avg"],
            "away_xg5_against_avg": away_xg["xg_against_avg"],

            # Venue win rates
            "home_home_win_rate": home_venue_form["win_rate"],
            "away_away_win_rate": away_venue_form["win_rate"],

            # Rest days
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "rest_days_diff": (home_rest - away_rest) if (
                not np.isnan(home_rest) and not np.isnan(away_rest)
            ) else np.nan,

            # League table (season-to-date)
            "home_points":    home_table["points"],
            "home_position":  home_table["position"],
            "home_goal_diff": home_table["goal_diff"],
            "away_points":    away_table["points"],
            "away_position":  away_table["position"],
            "away_goal_diff": away_table["goal_diff"],
            "points_diff":    home_table["points"] - away_table["points"],

            # Points per game (last 5)
            "home_ppg_last5": home_ppg5,
            "away_ppg_last5": away_ppg5,
            "ppg_diff":       (home_ppg5 - away_ppg5) if (
                not np.isnan(home_ppg5) and not np.isnan(away_ppg5)
            ) else np.nan,

            # Head-to-head (last 5 meetings)
            "h2h_home_win_rate": h2h["h2h_home_wins"],
            "h2h_draw_rate":     h2h["h2h_draws"],
            "h2h_away_win_rate": h2h["h2h_away_wins"],
            "h2h_total":         h2h["h2h_total"],

            # Target
            "result":       result,
            "result_label": result_label,
        })

    feature_df = pd.DataFrame(rows)
    logger.info("Feature matrix shape: %s", feature_df.shape)
    return feature_df
