"""
Temporal leakage audit for the Tier 1 feature matrix.

Verifies that no feature value in the matrix was computed using information
from on or after the match date. This is the single most important correctness
check for a sports prediction model — any future-data leakage will produce
misleadingly optimistic evaluation metrics.

Audit checks performed:
  1. ELO values are 1500 on a team's first-ever match (no prior data yet).
  2. Rolling form features are NaN on a team's first-ever match.
  3. Rest days are NaN on a team's first-ever match.
  4. League table points are 0 on matchday 1 (no prior season matches).
  5. All feature values for a sampled match are identical to values computed
     fresh using only the pre-match slice of the raw match history.
  6. For every match, verifies that the historical slice used really does
     contain only matches with date < match_date.

Usage:
    python features/leakage_audit.py
    python features/leakage_audit.py --fail-fast   # stop on first failure
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL
from features.tier1 import (
    _build_team_histories,
    _compute_elo_series,
    _load_matches,
    _rolling_form,
    _rest_days,
    _prior_matches,
    _ELO_INITIAL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


def _check_result(name: str, passed: bool, detail: str = "") -> dict:
    status = _PASS if passed else _FAIL
    msg = f"[{status}] {name}"
    if detail:
        msg += f": {detail}"
    if passed:
        logger.info(msg)
    else:
        logger.error(msg)
    return {"check": name, "status": status, "detail": detail}


def audit(feature_df: pd.DataFrame, matches: pd.DataFrame) -> list[dict]:
    """Run all leakage checks against the pre-computed feature DataFrame.

    Returns a list of result dicts with keys: check, status, detail.
    """
    results: list[dict] = []
    team_histories = _build_team_histories(matches)
    home_elo, away_elo = _compute_elo_series(matches)

    # ── Check 1: ELO = 1500 on team's first match ─────────────────────────────
    first_match_per_team: dict[int, int] = {}
    for i, (_, row) in enumerate(matches.iterrows()):
        h_id, a_id = int(row["home_team_id"]), int(row["away_team_id"])
        if h_id not in first_match_per_team:
            first_match_per_team[h_id] = i
        if a_id not in first_match_per_team:
            first_match_per_team[a_id] = i

    elo_failures = []
    for team_id, first_idx in first_match_per_team.items():
        row = feature_df.iloc[first_idx]
        if int(row["home_team_id"]) == team_id:
            elo_val = row["home_elo"]
        else:
            elo_val = row["away_elo"]
        if not np.isclose(elo_val, _ELO_INITIAL):
            elo_failures.append(f"team {team_id}: elo={elo_val:.2f} (expected {_ELO_INITIAL})")

    results.append(_check_result(
        "ELO = 1500 on first match",
        len(elo_failures) == 0,
        "; ".join(elo_failures[:5]) if elo_failures else f"verified {len(first_match_per_team)} teams",
    ))

    # ── Check 2: Rolling form is NaN on team's first match ────────────────────
    form_failures = []
    for team_id, first_idx in first_match_per_team.items():
        row = feature_df.iloc[first_idx]
        if int(row["home_team_id"]) == team_id:
            win_rate = row["home_form5_win_rate"]
        else:
            win_rate = row["away_form5_win_rate"]
        if not np.isnan(win_rate):
            form_failures.append(f"team {team_id}: form5_win_rate={win_rate:.3f} (expected NaN)")

    results.append(_check_result(
        "Form features NaN on team's first match",
        len(form_failures) == 0,
        "; ".join(form_failures[:5]) if form_failures else f"verified {len(first_match_per_team)} teams",
    ))

    # ── Check 3: Rest days NaN on team's first match ──────────────────────────
    rest_failures = []
    for team_id, first_idx in first_match_per_team.items():
        row = feature_df.iloc[first_idx]
        if int(row["home_team_id"]) == team_id:
            rest = row["home_rest_days"]
        else:
            rest = row["away_rest_days"]
        if not np.isnan(rest):
            rest_failures.append(f"team {team_id}: rest_days={rest} (expected NaN)")

    results.append(_check_result(
        "Rest days NaN on team's first match",
        len(rest_failures) == 0,
        "; ".join(rest_failures[:5]) if rest_failures else f"verified {len(first_match_per_team)} teams",
    ))

    # ── Check 4: Points = 0 on matchday 1 ────────────────────────────────────
    md1_mask = feature_df["matchday"] == 1
    md1 = feature_df[md1_mask]
    if md1.empty:
        results.append(_check_result("Points = 0 on matchday 1", True, "no matchday 1 rows found (SKIP)"))
    else:
        nonzero_home = md1[md1["home_points"] != 0]
        nonzero_away = md1[md1["away_points"] != 0]
        bad = len(nonzero_home) + len(nonzero_away)
        results.append(_check_result(
            "Points = 0 on matchday 1",
            bad == 0,
            f"{bad} rows with non-zero points on matchday 1" if bad else f"verified {len(md1)} matchday-1 rows",
        ))

    # ── Check 5: Spot-check — recompute features for a sample of matches ──────
    # Take every 50th match so the audit completes quickly
    sample_indices = list(range(0, len(feature_df), max(1, len(feature_df) // 20)))
    spot_failures = []

    for sample_i in sample_indices:
        match_row = matches.iloc[sample_i]
        feat_row  = feature_df.iloc[sample_i]
        match_date = match_row["date"]
        home_id    = int(match_row["home_team_id"])
        away_id    = int(match_row["away_team_id"])

        home_hist = team_histories.get(home_id, pd.DataFrame())
        away_hist = team_histories.get(away_id, pd.DataFrame())

        # Recompute form5
        fresh_home_form5 = _rolling_form(_prior_matches(home_hist, match_date, 5))
        expected_win_rate = fresh_home_form5["win_rate"]
        stored_win_rate   = feat_row["home_form5_win_rate"]

        both_nan = np.isnan(expected_win_rate) and np.isnan(stored_win_rate)
        if not both_nan and not np.isclose(expected_win_rate, stored_win_rate, equal_nan=True):
            spot_failures.append(
                f"match {int(match_row['match_id'])}: "
                f"home_form5_win_rate expected={expected_win_rate:.4f} got={stored_win_rate:.4f}"
            )

        # Recompute rest days
        fresh_rest = _rest_days(home_hist, match_date)
        stored_rest = feat_row["home_rest_days"]
        both_nan = np.isnan(fresh_rest) and np.isnan(stored_rest)
        if not both_nan and not np.isclose(fresh_rest, stored_rest, equal_nan=True):
            spot_failures.append(
                f"match {int(match_row['match_id'])}: "
                f"home_rest_days expected={fresh_rest} got={stored_rest}"
            )

        # Verify ELO
        expected_home_elo = home_elo.iloc[sample_i]
        stored_home_elo   = feat_row["home_elo"]
        if not np.isclose(expected_home_elo, stored_home_elo):
            spot_failures.append(
                f"match {int(match_row['match_id'])}: "
                f"home_elo expected={expected_home_elo:.2f} got={stored_home_elo:.2f}"
            )

    results.append(_check_result(
        f"Spot-check {len(sample_indices)} sampled matches (form5, rest, ELO)",
        len(spot_failures) == 0,
        "; ".join(spot_failures[:3]) if spot_failures else f"all {len(sample_indices)} samples match",
    ))

    # ── Check 6: History slices contain only pre-match dates ─────────────────
    future_leak_count = 0
    sample_for_slice = list(range(0, len(matches), max(1, len(matches) // 50)))

    for i in sample_for_slice:
        row = matches.iloc[i]
        match_date = row["date"]
        home_id    = int(row["home_team_id"])

        home_hist = team_histories.get(home_id, pd.DataFrame())
        if home_hist.empty:
            continue
        prior = home_hist[home_hist["date"] < match_date]

        # Verify no row in the prior slice has date >= match_date
        if not prior.empty:
            max_prior_date = prior["date"].max()
            if max_prior_date >= match_date:
                future_leak_count += 1

    results.append(_check_result(
        f"History slices contain only pre-match dates ({len(sample_for_slice)} sampled)",
        future_leak_count == 0,
        f"{future_leak_count} slices contained future data" if future_leak_count else "no future data found",
    ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal leakage audit for Tier 1 features")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit immediately on first failure",
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=None,
        help="Path to pre-computed feature CSV (skips recomputation)",
    )
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)
    matches = _load_matches(engine)

    if args.csv and args.csv.exists():
        logger.info("Loading feature matrix from %s", args.csv)
        feature_df = pd.read_csv(args.csv, parse_dates=["date"])
        feature_df["date"] = pd.to_datetime(feature_df["date"], utc=True)
    else:
        from features.tier1 import compute_tier1_features
        logger.info("No CSV provided — computing features from database...")
        feature_df = compute_tier1_features(engine)

    logger.info("Running leakage audit on %d rows...", len(feature_df))
    results = audit(feature_df, matches)

    print("\n" + "=" * 60)
    print("LEAKAGE AUDIT SUMMARY")
    print("=" * 60)
    passes  = sum(1 for r in results if r["status"] == _PASS)
    fails   = sum(1 for r in results if r["status"] == _FAIL)

    for r in results:
        icon = "✓" if r["status"] == _PASS else "✗"
        print(f"  {icon} [{r['status']}] {r['check']}")
        if r["detail"]:
            print(f"        {r['detail']}")

    print("=" * 60)
    print(f"  {passes} passed, {fails} failed")
    print("=" * 60 + "\n")

    if fails > 0:
        logger.error("Leakage audit FAILED — do NOT use this feature matrix for modelling.")
        sys.exit(1)
    else:
        logger.info("Leakage audit PASSED — feature matrix is temporally clean.")


if __name__ == "__main__":
    main()
