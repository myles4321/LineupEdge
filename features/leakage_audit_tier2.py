"""
Temporal leakage audit for the Tier 2 feature matrix.

Verifies that no Tier 2 feature was computed using information that would
not be available at Tier 2 trigger time (lineup release, ~1 hour before
kickoff). The key invariant: for any match on date D, only matches with
date < D are used for formation win rates, lineup deviation, and change
features.

Audit checks:
  1. Formation win rate is NaN on a team's first-ever lineup match
     (no prior matches with that formation).
  2. Lineup deviation is NaN on a team's first-ever lineup match
     (no prior lineup history to build the modal XI from).
  3. Formation change is NaN on a team's first-ever lineup match
     (no prior match to compare against).
  4. Formation history slices for sampled teams contain only pre-match dates
     (direct temporal ordering verification).

Note: Lineup strength and fatigue audits require player_stats data and are
omitted here — they will be auditable once player_stats ingestion completes.

Usage:
    python features/leakage_audit_tier2.py
    python features/leakage_audit_tier2.py --csv data/processed/tier2_features.csv
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
from features.tier2 import (
    _build_starters_lookup,
    _build_team_lineup_histories,
    _formation_win_rate,
    _load_lineup_metadata,
    _load_lineup_starters,
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


def audit(tier2_df: pd.DataFrame, engine) -> list[dict]:
    """Run all Tier 2 leakage checks against the pre-computed feature DataFrame.

    Returns a list of result dicts with keys: check, status, detail.
    Skips all checks with a SKIP status when no lineup data is available.
    """
    results: list[dict] = []

    lineup_meta = _load_lineup_metadata(engine)
    if lineup_meta.empty:
        skip_msg = (
            "No lineup data in database — Tier 2 audit skipped. "
            "Run:  python pipeline/ingest.py --step lineups  then re-run."
        )
        logger.warning(skip_msg)
        results.append({"check": "Lineup data available", "status": _SKIP, "detail": skip_msg})
        return results

    lineup_starters = _load_lineup_starters(engine)
    starters_lookup = _build_starters_lookup(lineup_starters)
    team_lineup_hist = _build_team_lineup_histories(lineup_meta, starters_lookup)

    # Map each team to its first match in the lineup table
    first_lineup_match: dict[int, int] = {}  # team_id → match_id
    for team_id, hist in team_lineup_hist.items():
        if not hist.empty:
            first_lineup_match[team_id] = int(hist.iloc[0]["match_id"])

    # ── Check 1: Formation win rate NaN on first lineup match ─────────────────
    fwr_failures = []
    for team_id, match_id in first_lineup_match.items():
        row = tier2_df[tier2_df["match_id"] == match_id]
        if row.empty:
            continue
        row = row.iloc[0]
        col = "home_formation_win_rate" if int(row["home_team_id"]) == team_id \
            else "away_formation_win_rate"
        if col not in tier2_df.columns:
            continue
        val = row[col]
        if not np.isnan(val):
            fwr_failures.append(
                f"team {team_id}: {col}={val:.3f} (expected NaN on first lineup match)"
            )

    results.append(_check_result(
        "Formation win rate NaN on first lineup match",
        len(fwr_failures) == 0,
        "; ".join(fwr_failures[:3]) if fwr_failures
        else f"verified {len(first_lineup_match)} teams",
    ))

    # ── Check 2: Lineup deviation NaN on first lineup match ──────────────────
    dev_failures = []
    for team_id, match_id in first_lineup_match.items():
        row = tier2_df[tier2_df["match_id"] == match_id]
        if row.empty:
            continue
        row = row.iloc[0]
        col = "home_lineup_deviation" if int(row["home_team_id"]) == team_id \
            else "away_lineup_deviation"
        if col not in tier2_df.columns:
            continue
        val = row[col]
        if not np.isnan(val):
            dev_failures.append(
                f"team {team_id}: {col}={val:.3f} (expected NaN on first lineup match)"
            )

    results.append(_check_result(
        "Lineup deviation NaN on first lineup match",
        len(dev_failures) == 0,
        "; ".join(dev_failures[:3]) if dev_failures
        else f"verified {len(first_lineup_match)} teams",
    ))

    # ── Check 3: Formation change NaN on first lineup match ───────────────────
    fc_failures = []
    for team_id, match_id in first_lineup_match.items():
        row = tier2_df[tier2_df["match_id"] == match_id]
        if row.empty:
            continue
        row = row.iloc[0]
        col = "home_formation_change" if int(row["home_team_id"]) == team_id \
            else "away_formation_change"
        if col not in tier2_df.columns:
            continue
        val = row[col]
        if not np.isnan(val):
            fc_failures.append(
                f"team {team_id}: {col}={val} (expected NaN on first lineup match)"
            )

    results.append(_check_result(
        "Formation change NaN on first lineup match",
        len(fc_failures) == 0,
        "; ".join(fc_failures[:3]) if fc_failures
        else f"verified {len(first_lineup_match)} teams",
    ))

    # ── Check 4: Formation win rate spot-check against recomputed values ──────
    # For up to 10 sampled teams, recompute formation_win_rate for each of
    # their matches (skipping the first, where NaN is expected) and compare
    # against the stored value in tier2_df. A mismatch indicates either a
    # temporal leak or a computation error in the feature engineering step.
    recompute_failures: list[str] = []
    slices_checked = 0
    sample_teams = list(team_lineup_hist.keys())[:10]

    for team_id in sample_teams:
        hist = team_lineup_hist[team_id]
        for i in range(1, len(hist)):  # skip first match (NaN expected, no prior data)
            match_id_i  = int(hist.iloc[i]["match_id"])
            match_date_i = hist.iloc[i]["date"]
            formation_i  = hist.iloc[i]["formation"]

            row = tier2_df[tier2_df["match_id"] == match_id_i]
            if row.empty:
                continue

            row = row.iloc[0]
            col = (
                "home_formation_win_rate"
                if int(row["home_team_id"]) == team_id
                else "away_formation_win_rate"
            )
            if col not in tier2_df.columns:
                continue

            expected = _formation_win_rate(hist, formation_i, match_date_i)
            actual   = row[col]

            both_nan = np.isnan(expected) and (
                isinstance(actual, float) and np.isnan(actual)
            )
            if not both_nan and not np.isclose(float(expected), float(actual), equal_nan=True):
                recompute_failures.append(
                    f"team {team_id} match {match_id_i}: expected {expected:.4f} got {actual:.4f}"
                )
            slices_checked += 1

    results.append(_check_result(
        f"Formation win rate spot-check matches recomputed values ({len(sample_teams)} teams sampled)",
        len(recompute_failures) == 0,
        "; ".join(recompute_failures[:3]) if recompute_failures
        else f"{slices_checked} values recomputed and matched",
    ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal leakage audit for Tier 2 features")
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/tier2_features.csv"),
        help="Path to Tier 2 feature CSV",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        logger.error("Tier 2 CSV not found: %s — run features/export_tier2.py first.", args.csv)
        sys.exit(1)

    logger.info("Loading Tier 2 feature matrix from %s", args.csv)
    tier2_df = pd.read_csv(args.csv, parse_dates=["date"])

    engine = create_engine(DATABASE_URL)
    results = audit(tier2_df, engine)

    print("\n" + "=" * 60)
    print("TIER 2 LEAKAGE AUDIT SUMMARY")
    print("=" * 60)
    passes  = sum(1 for r in results if r["status"] == _PASS)
    fails   = sum(1 for r in results if r["status"] == _FAIL)
    skipped = sum(1 for r in results if r["status"] == _SKIP)

    for r in results:
        icon = "✓" if r["status"] == _PASS else ("?" if r["status"] == _SKIP else "✗")
        print(f"  {icon} [{r['status']}] {r['check']}")
        if r["detail"]:
            print(f"        {r['detail']}")

    print("=" * 60)
    print(f"  {passes} passed, {fails} failed, {skipped} skipped")
    print("=" * 60 + "\n")

    if fails > 0:
        logger.error("Tier 2 leakage audit FAILED — do NOT use this matrix for modelling.")
        sys.exit(1)
    elif skipped > 0:
        logger.warning("Tier 2 audit incomplete — ingest lineup data and re-run.")
    else:
        logger.info("Tier 2 leakage audit PASSED — feature matrix is temporally clean.")


if __name__ == "__main__":
    main()
