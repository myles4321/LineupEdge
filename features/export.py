"""
Export the Tier 1 feature matrix to a CSV file for model training.

Computes all pre-lineup features from the database, runs the leakage audit,
and writes the result to data/processed/tier1_features.csv.

Usage:
    python features/export.py
    python features/export.py --output data/processed/tier1_features.csv
    python features/export.py --skip-audit   # skip leakage check (not recommended)
    python features/export.py --dry-run      # compute features but don't write file
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL
from features.tier1 import compute_tier1_features
from features import leakage_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT = pathlib.Path(__file__).parent.parent / "data" / "processed" / "tier1_features.csv"

# Feature columns (excludes metadata and target)
_FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff",
    "home_form5_win_rate", "home_form5_draw_rate", "home_form5_loss_rate",
    "away_form5_win_rate", "away_form5_draw_rate", "away_form5_loss_rate",
    "home_form10_win_rate", "home_form10_draw_rate", "home_form10_loss_rate",
    "away_form10_win_rate", "away_form10_draw_rate", "away_form10_loss_rate",
    "home_xg5_for_avg", "home_xg5_against_avg",
    "away_xg5_for_avg", "away_xg5_against_avg",
    "home_home_win_rate", "away_away_win_rate",
    "home_rest_days", "away_rest_days", "rest_days_diff",
    "home_points", "home_position", "home_goal_diff",
    "away_points", "away_position", "away_goal_diff",
    "points_diff",
    "home_ppg_last5", "away_ppg_last5", "ppg_diff",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate", "h2h_total",
]

_METADATA_COLS = [
    "match_id", "date", "season", "matchday", "home_team_id", "away_team_id",
]

_TARGET_COLS = ["result", "result_label"]


def _print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the feature matrix."""
    print("\n" + "=" * 60)
    print("TIER 1 FEATURE MATRIX SUMMARY")
    print("=" * 60)
    print(f"  Rows     : {len(df):,}")
    print(f"  Features : {len(_FEATURE_COLS)}")
    print(f"  Seasons  : {sorted(df['season'].unique())}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    print("\n  Target distribution:")
    dist = df["result"].value_counts()
    total = len(df)
    for label, count in dist.items():
        print(f"    {label}: {count:,} ({count/total*100:.1f}%)")

    print("\n  Missing value rates (feature columns):")
    for col in _FEATURE_COLS:
        if col not in df.columns:
            continue
        null_rate = df[col].isna().mean()
        if null_rate > 0:
            print(f"    {col}: {null_rate*100:.1f}% null")

    print("\n  Split summary (temporal):")
    for season, group in df.groupby("season"):
        print(f"    Season {season}: {len(group):,} matches")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tier 1 feature matrix to CSV")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the temporal leakage audit (not recommended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features and print summary without writing the file",
    )
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    logger.info("Computing Tier 1 features...")
    df = compute_tier1_features(engine)

    if df.empty:
        logger.error("No features computed — is the database populated? Run the ingestion pipeline first.")
        sys.exit(1)

    _print_summary(df)

    # ── Leakage audit ──────────────────────────────────────────────────────────
    if not args.skip_audit:
        logger.info("Running temporal leakage audit...")
        from features.tier1 import _load_matches
        matches = _load_matches(engine)
        audit_results = leakage_audit.audit(df, matches)

        failed = [r for r in audit_results if r["status"] == leakage_audit._FAIL]
        if failed:
            logger.error(
                "Leakage audit FAILED (%d checks failed). "
                "Do NOT export this matrix. Fix the feature engineering first.",
                len(failed),
            )
            for r in failed:
                logger.error("  FAIL: %s — %s", r["check"], r["detail"])
            sys.exit(1)
        else:
            logger.info("Leakage audit passed (%d checks).", len(audit_results))
    else:
        logger.warning("Leakage audit skipped — use with caution.")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry run — not writing file.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Write all columns (metadata + features + targets)
    col_order = _METADATA_COLS + _FEATURE_COLS + _TARGET_COLS
    # Only include columns that actually exist in the DataFrame
    col_order = [c for c in col_order if c in df.columns]
    df[col_order].to_csv(args.output, index=False)

    logger.info("Wrote %d rows × %d columns to %s", len(df), len(col_order), args.output)
    print(f"Feature matrix saved to: {args.output}")


if __name__ == "__main__":
    main()
