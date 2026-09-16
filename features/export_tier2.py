"""
Export the Tier 2 feature matrix to a CSV file for model training.

Loads the existing Tier 1 features CSV, appends Tier 2 post-lineup features
from the database, runs the Tier 2 leakage audit, and writes the combined
result to data/processed/tier2_features.csv.

Prerequisite: lineup data must be ingested first:
    python pipeline/ingest.py --step lineups
    python pipeline/ingest.py --step player_stats   # for strength + fatigue features

Usage:
    python features/export_tier2.py
    python features/export_tier2.py --output data/processed/tier2_features.csv
    python features/export_tier2.py --skip-audit
    python features/export_tier2.py --dry-run
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
from features.tier2 import TIER2_FEATURE_COLS, compute_tier2_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_TIER1_CSV = pathlib.Path("data/processed/tier1_features.csv")
_DEFAULT_OUTPUT    = pathlib.Path("data/processed/tier2_features.csv")

_METADATA_COLS = [
    "match_id", "date", "season", "matchday", "home_team_id", "away_team_id",
]
_TIER1_FEATURE_COLS = [
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
    "away_points", "away_position", "away_goal_diff", "points_diff",
    "home_ppg_last5", "away_ppg_last5", "ppg_diff",
    "h2h_home_win_rate", "h2h_draw_rate", "h2h_away_win_rate", "h2h_total",
]
# Categorical formation strings — written to CSV as metadata, not used as
# direct model inputs (would need one-hot encoding or embedding first)
_FORMATION_COLS = ["home_formation", "away_formation"]
_TARGET_COLS    = ["result", "result_label"]


def _print_summary(df: pd.DataFrame) -> None:
    tier1_present = [c for c in _TIER1_FEATURE_COLS if c in df.columns]
    tier2_present = [c for c in TIER2_FEATURE_COLS if c in df.columns]

    tier2_null = {
        c: f"{df[c].isna().mean() * 100:.1f}%"
        for c in tier2_present
        if df[c].isna().mean() > 0
    }

    print("\n" + "=" * 65)
    print("TIER 2 FEATURE MATRIX SUMMARY")
    print("=" * 65)
    print(f"  Rows                : {len(df):,}")
    print(f"  Tier 1 features     : {len(tier1_present)}")
    print(f"  Tier 2 features     : {len(tier2_present)}")
    print(f"  Seasons             : {sorted(df['season'].unique())}")
    print(f"  Date range          : {df['date'].min()} → {df['date'].max()}")

    print("\n  Target distribution:")
    for label, count in df["result"].value_counts().items():
        print(f"    {label}: {count:,} ({count / len(df) * 100:.1f}%)")

    if tier2_null:
        print("\n  Tier 2 null rates (non-zero only):")
        for col, rate in sorted(tier2_null.items()):
            print(f"    {col}: {rate} null")
    else:
        print("\n  Tier 2 null rates   : all 0% (full coverage)")

    if "home_formation" in df.columns:
        covered = df["home_formation"].notna().sum()
        print(
            f"\n  Lineup coverage     : {covered}/{len(df)} matches "
            f"({covered / len(df) * 100:.1f}%)"
        )

    print("\n  Split summary (temporal):")
    for season, group in df.groupby("season"):
        print(f"    Season {season}: {len(group):,} matches")

    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tier 2 feature matrix to CSV")
    parser.add_argument(
        "--tier1-csv",
        type=pathlib.Path,
        default=_DEFAULT_TIER1_CSV,
        help=f"Path to Tier 1 features CSV (default: {_DEFAULT_TIER1_CSV})",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip the Tier 2 leakage audit (not recommended)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features and print summary without writing the output file",
    )
    args = parser.parse_args()

    if not args.tier1_csv.exists():
        logger.error(
            "Tier 1 CSV not found at %s — run features/export.py first.",
            args.tier1_csv,
        )
        sys.exit(1)

    logger.info("Loading Tier 1 features from %s...", args.tier1_csv)
    tier1_df = pd.read_csv(args.tier1_csv, parse_dates=["date"])
    tier1_df["date"] = pd.to_datetime(tier1_df["date"], utc=True)
    logger.info("Loaded %d Tier 1 rows × %d columns.", len(tier1_df), len(tier1_df.columns))

    engine = create_engine(DATABASE_URL)

    logger.info("Computing Tier 2 features...")
    tier2_df = compute_tier2_features(engine, tier1_df)

    _print_summary(tier2_df)

    # ── Leakage audit ─────────────────────────────────────────────────────────
    if not args.skip_audit:
        logger.info("Running Tier 2 leakage audit...")
        from features.leakage_audit_tier2 import _FAIL, _SKIP, audit
        audit_results = audit(tier2_df, engine)

        failed  = [r for r in audit_results if r["status"] == _FAIL]
        skipped = [r for r in audit_results if r["status"] == _SKIP]

        if failed:
            logger.error(
                "Tier 2 leakage audit FAILED (%d checks failed). "
                "Do NOT export this matrix. Fix the feature engineering first.",
                len(failed),
            )
            for r in failed:
                logger.error("  FAIL: %s — %s", r["check"], r["detail"])
            sys.exit(1)
        elif skipped:
            logger.warning(
                "Tier 2 audit skipped (%d checks) — lineup data not yet ingested. "
                "Re-run after:  python pipeline/ingest.py --step lineups",
                len(skipped),
            )
        else:
            logger.info("Tier 2 leakage audit passed (%d checks).", len(audit_results))
    else:
        logger.warning("Tier 2 leakage audit skipped — use with caution.")

    if args.dry_run:
        logger.info("Dry run — not writing output file.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Column order: metadata → Tier 1 features → formation strings → Tier 2 features → targets
    col_order = (
        _METADATA_COLS
        + [c for c in _TIER1_FEATURE_COLS if c in tier2_df.columns]
        + [c for c in _FORMATION_COLS if c in tier2_df.columns]
        + [c for c in TIER2_FEATURE_COLS if c in tier2_df.columns]
        + _TARGET_COLS
    )
    col_order = [c for c in col_order if c in tier2_df.columns]

    tier2_df[col_order].to_csv(args.output, index=False)
    logger.info(
        "Wrote %d rows × %d columns to %s",
        len(tier2_df), len(col_order), args.output,
    )
    print(f"Tier 2 feature matrix saved to: {args.output}")


if __name__ == "__main__":
    main()
