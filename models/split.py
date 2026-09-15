"""
Temporal train/val/test split constants for the Edge project.

Split is fixed by season — never by random shuffle. This is mandatory
for any sports prediction model: using future data to train on past
matches would invalidate every evaluation result.

    Train  : season 2022  (380 matches)
    Val    : season 2023  (380 matches) — used for hyperparam tuning
    Test   : season 2024  (380 matches) — LOCKED until final evaluation

The test set must never be touched during any training, feature selection,
or hyperparameter search step. Only the final, fixed model may predict on it.
"""

from __future__ import annotations

import pandas as pd

# ── Season assignments ─────────────────────────────────────────────────────────
TRAIN_SEASON = 2022
VAL_SEASON   = 2023
TEST_SEASON  = 2024

# ── Column roles ──────────────────────────────────────────────────────────────
# Metadata: identifiers used for joining / logging — NOT model inputs
METADATA_COLS = [
    "match_id",
    "date",
    "season",
    "matchday",
    "home_team_id",
    "away_team_id",
]

# Feature columns fed to the model
FEATURE_COLS = [
    # ELO
    "home_elo",
    "away_elo",
    "elo_diff",
    # Rolling form — last 5
    "home_form5_win_rate",
    "home_form5_draw_rate",
    "home_form5_loss_rate",
    "away_form5_win_rate",
    "away_form5_draw_rate",
    "away_form5_loss_rate",
    # Rolling form — last 10
    "home_form10_win_rate",
    "home_form10_draw_rate",
    "home_form10_loss_rate",
    "away_form10_win_rate",
    "away_form10_draw_rate",
    "away_form10_loss_rate",
    # xG rolling averages (may be 100% null until team_stats ingestion completes)
    "home_xg5_for_avg",
    "home_xg5_against_avg",
    "away_xg5_for_avg",
    "away_xg5_against_avg",
    # Venue splits
    "home_home_win_rate",
    "away_away_win_rate",
    # Rest days
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    # League table
    "home_points",
    "home_position",
    "home_goal_diff",
    "away_points",
    "away_position",
    "away_goal_diff",
    "points_diff",
    # Points per game (last 5)
    "home_ppg_last5",
    "away_ppg_last5",
    "ppg_diff",
    # Head-to-head (last 5 meetings)
    "h2h_home_win_rate",
    "h2h_draw_rate",
    "h2h_away_win_rate",
    "h2h_total",
]

# Target columns
TARGET_COL       = "result_label"   # 1=H, 0=D, -1=A  (for model training)
TARGET_STR_COL   = "result"         # 'H'/'D'/'A'      (for human readability)

# Canonical label order used in probability arrays throughout this project:
#   index 0 → 'H'  (prob_home)
#   index 1 → 'D'  (prob_draw)
#   index 2 → 'A'  (prob_away)
LABEL_ORDER = ["H", "D", "A"]
LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_ORDER)}

# Maps result_label integers (1=H, 0=D, -1=A) to class index (0/1/2)
INT_LABEL_TO_IDX = {1: 0, 0: 1, -1: 2}


def load_splits(
    csv_path: str = "data/processed/tier1_features.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the feature matrix and return (train, val, test) DataFrames.

    Each returned DataFrame contains metadata + feature + target columns.
    No random shuffling is ever applied.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], utc=True)

    train = df[df["season"] == TRAIN_SEASON].copy().reset_index(drop=True)
    val   = df[df["season"] == VAL_SEASON].copy().reset_index(drop=True)
    test  = df[df["season"] == TEST_SEASON].copy().reset_index(drop=True)

    return train, val, test


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) — feature matrix and integer label series."""
    # Only include feature columns that actually exist in this DataFrame
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols]
    y = df[TARGET_COL]
    return X, y
