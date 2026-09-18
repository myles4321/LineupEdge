"""
Phase 5 — Tier 1 vs Tier 2 Core Comparison Experiment.

Reads logged predictions from the predictions table and computes the full
comparison between the pre-lineup (Tier 1) and post-lineup (Tier 2) models:

  - Log loss delta and Brier score delta
  - Per-class probability improvement (H / D / A)
  - Bootstrap significance test on the log loss delta (1000 resamples)
  - Flat-stake betting simulation against B365 odds
  - Side-by-side calibration curves (T1 vs T2 vs Bookmaker)
  - Per-class improvement bar chart

Primary comparison is XGBoost T1 vs XGBoost T2 (best model from each tier).
Logistic regression results are included in the summary table for completeness.

Prerequisites (run in order):
  python models/train_tier1.py        — writes tier=1 predictions to DB
  python features/export_tier2.py     — generates tier2_features.csv
  python models/train_tier2.py        — writes tier=2 predictions to DB

Usage:
    python models/compare_tiers.py
    python models/compare_tiers.py --no-plots   # skip plot generation
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL
from models.split import LABEL_ORDER, LABEL_TO_IDX

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPORTS_DIR = pathlib.Path("reports")

# Model versions — must match the tags used in train_tier1.py / train_tier2.py
_T1_XGB_VERSION    = "tier1_xgb_v1"
_T2_XGB_VERSION    = "tier2_xgb_v1"
_T1_LOGREG_VERSION = "tier1_logreg_v1"
_T2_LOGREG_VERSION = "tier2_logreg_v1"

_N_BOOTSTRAP   = 1000
_BOOTSTRAP_SEED = 42
_TEST_SEASON    = 2024


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_predictions(engine, model_version: str) -> pd.DataFrame:
    """Load predictions for one model version from the database.

    Returns columns: match_id, prob_home, prob_draw, prob_away.
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT match_id, prob_home, prob_draw, prob_away
                FROM predictions
                WHERE model_version = :version
                ORDER BY match_id ASC
            """),
            conn,
            params={"version": model_version},
        )
    return df


def _load_test_results(engine) -> pd.DataFrame:
    """Load 2024 test season match results.

    Returns columns: match_id, result (H/D/A).
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT match_id, result
                FROM matches
                WHERE season = :season
                  AND status = 'FT'
                  AND result IS NOT NULL
                ORDER BY match_id ASC
            """),
            conn,
            params={"season": _TEST_SEASON},
        )
    return df


def _load_b365_odds(engine, match_ids: list[int]) -> pd.DataFrame:
    """Load B365 odds for the given match IDs.

    Returns columns: match_id, home_odds, draw_odds, away_odds,
                     implied_home, implied_draw, implied_away.
    implied_* columns are pre-normalised probabilities (vig removed).
    home/draw/away_odds are decimal payout odds for the betting simulation.
    """
    if not match_ids:
        return pd.DataFrame()

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT match_id,
                       home_odds, draw_odds, away_odds,
                       implied_home, implied_draw, implied_away
                FROM odds
                WHERE match_id IN :ids AND bookmaker = 'B365'
            """),
            conn,
            params={"ids": tuple(match_ids)},
        )
    return df


# ── Metric helpers ────────────────────────────────────────────────────────────

def _multiclass_brier(y_true_str: pd.Series, proba: np.ndarray) -> float:
    """Mean one-vs-rest Brier score across all three outcome classes."""
    scores = [
        brier_score_loss((y_true_str == label).astype(float), proba[:, i])
        for i, label in enumerate(LABEL_ORDER)
    ]
    return float(np.mean(scores))


def _per_class_log_loss(y_true_str: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """One-vs-rest log loss per outcome class."""
    results = {}
    for i, label in enumerate(LABEL_ORDER):
        y_binary = (y_true_str == label).astype(int)
        class_proba = np.column_stack([1 - proba[:, i], proba[:, i]])
        results[label] = float(log_loss(y_binary, class_proba))
    return results


def _compute_all_metrics(
    y_true_str: pd.Series,
    proba: np.ndarray,
    label: str,
) -> dict:
    """Compute log loss, Brier score, and per-class log loss for one model."""
    y_enc = y_true_str.map(LABEL_TO_IDX).values
    return {
        "model":        label,
        "log_loss":     float(log_loss(y_enc, proba, labels=[0, 1, 2])),
        "brier":        _multiclass_brier(y_true_str, proba),
        "per_class_ll": _per_class_log_loss(y_true_str, proba),
    }


# ── Bootstrap significance test ───────────────────────────────────────────────

def _bootstrap_significance(
    y_true_idx: np.ndarray,
    proba_t1: np.ndarray,
    proba_t2: np.ndarray,
    n_bootstrap: int = _N_BOOTSTRAP,
    random_seed: int = _BOOTSTRAP_SEED,
) -> dict:
    """Bootstrap significance test on the log loss improvement T1 → T2.

    For each of n_bootstrap resamples (with replacement, size N):
      delta = log_loss(T1) − log_loss(T2)   (positive = T2 improved)

    p-value = fraction of bootstrap deltas ≤ 0 (no improvement).
    A p-value < 0.05 indicates T2 significantly outperforms T1 at α=5%.
    The 95% CI is the 2.5th–97.5th percentile interval of bootstrap deltas.
    """
    rng = np.random.default_rng(random_seed)
    n   = len(y_true_idx)

    bootstrap_deltas = np.empty(n_bootstrap)
    for k in range(n_bootstrap):
        idx   = rng.integers(0, n, size=n)
        ll_t1 = log_loss(y_true_idx[idx], proba_t1[idx], labels=[0, 1, 2])
        ll_t2 = log_loss(y_true_idx[idx], proba_t2[idx], labels=[0, 1, 2])
        bootstrap_deltas[k] = ll_t1 - ll_t2  # positive = T2 improved

    observed_delta = float(
        log_loss(y_true_idx, proba_t1, labels=[0, 1, 2])
        - log_loss(y_true_idx, proba_t2, labels=[0, 1, 2])
    )
    p_value  = float((bootstrap_deltas <= 0).mean())
    ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
    ci_upper = float(np.percentile(bootstrap_deltas, 97.5))

    return {
        "observed_delta_log_loss": observed_delta,
        "p_value":                 p_value,
        "ci_95_lower":             ci_lower,
        "ci_95_upper":             ci_upper,
        "n_bootstrap":             n_bootstrap,
        "significant_at_05":       p_value < 0.05,
    }


# ── Betting simulation ────────────────────────────────────────────────────────

def _betting_simulation(
    y_true_str: pd.Series,
    proba: np.ndarray,
    odds_df: pd.DataFrame,
    model_label: str,
) -> dict:
    """Flat-stake (£1) betting simulation against B365 decimal odds.

    Strategy: bet on the outcome with the highest predicted probability.

    Profit per bet:
      Correct: (decimal_odds − 1) × £1
      Wrong:   −£1

    Matches with missing or sub-1.0 odds are skipped.
    odds_df must be row-aligned with y_true_str and proba (same order).
    """
    odds_col = {"H": "home_odds", "D": "draw_odds", "A": "away_odds"}

    total_staked = 0.0
    total_pnl    = 0.0
    n_bets       = 0
    n_wins       = 0

    for i, (true_outcome, row_proba) in enumerate(zip(y_true_str.values, proba)):
        if i >= len(odds_df):
            break
        odds_row = odds_df.iloc[i]

        predicted_outcome = LABEL_ORDER[int(np.argmax(row_proba))]
        raw_odds_val = odds_row[odds_col[predicted_outcome]]

        # Skip rows with NaN or invalid odds
        if pd.isna(raw_odds_val) or float(raw_odds_val) <= 1.0:
            continue

        bet_odds = float(raw_odds_val)
        n_bets       += 1
        total_staked += 1.0

        if predicted_outcome == true_outcome:
            total_pnl += (bet_odds - 1.0)
            n_wins    += 1
        else:
            total_pnl -= 1.0

    roi      = (total_pnl / total_staked * 100) if total_staked > 0 else 0.0
    win_rate = (n_wins / n_bets * 100) if n_bets > 0 else 0.0

    return {
        "model":        model_label,
        "n_bets":       n_bets,
        "n_wins":       n_wins,
        "win_rate_pct": win_rate,
        "total_staked": total_staked,
        "total_pnl":    total_pnl,
        "roi_pct":      roi,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_calibration_comparison(
    y_true_str: pd.Series,
    model_probas: dict[str, np.ndarray],
) -> pathlib.Path:
    """Side-by-side calibration curves: T1 XGBoost vs T2 XGBoost vs Bookmaker."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    plot_path = _REPORTS_DIR / "calibration_comparison.png"

    colours = {
        "Tier 1 XGBoost":  "#2563eb",
        "Tier 2 XGBoost":  "#e85d04",
        "Bookmaker B365":  "#dc2626",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    for ax_idx, (class_label, ax) in enumerate(zip(LABEL_ORDER, axes)):
        y_binary = (y_true_str == class_label).astype(float).values

        for model_name, proba in model_probas.items():
            class_proba = np.array(proba[:, LABEL_TO_IDX[class_label]], dtype=float)
            frac_pos, mean_pred = calibration_curve(
                y_binary, class_proba, n_bins=8, strategy="uniform"
            )
            ax.plot(
                mean_pred, frac_pos, marker="o", linewidth=1.8,
                label=model_name, color=colours.get(model_name, "grey"),
            )

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
        ax.set_title(f"Outcome: {class_label}", fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        if ax_idx == 0:
            ax.set_ylabel("Fraction of positives")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    fig.suptitle(
        "Calibration Comparison — Tier 1 vs Tier 2 XGBoost vs Bookmaker (2024 test)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def _plot_tier_improvement(
    metrics_t1: dict,
    metrics_t2: dict,
    bm_metrics: dict | None,
) -> pathlib.Path:
    """Two-panel chart: overall log loss comparison + per-class improvement."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    plot_path = _REPORTS_DIR / "tier_improvement.png"

    models     = ["Tier 1 XGBoost", "Tier 2 XGBoost"]
    ll_vals    = [metrics_t1["log_loss"], metrics_t2["log_loss"]]
    colours    = ["#2563eb", "#e85d04"]

    if bm_metrics:
        models.append("Bookmaker B365")
        ll_vals.append(bm_metrics["log_loss"])
        colours.append("#dc2626")

    x = np.arange(len(models))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall log loss
    bars = ax1.bar(x, ll_vals, color=colours, alpha=0.85, width=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=12, ha="right")
    ax1.set_title("Log Loss — 2024 Test Season\n(lower = better)", fontsize=11)
    ax1.set_ylabel("Log Loss")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, ll_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # Right: per-class log loss improvement (T1 − T2)
    classes = list(LABEL_ORDER)
    deltas  = [
        metrics_t1["per_class_ll"][c] - metrics_t2["per_class_ll"][c]
        for c in classes
    ]
    bar_colours_delta = ["#16a34a" if d >= 0 else "#dc2626" for d in deltas]
    bars2 = ax2.bar(classes, deltas, color=bar_colours_delta, alpha=0.85, width=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title(
        "Per-Class Log Loss Improvement\n(Tier 1 − Tier 2, green = T2 better)",
        fontsize=11,
    )
    ax2.set_ylabel("Δ Log Loss")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars2, deltas):
        y_off = 0.001 if val >= 0 else -0.004
        ax2.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                 f"{val:+.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Tier 1 vs Tier 2 — Core Experiment Result", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tier 1 vs Tier 2 core comparison experiment"
    )
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    _REPORTS_DIR.mkdir(exist_ok=True)
    engine = create_engine(DATABASE_URL)

    # ── Load predictions ──────────────────────────────────────────────────────
    logger.info("Loading predictions from database...")
    preds_t1_xgb    = _load_predictions(engine, _T1_XGB_VERSION)
    preds_t2_xgb    = _load_predictions(engine, _T2_XGB_VERSION)
    preds_t1_logreg = _load_predictions(engine, _T1_LOGREG_VERSION)
    preds_t2_logreg = _load_predictions(engine, _T2_LOGREG_VERSION)

    if preds_t1_xgb.empty:
        logger.error(
            "No Tier 1 XGBoost predictions found (model_version='%s'). "
            "Run:  python models/train_tier1.py",
            _T1_XGB_VERSION,
        )
        sys.exit(1)

    if preds_t2_xgb.empty:
        logger.error(
            "No Tier 2 XGBoost predictions found (model_version='%s'). "
            "Run:  python models/train_tier2.py  (after python features/export_tier2.py)",
            _T2_XGB_VERSION,
        )
        sys.exit(1)

    logger.info(
        "Loaded: T1 XGB=%d  T2 XGB=%d  T1 LogReg=%d  T2 LogReg=%d rows",
        len(preds_t1_xgb), len(preds_t2_xgb),
        len(preds_t1_logreg), len(preds_t2_logreg),
    )

    # ── Load ground truth ─────────────────────────────────────────────────────
    logger.info("Loading 2024 test season results...")
    test_results = _load_test_results(engine)
    if test_results.empty:
        logger.error("No 2024 test results found in the matches table.")
        sys.exit(1)

    # Align T1 and T2 predictions with ground truth via inner join on match_id
    aligned = (
        test_results
        .merge(
            preds_t1_xgb.rename(columns={
                "prob_home": "t1_home", "prob_draw": "t1_draw", "prob_away": "t1_away",
            }),
            on="match_id", how="inner",
        )
        .merge(
            preds_t2_xgb.rename(columns={
                "prob_home": "t2_home", "prob_draw": "t2_draw", "prob_away": "t2_away",
            }),
            on="match_id", how="inner",
        )
        .sort_values("match_id")
        .reset_index(drop=True)
    )

    if aligned.empty:
        logger.error("No rows shared between test results and predictions — check match_ids.")
        sys.exit(1)

    logger.info(
        "Aligned dataset: %d matches (from %d T1, %d T2, %d test rows)",
        len(aligned), len(preds_t1_xgb), len(preds_t2_xgb), len(test_results),
    )

    y_true_str = aligned["result"]
    y_true_idx = y_true_str.map(LABEL_TO_IDX).values.astype(int)

    proba_t1 = aligned[["t1_home", "t1_draw", "t1_away"]].values.astype(float)
    proba_t2 = aligned[["t2_home", "t2_draw", "t2_away"]].values.astype(float)

    # ── Metrics ───────────────────────────────────────────────────────────────
    logger.info("Computing metrics...")
    metrics_t1 = _compute_all_metrics(y_true_str, proba_t1, "Tier 1 XGBoost")
    metrics_t2 = _compute_all_metrics(y_true_str, proba_t2, "Tier 2 XGBoost")

    # LogReg supplementary metrics (if both tiers are logged)
    logreg_metrics: list[dict] = []
    for lr_label, lr_preds in [
        ("Tier 1 LogReg", preds_t1_logreg),
        ("Tier 2 LogReg", preds_t2_logreg),
    ]:
        if lr_preds.empty:
            continue
        lr_aligned = (
            test_results
            .merge(
                lr_preds.rename(columns={
                    "prob_home": "lr_home", "prob_draw": "lr_draw", "prob_away": "lr_away",
                }),
                on="match_id", how="inner",
            )
            .sort_values("match_id")
            .reset_index(drop=True)
        )
        if not lr_aligned.empty:
            p = lr_aligned[["lr_home", "lr_draw", "lr_away"]].values.astype(float)
            logreg_metrics.append(_compute_all_metrics(lr_aligned["result"], p, lr_label))

    # Bookmaker benchmark — use pre-normalised implied probabilities from odds table
    bm_odds    = _load_b365_odds(engine, aligned["match_id"].tolist())
    bm_metrics: dict | None = None
    bm_aligned_df: pd.DataFrame | None = None

    if not bm_odds.empty:
        bm_joined  = aligned.merge(bm_odds, on="match_id", how="left")
        bm_mask    = bm_joined["implied_home"].notna()
        bm_rows    = bm_joined[bm_mask].reset_index(drop=True)

        if len(bm_rows) > 0:
            bm_proba = bm_rows[["implied_home", "implied_draw", "implied_away"]].values.astype(float)
            bm_metrics = _compute_all_metrics(bm_rows["result"], bm_proba, "Bookmaker B365")
            bm_aligned_df = bm_rows  # used for calibration plot and betting sim
            logger.info(
                "Bookmaker B365 odds available for %d/%d aligned matches.",
                len(bm_rows), len(aligned),
            )

    # ── Bootstrap significance test ───────────────────────────────────────────
    logger.info("Running bootstrap significance test (%d resamples)...", _N_BOOTSTRAP)
    bootstrap = _bootstrap_significance(y_true_idx, proba_t1, proba_t2)

    # ── Betting simulation ────────────────────────────────────────────────────
    bet_results: list[dict] = []
    if bm_aligned_df is not None:
        logger.info("Running betting simulation...")
        # Align odds to full aligned dataset (not just bm_rows) so row indices match
        bm_full = aligned.merge(bm_odds, on="match_id", how="left")
        odds_cols = bm_full[["home_odds", "draw_odds", "away_odds"]].copy()

        bet_results.append(_betting_simulation(y_true_str, proba_t1, odds_cols, "Tier 1 XGBoost"))
        bet_results.append(_betting_simulation(y_true_str, proba_t2, odds_cols, "Tier 2 XGBoost"))
    else:
        logger.warning("No B365 odds available — betting simulation skipped.")

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TIER 1 vs TIER 2 COMPARISON — 2024 Test Season")
    print("=" * 70)

    all_display_metrics = [metrics_t1, metrics_t2] + logreg_metrics
    if bm_metrics:
        all_display_metrics.append(bm_metrics)

    print(f"\n{'Model':<28}  {'Log Loss':>9}  {'Brier':>8}  {'Δ Log Loss vs T1':>17}")
    print("─" * 70)
    for m in all_display_metrics:
        delta_str = ""
        if m["model"] != "Tier 1 XGBoost":
            delta = metrics_t1["log_loss"] - m["log_loss"]
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "")
            delta_str = f"{delta:+.4f} {arrow}"
        print(f"  {m['model']:<26}  {m['log_loss']:>9.4f}  {m['brier']:>8.4f}  {delta_str:>17}")

    print(f"\n{'─' * 70}")
    print("Per-class log loss  (lower = better):")
    print(f"  {'Model':<26}  {'H (home win)':>14}  {'D (draw)':>10}  {'A (away win)':>13}")
    for m in [metrics_t1, metrics_t2]:
        pc = m["per_class_ll"]
        print(f"  {m['model']:<26}  {pc['H']:>14.4f}  {pc['D']:>10.4f}  {pc['A']:>13.4f}")

    print(f"\n{'─' * 70}")
    print("Per-class improvement  (Tier 1 − Tier 2, positive = T2 improved):")
    for c, full_name in zip(LABEL_ORDER, ["H (home win)", "D (draw)", "A (away win)"]):
        delta = metrics_t1["per_class_ll"][c] - metrics_t2["per_class_ll"][c]
        direction = "T2 better ✓" if delta > 0 else "T1 better ✗"
        print(f"  {full_name:<18}: {delta:+.4f}  ({direction})")

    print(f"\n{'─' * 70}")
    print(f"Bootstrap Significance Test  (n={_N_BOOTSTRAP} resamples, seed={_BOOTSTRAP_SEED}):")
    print(f"  Observed Δ log_loss (T1−T2): {bootstrap['observed_delta_log_loss']:+.4f}")
    print(f"  p-value                     : {bootstrap['p_value']:.4f}")
    print(f"  95% CI on delta             : [{bootstrap['ci_95_lower']:+.4f}, {bootstrap['ci_95_upper']:+.4f}]")
    sig = "YES — T2 significantly outperforms T1 (p < 0.05)" if bootstrap["significant_at_05"] \
        else "NO — improvement not statistically significant (p ≥ 0.05)"
    print(f"  Significant at α=5%?        : {sig}")

    if bet_results:
        print(f"\n{'─' * 70}")
        print("Flat-stake Betting Simulation  (£1/bet, B365 decimal odds):")
        print(f"  {'Model':<28}  {'Bets':>5}  {'Wins':>5}  {'Win%':>6}  {'P&L':>9}  {'ROI':>7}")
        for b in bet_results:
            print(
                f"  {b['model']:<28}  {b['n_bets']:>5}  {b['n_wins']:>5}  "
                f"{b['win_rate_pct']:>5.1f}%  "
                f"£{b['total_pnl']:>+8.2f}  {b['roi_pct']:>+6.1f}%"
            )

    print("\n" + "=" * 70 + "\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        logger.info("Generating comparison plots...")

        if bm_aligned_df is not None:
            bm_mask_arr = aligned["match_id"].isin(bm_aligned_df["match_id"]).values
            bm_p = bm_aligned_df[["implied_home", "implied_draw", "implied_away"]].values.astype(float)
            cal_probas = {
                "Tier 1 XGBoost":  proba_t1[bm_mask_arr],
                "Tier 2 XGBoost":  proba_t2[bm_mask_arr],
                "Bookmaker B365":   bm_p,
            }
            cal_path = _plot_calibration_comparison(
                bm_aligned_df["result"].reset_index(drop=True), cal_probas
            )
        else:
            cal_path = _plot_calibration_comparison(
                y_true_str, {"Tier 1 XGBoost": proba_t1, "Tier 2 XGBoost": proba_t2}
            )

        imp_path = _plot_tier_improvement(metrics_t1, metrics_t2, bm_metrics)

        logger.info("Calibration comparison: %s", cal_path)
        logger.info("Tier improvement chart: %s", imp_path)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_path = _REPORTS_DIR / "tier_comparison.json"
    results_path.write_text(json.dumps({
        "test_season":       _TEST_SEASON,
        "n_matches":         len(aligned),
        "tier1_xgb":         {k: v for k, v in metrics_t1.items() if k != "per_class_ll"},
        "tier2_xgb":         {k: v for k, v in metrics_t2.items() if k != "per_class_ll"},
        "per_class_t1":      metrics_t1["per_class_ll"],
        "per_class_t2":      metrics_t2["per_class_ll"],
        "per_class_delta":   {
            c: metrics_t1["per_class_ll"][c] - metrics_t2["per_class_ll"][c]
            for c in LABEL_ORDER
        },
        "bootstrap":          bootstrap,
        "betting_simulation": bet_results,
        "bookmaker":          {
            k: v for k, v in bm_metrics.items() if k != "per_class_ll"
        } if bm_metrics else None,
    }, indent=2))
    logger.info("Comparison results saved to: %s", results_path)
    logger.info("Phase 5 comparison experiment complete.")


if __name__ == "__main__":
    main()
