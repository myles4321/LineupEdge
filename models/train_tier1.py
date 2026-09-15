"""
Phase 3 — Tier 1 Baseline Model Training & Evaluation.

Trains two models on the pre-lineup feature set:
  1. Multinomial logistic regression (calibrated baseline)
  2. XGBoost multiclass (gradient boosted trees)

Both are evaluated on the 2024 test season using:
  - Log loss  (primary metric — measures probabilistic sharpness)
  - Brier score (secondary — measures calibration quality)
  - Per-class precision / recall
  - Calibration curves (reliability diagrams)

Results are benchmarked against bookmaker B365 implied probabilities.
All test-set predictions are logged to the `predictions` database table.

Usage:
    python models/train_tier1.py
    python models/train_tier1.py --skip-analysis   # skip feature analysis step
    python models/train_tier1.py --no-db           # don't write to DB
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    classification_report,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
import xgboost as xgb

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL
from models.split import (
    FEATURE_COLS,
    INT_LABEL_TO_IDX,
    LABEL_ORDER,
    LABEL_TO_IDX,
    TARGET_COL,
    TARGET_STR_COL,
    VAL_SEASON,
    load_splits,
    xy,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPORTS_DIR  = pathlib.Path("reports")
_MODELS_DIR   = pathlib.Path("models/artifacts")

# ── Model version tags (written to predictions table) ─────────────────────────
_LOGREG_VERSION = "tier1_logreg_v1"
_XGB_VERSION    = "tier1_xgb_v1"


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _multiclass_brier(y_true_str: pd.Series, proba: np.ndarray) -> float:
    """Mean one-vs-rest Brier score across all three outcome classes."""
    scores = []
    for i, label in enumerate(LABEL_ORDER):
        y_binary = (y_true_str == label).astype(float)
        scores.append(brier_score_loss(y_binary, proba[:, i]))
    return float(np.mean(scores))


def _load_bookmaker_benchmark(
    match_ids: pd.Series,
    engine,
) -> pd.DataFrame | None:
    """Load B365 implied probabilities for the given match IDs.

    Returns a DataFrame with columns: match_id, bm_home, bm_draw, bm_away.
    Returns None if odds are not available.
    """
    ids = tuple(int(m) for m in match_ids)
    if not ids:
        return None

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT match_id, implied_home, implied_draw, implied_away
                FROM odds
                WHERE match_id IN :ids AND bookmaker = 'B365'
            """),
            {"ids": ids},
        ).fetchall()

    if not rows:
        return None

    return pd.DataFrame(rows, columns=["match_id", "bm_home", "bm_draw", "bm_away"])


def _log_loss_from_bm(y_true_str: pd.Series, bm: pd.DataFrame) -> float:
    """Compute log loss from bookmaker implied probabilities."""
    # Align bm with y_true by match_id order
    proba = bm[["bm_home", "bm_draw", "bm_away"]].values
    y_enc = y_true_str.map(LABEL_TO_IDX).values
    return log_loss(y_enc, proba, labels=[0, 1, 2])


def _brier_from_bm(y_true_str: pd.Series, bm: pd.DataFrame) -> float:
    """Compute multiclass Brier from bookmaker probabilities."""
    proba = bm[["bm_home", "bm_draw", "bm_away"]].values
    scores = []
    for i, label in enumerate(LABEL_ORDER):
        y_binary = (y_true_str == label).astype(float)
        scores.append(brier_score_loss(y_binary, proba[:, i]))
    return float(np.mean(scores))


def _print_eval(name: str, y_true_str: pd.Series, proba: np.ndarray) -> dict:
    """Print and return evaluation metrics for one model."""
    y_enc = y_true_str.map(LABEL_TO_IDX).values
    ll    = log_loss(y_enc, proba, labels=[0, 1, 2])
    bs    = _multiclass_brier(y_true_str, proba)
    y_pred_idx = np.argmax(proba, axis=1)
    y_pred_str = [LABEL_ORDER[i] for i in y_pred_idx]

    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Log loss  : {ll:.4f}")
    print(f"  Brier     : {bs:.4f}")
    print(f"\n  Classification report (hard predictions — informational only):")
    print(classification_report(y_true_str, y_pred_str, target_names=["H", "D", "A"], zero_division=0))

    return {"model": name, "log_loss": ll, "brier": bs}


# ── Training functions ────────────────────────────────────────────────────────

def _tune_logreg_C(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Grid search C on validation set. Returns best C value."""
    y_val_enc = y_val.map(INT_LABEL_TO_IDX).values
    best_c, best_loss = 1.0, float("inf")

    for c in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("clf",    LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=2000, C=c, random_state=42,
            )),
        ])
        pipeline.fit(X_train, y_train.map(INT_LABEL_TO_IDX))
        proba = pipeline.predict_proba(X_val)
        ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
        logger.info("  LogReg C=%-6s  val log_loss=%.4f", c, ll)
        if ll < best_loss:
            best_loss, best_c = ll, c

    logger.info("Best C=%.4f  val log_loss=%.4f", best_c, best_loss)
    return best_c


def train_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Pipeline:
    """Train multinomial logistic regression with C tuned on validation set."""
    logger.info("Tuning logistic regression C on validation set...")
    best_c = _tune_logreg_C(X_train, y_train, X_val, y_val)

    logger.info("Training final logistic regression on train+val with C=%.4f...", best_c)
    # Final model trained on full train+val to maximise data for test evaluation
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = pd.concat([y_train, y_val], ignore_index=True)

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=2000, C=best_c, random_state=42,
        )),
    ])
    pipeline.fit(X_full, y_full.map(INT_LABEL_TO_IDX))
    return pipeline


def _tune_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Grid search key XGBoost hyperparameters on validation log loss."""
    y_train_enc = y_train.map(INT_LABEL_TO_IDX).values
    y_val_enc   = y_val.map(INT_LABEL_TO_IDX).values

    param_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 4, "learning_rate": 0.1,  "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6},
        {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7},
    ]

    best_params, best_loss = param_grid[0], float("inf")

    for params in param_grid:
        model = xgb.XGBClassifier(
            n_estimators=500,
            num_class=3,
            objective="multi:softprob",
            eval_metric="mlogloss",
            early_stopping_rounds=30,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(
            X_train, y_train_enc,
            eval_set=[(X_val, y_val_enc)],
            verbose=False,
        )
        proba = model.predict_proba(X_val)
        ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
        logger.info(
            "  XGB depth=%-2d lr=%.3f sub=%.1f col=%.1f  val log_loss=%.4f  best_iter=%d",
            params["max_depth"], params["learning_rate"],
            params["subsample"], params["colsample_bytree"],
            ll, model.best_iteration,
        )
        if ll < best_loss:
            best_loss = ll
            best_params = params

    logger.info("Best XGB params: %s  val log_loss=%.4f", best_params, best_loss)
    return best_params


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBClassifier:
    """Train XGBoost with hyperparams tuned on validation set.

    XGBoost handles NaN natively (learns optimal split direction for missing
    values), so no imputation is needed. The final model is trained on
    train+val with the number of trees fixed at the best iteration found
    during tuning — preventing test leakage from early stopping.
    """
    logger.info("Tuning XGBoost on validation set...")
    y_train_enc = y_train.map(INT_LABEL_TO_IDX).values
    y_val_enc   = y_val.map(INT_LABEL_TO_IDX).values

    best_params = _tune_xgb(X_train, y_train, X_val, y_val)

    # Determine best n_estimators using val early stopping
    tuning_model = xgb.XGBClassifier(
        n_estimators=1000,
        num_class=3,
        objective="multi:softprob",
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    tuning_model.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=False,
    )
    best_n_estimators = tuning_model.best_iteration + 1
    logger.info("Best n_estimators=%d — training final model on train+val...", best_n_estimators)

    # Final model: train+val, fixed n_estimators (no early stopping — test set not seen)
    X_full = pd.concat([X_train, X_val], ignore_index=True)
    y_full = np.concatenate([y_train_enc, y_val_enc])

    final_model = xgb.XGBClassifier(
        n_estimators=best_n_estimators,
        num_class=3,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X_full, y_full, verbose=False)
    return final_model


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_calibration(
    y_true_str: pd.Series,
    predictions: dict[str, np.ndarray],
    bm: pd.DataFrame | None,
) -> pathlib.Path:
    """Reliability diagrams (calibration curves) for each outcome class."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    plot_path = _REPORTS_DIR / "calibration_curves.png"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    colours = {"LogisticRegression": "#2563eb", "XGBoost": "#16a34a", "Bookmaker B365": "#dc2626"}

    for ax_idx, (class_label, ax) in enumerate(zip(LABEL_ORDER, axes)):
        y_binary = (y_true_str == class_label).astype(float).values

        for model_name, proba in predictions.items():
            class_proba = np.array(proba[:, LABEL_TO_IDX[class_label]], dtype=float)
            frac_pos, mean_pred = calibration_curve(y_binary, class_proba, n_bins=8, strategy="uniform")
            ax.plot(mean_pred, frac_pos, marker="o", linewidth=1.8,
                    label=model_name, color=colours.get(model_name, "grey"))

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect calibration")
        ax.set_title(f"Outcome: {class_label}", fontsize=11)
        ax.set_xlabel("Mean predicted probability")
        if ax_idx == 0:
            ax.set_ylabel("Fraction of positives")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

    fig.suptitle("Calibration Curves — Tier 1 Models vs Bookmaker", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def _plot_feature_importance(model: xgb.XGBClassifier, feature_names: list[str]) -> pathlib.Path:
    """XGBoost gain-based feature importance bar chart."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    plot_path = _REPORTS_DIR / "feature_importance_xgb.png"

    importance = model.feature_importances_
    idx = np.argsort(importance)[-20:]  # top 20

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        [feature_names[i] for i in idx],
        importance[idx],
        color="#16a34a",
        alpha=0.85,
    )
    ax.set_xlabel("Feature importance (gain)", fontsize=11)
    ax.set_title("XGBoost Feature Importance — Tier 1 Model\n(top 20 features by gain)", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def _plot_summary(results: list[dict]) -> pathlib.Path:
    """Bar chart comparing log loss across models including bookmaker."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    plot_path = _REPORTS_DIR / "model_comparison.png"

    names  = [r["model"] for r in results]
    losses = [r["log_loss"] for r in results]
    briers = [r["brier"] for r in results]
    colours = ["#2563eb", "#16a34a", "#dc2626", "#9333ea"][:len(names)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    bars1 = ax1.bar(names, losses, color=colours, alpha=0.85)
    ax1.set_title("Log Loss (lower = better)", fontsize=11)
    ax1.set_ylabel("Log Loss")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars1, losses):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    bars2 = ax2.bar(names, briers, color=colours, alpha=0.85)
    ax2.set_title("Brier Score (lower = better)", fontsize=11)
    ax2.set_ylabel("Brier Score")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars2, briers):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=15, ha="right")
    fig.suptitle("Tier 1 Model Comparison — 2024 Test Season", fontsize=13)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


# ── Database logging ──────────────────────────────────────────────────────────

def _log_predictions(
    engine,
    test_df: pd.DataFrame,
    proba: np.ndarray,
    model_version: str,
    tier: int = 1,
) -> int:
    """Upsert model predictions into the predictions table.

    Returns the number of rows inserted/updated.
    """
    inserted = 0
    with engine.connect() as conn:
        for i, row in test_df.iterrows():
            conn.execute(
                text("""
                    INSERT INTO predictions
                        (match_id, tier, model_version, prob_home, prob_draw, prob_away)
                    VALUES
                        (:match_id, :tier, :model_version, :prob_home, :prob_draw, :prob_away)
                    ON CONFLICT (match_id, tier, model_version) DO UPDATE SET
                        prob_home = EXCLUDED.prob_home,
                        prob_draw = EXCLUDED.prob_draw,
                        prob_away = EXCLUDED.prob_away
                """),
                {
                    "match_id":      int(row["match_id"]),
                    "tier":          tier,
                    "model_version": model_version,
                    "prob_home":     float(proba[i, LABEL_TO_IDX["H"]]),
                    "prob_draw":     float(proba[i, LABEL_TO_IDX["D"]]),
                    "prob_away":     float(proba[i, LABEL_TO_IDX["A"]]),
                },
            )
            inserted += 1
        conn.commit()
    return inserted


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Tier 1 baseline models")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip feature analysis step (saves ~1 min)")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip writing predictions to database")
    args = parser.parse_args()

    _REPORTS_DIR.mkdir(exist_ok=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Feature analysis ──────────────────────────────────────────────────────
    if not args.skip_analysis:
        logger.info("=== Step 1: Feature analysis ===")
        from models.analyse_features import run_analysis
        run_analysis()

    # ── Load splits ───────────────────────────────────────────────────────────
    logger.info("=== Step 2: Loading train/val/test splits ===")
    train, val, test = load_splits()

    X_train, y_train = xy(train)
    X_val,   y_val   = xy(val)
    X_test,  y_test  = xy(test)

    # Target as string for metrics (LABEL_ORDER-indexed)
    y_test_str  = test[TARGET_STR_COL]

    # Drop features that are 100% null in the training set.
    # These cannot be imputed and add no information (currently the 4 xG columns).
    # We check only the train set — val/test are NOT inspected here.
    null_in_train = X_train.isna().mean()
    scorable_cols = [c for c in X_train.columns if null_in_train[c] < 1.0]
    dropped_cols  = [c for c in X_train.columns if null_in_train[c] >= 1.0]
    if dropped_cols:
        logger.info("Dropping 100%%-null features (train set): %s", dropped_cols)
    X_train = X_train[scorable_cols]
    X_val   = X_val[scorable_cols]
    X_test  = X_test[scorable_cols]

    feature_names = scorable_cols
    logger.info(
        "Train: %d  Val: %d  Test: %d  Features: %d",
        len(X_train), len(X_val), len(X_test), len(feature_names),
    )

    # Encode integer result_label (1/0/-1) → class index (0=H, 1=D, 2=A)
    y_train_enc = y_train.map(INT_LABEL_TO_IDX)
    y_val_enc   = y_val.map(INT_LABEL_TO_IDX)

    # ── Train logistic regression ─────────────────────────────────────────────
    logger.info("=== Step 3: Training logistic regression ===")
    logreg = train_logreg(X_train, y_train, X_val, y_val)

    logger.info("=== Step 4: Training XGBoost ===")
    xgb_model = train_xgb(X_train, y_train, X_val, y_val)

    # ── Test-set predictions ──────────────────────────────────────────────────
    logger.info("=== Step 5: Evaluating on 2024 test set ===")
    logreg_proba = logreg.predict_proba(X_test)
    xgb_proba    = xgb_model.predict_proba(X_test)

    # ── Bookmaker benchmark ───────────────────────────────────────────────────
    engine = create_engine(DATABASE_URL)
    bm = _load_bookmaker_benchmark(test["match_id"], engine)

    if bm is not None:
        # Merge bm with test to align rows (test may have rows with no odds)
        test_with_bm = test.merge(bm, on="match_id", how="left")
        bm_mask = test_with_bm["bm_home"].notna()
        bm_aligned = test_with_bm[bm_mask]
        bm_proba = bm_aligned[["bm_home", "bm_draw", "bm_away"]].values
        y_bm_str = bm_aligned[TARGET_STR_COL]
        logger.info("Bookmaker odds available for %d/%d test matches.", len(bm_aligned), len(test))
    else:
        logger.warning("No bookmaker odds found for test set — skipping benchmark.")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION — Season 2024")
    print("=" * 60)

    all_results = []
    all_results.append(_print_eval("Logistic Regression", y_test_str, logreg_proba))
    all_results.append(_print_eval("XGBoost", y_test_str, xgb_proba))

    if bm is not None and len(bm_aligned) > 0:
        bm_ll = _log_loss_from_bm(y_bm_str, bm_aligned)
        bm_bs = _brier_from_bm(y_bm_str, bm_aligned)
        all_results.append({"model": "Bookmaker B365", "log_loss": bm_ll, "brier": bm_bs})
        print(f"\n{'─' * 50}")
        print("  Bookmaker B365 (benchmark)")
        print(f"{'─' * 50}")
        print(f"  Log loss  : {bm_ll:.4f}  (on {len(bm_aligned)} matches with odds)")
        print(f"  Brier     : {bm_bs:.4f}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        gap = ""
        if bm is not None and r["model"] != "Bookmaker B365":
            bm_entry = next((x for x in all_results if x["model"] == "Bookmaker B365"), None)
            if bm_entry:
                delta = r["log_loss"] - bm_entry["log_loss"]
                gap = f"  (Δ vs bookmaker: {delta:+.4f})"
        print(f"  {r['model']:<28}  log_loss={r['log_loss']:.4f}  brier={r['brier']:.4f}{gap}")
    print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("=== Step 6: Generating plots ===")
    predictions_dict = {
        "LogisticRegression": logreg_proba,
        "XGBoost":            xgb_proba,
    }
    if bm is not None and len(bm_aligned) > 0:
        # Calibration curves need aligned arrays — use only matches with odds
        bm_positions = bm_mask[bm_mask].index.tolist()
        logreg_proba_bm = logreg.predict_proba(X_test.iloc[bm_positions])
        xgb_proba_bm    = xgb_model.predict_proba(X_test.iloc[bm_positions])
        predictions_bm  = {
            "LogisticRegression": logreg_proba_bm,
            "XGBoost":            xgb_proba_bm,
            "Bookmaker B365":     bm_proba,
        }
        cal_path = _plot_calibration(y_bm_str, predictions_bm, bm_aligned)
    else:
        cal_path = _plot_calibration(y_test_str, predictions_dict, None)

    imp_path     = _plot_feature_importance(xgb_model, feature_names)
    summary_path = _plot_summary(all_results)

    logger.info("Calibration plot: %s", cal_path)
    logger.info("XGB importance:   %s", imp_path)
    logger.info("Model comparison: %s", summary_path)

    # ── Save results JSON ─────────────────────────────────────────────────────
    results_path = _REPORTS_DIR / "tier1_results.json"
    results_path.write_text(json.dumps({
        "split": {"train": 2022, "val": 2023, "test": 2024},
        "n_features": len(feature_names),
        "results": all_results,
    }, indent=2))
    logger.info("Results JSON: %s", results_path)

    # ── Log predictions to DB ─────────────────────────────────────────────────
    if not args.no_db:
        logger.info("=== Step 7: Logging predictions to database ===")
        n_logreg = _log_predictions(engine, test.reset_index(drop=True), logreg_proba, _LOGREG_VERSION)
        n_xgb    = _log_predictions(engine, test.reset_index(drop=True), xgb_proba,    _XGB_VERSION)
        logger.info("Logged %d LogReg + %d XGBoost predictions to DB.", n_logreg, n_xgb)

    logger.info("Phase 3 complete.")


if __name__ == "__main__":
    main()
