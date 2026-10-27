"""
Phase 7e — SHAP Explainability.

Computes TreeExplainer SHAP values for XGBoost trained on Tier 1 features
(train+val = 2022–2023) and evaluated on the 2024 test season.

Produces six publication-quality figures and a JSON results file:
  reports/shap_beeswarm.png         — beeswarm (Home Win class), top 15 features
  reports/shap_bar.png              — mean |SHAP| bar across all 3 classes
  reports/shap_waterfall_correct.png — waterfall for highest-confidence correct prediction
  reports/shap_waterfall_wrong.png   — waterfall for highest-confidence wrong prediction
  reports/shap_waterfall_draw.png    — waterfall for most-uncertain prediction
  reports/shap_draw_analysis.png     — mean SHAP for Draw class, top 10 features
  reports/shap_results.json

Usage:
    python models/explain.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # headless — no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from models.split import INT_LABEL_TO_IDX, load_splits, xy

_REPORTS_DIR = pathlib.Path("reports")

# Class index constants — matches INT_LABEL_TO_IDX = {1:0, 0:1, -1:2}
_CLASS_H = 0  # Home Win
_CLASS_D = 1  # Draw
_CLASS_A = 2  # Away Win

_LABEL_NAMES = {_CLASS_H: "H", _CLASS_D: "D", _CLASS_A: "A"}


# ── Model construction ────────────────────────────────────────────────────────


def _xgb_classifier() -> XGBClassifier:
    """XGBoost with the Phase 3/7c reference hyperparameters."""
    return XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=200,
        objective="multi:softprob",
        num_class=3,
        importance_type="gain",
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
    )


def _build_pipeline() -> Pipeline:
    """Impute-then-XGBoost pipeline (no scaling — trees don't need it)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb",     _xgb_classifier()),
    ])


# ── Data preparation ──────────────────────────────────────────────────────────


def _load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Load splits, encode labels, drop 100%-null features, and return:
        X_full, X_test, y_full_enc, y_test_enc, feature_names
    """
    train, val, test = load_splits()

    X_train, y_train = xy(train)
    X_val,   y_val   = xy(val)
    X_test,  y_test  = xy(test)

    # Drop columns that are entirely null in the training set (e.g. xG cols)
    null_rate  = X_train.isna().mean()
    scorable   = [c for c in X_train.columns if null_rate[c] < 1.0]
    X_train    = X_train[scorable].reset_index(drop=True)
    X_val      = X_val[scorable].reset_index(drop=True)
    X_test     = X_test[scorable].reset_index(drop=True)
    feature_names = list(scorable)

    y_train_enc = y_train.map(INT_LABEL_TO_IDX).values
    y_val_enc   = y_val.map(INT_LABEL_TO_IDX).values
    y_test_enc  = y_test.map(INT_LABEL_TO_IDX).values

    X_full     = pd.concat([X_train, X_val], ignore_index=True)
    y_full_enc = np.concatenate([y_train_enc, y_val_enc])

    logger.info(
        "Train+Val: %d  Test: %d  Features: %d",
        len(X_full), len(X_test), len(feature_names),
    )
    return X_full, X_test, y_full_enc, y_test_enc, feature_names


# ── Gain rank lookup ──────────────────────────────────────────────────────────


def _load_gain_ranks(
    pipeline: Pipeline,
    feature_names: list[str],
) -> dict[str, int]:
    """
    Return per-feature XGBoost gain rank (1 = highest gain).
    Reads from reports/dim_reduction_results.json if available;
    otherwise re-derives from the fitted pipeline.
    """
    json_path = _REPORTS_DIR / "dim_reduction_results.json"
    if json_path.exists():
        with open(json_path) as fh:
            data = json.load(fh)
        details = data.get("lasso_selection", {}).get("feature_details", {})
        if details:
            logger.info("Loaded XGB gain ranks from %s", json_path)
            return {feat: info["xgb_gain_rank"] for feat, info in details.items()}

    logger.info("dim_reduction_results.json not found — deriving gain ranks from model")
    gain_scores = pipeline["xgb"].feature_importances_
    gain_order  = np.argsort(-gain_scores)           # descending
    rank_arr    = np.empty_like(gain_order)
    rank_arr[gain_order] = np.arange(1, len(gain_order) + 1)
    return {feature_names[i]: int(rank_arr[i]) for i in range(len(feature_names))}


# ── SHAP computation ─────────────────────────────────────────────────────────


def _compute_shap(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    feature_names: list[str],
) -> shap.Explanation:
    """
    Compute exact TreeExplainer SHAP values for the test set.

    Returns a shap.Explanation with shape (n_test, n_features, n_classes).
    Feature names are embedded so waterfall / beeswarm labels are readable.
    """
    X_test_imp = pipeline["imputer"].transform(X_test)
    X_test_imp_df = pd.DataFrame(X_test_imp, columns=feature_names)

    explainer   = shap.TreeExplainer(pipeline["xgb"])
    shap_values = explainer(X_test_imp_df)
    logger.info("SHAP values computed: shape %s", str(shap_values.values.shape))
    return shap_values


# ── Illustrative match selection ─────────────────────────────────────────────


def _select_illustrative_matches(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test_enc: np.ndarray,
) -> dict[str, dict]:
    """
    Select three illustrative test matches:
      high_conf_correct — highest max_prob where prediction == actual
      high_conf_wrong   — highest max_prob where prediction != actual
      near_draw         — lowest max_prob (most uncertain / closest to uniform)
    """
    proba       = pipeline.predict_proba(X_test)          # (380, 3)
    max_probs   = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct_mask = predictions == y_test_enc

    # High-confidence correct
    correct_indices = np.where(correct_mask)[0]
    hcc_local = int(correct_indices[np.argmax(max_probs[correct_indices])])

    # High-confidence wrong
    wrong_indices = np.where(~correct_mask)[0]
    hcw_local = int(wrong_indices[np.argmax(max_probs[wrong_indices])])

    # Most uncertain (lowest max probability)
    uncertain_local = int(np.argmin(max_probs))

    def _record(idx: int) -> dict:
        return {
            "match_idx":  idx,
            "predicted":  _LABEL_NAMES[int(predictions[idx])],
            "actual":     _LABEL_NAMES[int(y_test_enc[idx])],
            "max_prob":   round(float(max_probs[idx]), 4),
            "pred_class": int(predictions[idx]),
        }

    matches = {
        "high_conf_correct": _record(hcc_local),
        "high_conf_wrong":   _record(hcw_local),
        "near_draw":         _record(uncertain_local),
    }

    for key, m in matches.items():
        logger.info(
            "  %s → idx=%d  pred=%s  actual=%s  max_prob=%.3f",
            key, m["match_idx"], m["predicted"], m["actual"], m["max_prob"],
        )
    return matches


# ── Figure helpers ────────────────────────────────────────────────────────────


def _save_current_fig(path: pathlib.Path, width: float, height: float, title: str) -> None:
    """Resize, title, and save the figure that shap just created."""
    fig = plt.gcf()
    fig.set_size_inches(width, height)
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ── Figures ───────────────────────────────────────────────────────────────────


def _plot_beeswarm(shap_values: shap.Explanation, n_test: int) -> None:
    """Beeswarm plot for the Home Win class (class 0), top 15 features."""
    shap.plots.beeswarm(shap_values[:, :, _CLASS_H], max_display=15, show=False)
    _save_current_fig(
        _REPORTS_DIR / "shap_beeswarm.png",
        width=12, height=7,
        title=f"SHAP Beeswarm — XGBoost (Home Win Class)\nEPL 2024 Test Set (n={n_test})",
    )


def _plot_bar(shap_values: shap.Explanation, feature_names: list[str], n_test: int) -> None:
    """
    Mean |SHAP| bar chart across all 3 classes, top 15 features.

    shap.plots.bar has an IndexError with multi-output Explanation objects in
    shap 0.49.x, so we compute mean |SHAP| manually and draw with matplotlib.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=(0, 2))   # (n_features,)
    top15_idx = np.argsort(mean_abs)[::-1][:15]
    feats  = [feature_names[i] for i in top15_idx]
    vals   = mean_abs[top15_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feats[::-1], vals[::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value| (averaged across H/D/A classes)", fontsize=10)
    ax.set_title(
        f"Mean |SHAP| — XGBoost (all classes)\nEPL 2024 Test Set (n={n_test})",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = _REPORTS_DIR / "shap_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _plot_waterfall(
    shap_values: shap.Explanation,
    match_info: dict,
    filename: str,
) -> None:
    """Waterfall for a single match/class slice."""
    idx        = match_info["match_idx"]
    pred_class = match_info["pred_class"]
    predicted  = match_info["predicted"]
    actual     = match_info["actual"]
    max_prob   = match_info["max_prob"]

    shap.plots.waterfall(shap_values[idx, :, pred_class], max_display=15, show=False)
    _save_current_fig(
        _REPORTS_DIR / filename,
        width=10, height=6,
        title=(
            f"SHAP Waterfall — Predicted: {predicted}  Actual: {actual}  "
            f"Max P: {max_prob:.3f}\nXGBoost, EPL 2024 Test Set"
        ),
    )


def _plot_draw_analysis(shap_values: shap.Explanation, feature_names: list[str]) -> list[dict]:
    """
    Bar chart of mean SHAP value for the Draw class (class 1), top 10 by |mean|.
    Returns the draw SHAP data for JSON output.
    """
    draw_shap_mean = shap_values.values[:, :, _CLASS_D].mean(axis=0)  # (n_features,)
    top10_idx = np.argsort(np.abs(draw_shap_mean))[::-1][:10]

    features_top10   = [feature_names[i] for i in top10_idx]
    shap_vals_top10  = draw_shap_mean[top10_idx]
    colours          = ["#2196F3" if v >= 0 else "#F44336" for v in shap_vals_top10]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features_top10[::-1], shap_vals_top10[::-1], color=colours[::-1])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean SHAP value (Draw class)", fontsize=10)
    ax.set_title(
        f"Mean SHAP for Draw Outcome — XGBoost\nEPL 2024 Test Set (n={shap_values.values.shape[0]})",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = _REPORTS_DIR / "shap_draw_analysis.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)

    return [
        {
            "feature":          features_top10[i],
            "mean_shap_draw":   round(float(shap_vals_top10[i]), 6),
            "direction":        "toward_draw" if shap_vals_top10[i] >= 0 else "away_from_draw",
        }
        for i in range(len(features_top10))
    ]


# ── Ranking table ─────────────────────────────────────────────────────────────


def _build_shap_ranking(
    shap_values: shap.Explanation,
    feature_names: list[str],
    gain_ranks: dict[str, int],
) -> list[dict]:
    """
    Compute mean |SHAP| per feature (averaged across all 3 classes) and
    return top-15 rows with SHAP rank, gain rank, and delta.
    """
    mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))  # (n_features,)
    shap_order    = np.argsort(-mean_abs_shap)

    top15 = []
    for rank, idx in enumerate(shap_order[:15], start=1):
        feat      = feature_names[idx]
        gain_rank = gain_ranks.get(feat, None)
        top15.append({
            "rank":          rank,
            "feature":       feat,
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
            "gain_rank":     gain_rank,
            "delta_rank":    (gain_rank - rank) if gain_rank is not None else None,
        })
    return top15


# ── Summary printing ──────────────────────────────────────────────────────────


def _print_summary(shap_top15: list[dict], draw_top10: list[dict]) -> None:
    sep  = "=" * 66
    dash = "-" * 66

    print(f"\n{sep}")
    print("  PHASE 7e SHAP EXPLAINABILITY — XGBoost EPL 2024")
    print(f"{sep}")

    print("\nTop 15 features by mean |SHAP| (all classes):")
    print(f"  {'Rank':>4}  {'Feature':<30}  {'Mean|SHAP|':>10}  {'Gain Rank':>9}  {'Δ Rank':>6}")
    print(f"  {dash}")
    for row in shap_top15:
        delta_str = f"{row['delta_rank']:+d}" if row["delta_rank"] is not None else "—"
        print(
            f"  {row['rank']:>4}  {row['feature']:<30}  {row['mean_abs_shap']:>10.6f}"
            f"  {str(row['gain_rank'] or '—'):>9}  {delta_str:>6}"
        )

    print(f"\nDraw SHAP analysis (top 10 features pushing toward/away from draw):")
    print(f"  {'Feature':<30}  {'Mean SHAP (Draw)':>16}  Direction")
    print(f"  {dash}")
    for row in draw_top10:
        print(
            f"  {row['feature']:<30}  {row['mean_shap_draw']:>16.6f}  {row['direction']}"
        )

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _REPORTS_DIR.mkdir(exist_ok=True)

    # ── Load data and train model ──────────────────────────────────────────────
    X_full, X_test, y_full_enc, y_test_enc, feature_names = _load_and_prepare()

    logger.info("Training XGBoost on train+val (%d samples)...", len(X_full))
    pipeline = _build_pipeline()
    pipeline.fit(X_full, y_full_enc)
    logger.info("XGBoost training complete.")

    # ── Gain rank lookup ───────────────────────────────────────────────────────
    gain_ranks = _load_gain_ranks(pipeline, feature_names)

    # ── SHAP values ───────────────────────────────────────────────────────────
    logger.info("Computing TreeExplainer SHAP values on test set (%d samples)...", len(X_test))
    shap_values = _compute_shap(pipeline, X_test, feature_names)

    # ── Illustrative matches ───────────────────────────────────────────────────
    logger.info("Selecting illustrative matches...")
    illustrative = _select_illustrative_matches(pipeline, X_test, y_test_enc)

    # ── Figures ───────────────────────────────────────────────────────────────
    logger.info("Generating figures...")
    _plot_beeswarm(shap_values, len(X_test))
    _plot_bar(shap_values, feature_names, len(X_test))
    _plot_waterfall(shap_values, illustrative["high_conf_correct"], "shap_waterfall_correct.png")
    _plot_waterfall(shap_values, illustrative["high_conf_wrong"],   "shap_waterfall_wrong.png")
    _plot_waterfall(shap_values, illustrative["near_draw"],         "shap_waterfall_draw.png")
    draw_top10 = _plot_draw_analysis(shap_values, feature_names)

    # ── Rankings ───────────────────────────────────────────────────────────────
    shap_top15 = _build_shap_ranking(shap_values, feature_names, gain_ranks)

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(shap_top15, draw_top10)

    # ── JSON output ───────────────────────────────────────────────────────────
    results = {
        "model":      "XGBoost (train+val fit, test SHAP)",
        "n_test":     len(X_test),
        "n_features": len(feature_names),
        "shap_top15": shap_top15,
        "draw_shap_top10": draw_top10,
        "illustrative_matches": {
            k: {kk: vv for kk, vv in v.items() if kk != "pred_class"}
            for k, v in illustrative.items()
        },
    }
    json_path = _REPORTS_DIR / "shap_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Results written to %s", json_path)
    logger.info("Phase 7e complete.")


if __name__ == "__main__":
    main()
