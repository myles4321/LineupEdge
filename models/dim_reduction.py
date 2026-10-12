"""
Phase 7c — Dimensionality Reduction & Feature Selection Experiments.

Three experiments on the Tier 1 feature matrix (34 features, 2022–2024):

  1. PCA — scree analysis; XGBoost and Random Forest in PCA-reduced space
  2. LDA Projection — 2D discriminant scatter; LogReg and KNN in LDA space
  3. LASSO Feature Selection — 20-feature subset vs full-feature XGBoost;
     cross-reference LASSO survivors against XGBoost gain rankings

Usage:
    python models/dim_reduction.py
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from models.split import INT_LABEL_TO_IDX, load_splits, xy

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPORTS_DIR = pathlib.Path("reports")

# ── Reference log-loss values from prior phases ───────────────────────────────
_REF_XGB_LL   = 1.0201   # Phase 3 XGBoost on full 34 features
_REF_RF_LL    = 1.0250   # Phase 7a RF on full 34 features
_REF_LASSO_LL = 0.9964   # Phase 7a LASSO LogReg on full 34 features
_REF_KNN_LL   = 1.0131   # Phase 7a KNN (k=51) on full 34 features


# ── Shared helpers ────────────────────────────────────────────────────────────


def _evaluate_log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(log_loss(y_true, proba, labels=[0, 1, 2]))


def _xgb_classifier() -> XGBClassifier:
    """XGBoost with Phase 3 best hyperparameters."""
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


def _load_and_prep_data() -> tuple:
    """Load splits, drop 100%-null features, encode labels, build X_full."""
    logger.info("Loading train/val/test splits...")
    train, val, test = load_splits()

    X_train, y_train = xy(train)
    X_val,   y_val   = xy(val)
    X_test,  y_test  = xy(test)

    # Drop 100%-null features (xG columns until stats ingestion completes)
    null_in_train = X_train.isna().mean()
    scorable_cols = [c for c in X_train.columns if null_in_train[c] < 1.0]
    dropped_cols  = [c for c in X_train.columns if null_in_train[c] >= 1.0]
    if dropped_cols:
        logger.info("Dropping 100%%-null features: %s", dropped_cols)

    X_train = X_train[scorable_cols].reset_index(drop=True)
    X_val   = X_val[scorable_cols].reset_index(drop=True)
    X_test  = X_test[scorable_cols].reset_index(drop=True)
    X_full  = pd.concat([X_train, X_val], ignore_index=True)

    feature_names = list(scorable_cols)

    y_train_enc = y_train.map(INT_LABEL_TO_IDX).values
    y_val_enc   = y_val.map(INT_LABEL_TO_IDX).values
    y_test_enc  = y_test.map(INT_LABEL_TO_IDX).values
    y_full_enc  = np.concatenate([y_train_enc, y_val_enc])

    logger.info(
        "Train: %d  Val: %d  Test: %d  Full: %d  Features: %d",
        len(X_train), len(X_val), len(X_test), len(X_full), len(feature_names),
    )
    return (
        X_train, X_val, X_test, X_full,
        y_train_enc, y_val_enc, y_test_enc, y_full_enc,
        feature_names,
    )


# ── Experiment 1: PCA ─────────────────────────────────────────────────────────


def run_pca_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    n_features: int,
) -> dict:
    """
    PCA scree analysis + XGBoost / Random Forest in PCA-reduced space.

    Preprocessing pipeline (fit on X_train only):
        SimpleImputer(median) → StandardScaler → PCA
    """
    logger.info("=== Experiment 1: PCA ===")

    # ── Shared preprocessing — fit on X_train only ────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # ── Full-component PCA for scree analysis ─────────────────────────────────
    pca_full = PCA(n_components=n_features)
    pca_full.fit(X_train_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = int(np.argmax(cumvar >= 0.95) + 1)
    logger.info(
        "PCA: %d components retain ≥95%% variance (of %d total features)",
        n_components_95, n_features,
    )

    # ── Scree plot ─────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.bar(
        range(1, n_features + 1),
        pca_full.explained_variance_ratio_ * 100,
        color="steelblue", alpha=0.7, label="Per-component variance (%)",
    )
    ax1.set_xlabel("Principal Component", fontsize=11)
    ax1.set_ylabel("Explained Variance (%)", color="steelblue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xlim(0.4, n_features + 0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        range(1, n_features + 1),
        cumvar * 100,
        color="darkorange", linewidth=2, marker="o", markersize=4,
        label="Cumulative variance (%)",
    )
    ax2.axhline(95, color="crimson", linestyle="--", linewidth=1.2, label="95% threshold")
    ax2.axvline(
        n_components_95, color="crimson", linestyle=":", linewidth=1.2,
        label=f"n={n_components_95} components",
    )
    ax2.set_ylabel("Cumulative Variance (%)", color="darkorange", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.set_ylim(0, 108)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title(
        f"PCA Scree Plot — Tier 1 Features ({n_features} components)\n"
        f"{n_components_95} components retain ≥95% variance",
        fontsize=12,
    )
    plt.tight_layout()
    scree_path = _REPORTS_DIR / "pca_scree.png"
    fig.savefig(scree_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Scree plot saved to %s", scree_path)

    # ── Project all splits to 95%-variance PCA space ──────────────────────────
    pca_95 = PCA(n_components=n_components_95)
    X_train_pca = pca_95.fit_transform(X_train_scaled)
    X_test_pca  = pca_95.transform(X_test_scaled)

    # ── XGBoost in PCA space ───────────────────────────────────────────────────
    logger.info("Training XGBoost on %d PCA components...", n_components_95)
    xgb_pca = _xgb_classifier()
    xgb_pca.fit(X_train_pca, y_train_enc)
    xgb_pca_ll = _evaluate_log_loss(y_test_enc, xgb_pca.predict_proba(X_test_pca))
    logger.info("XGBoost-PCA LL=%.4f  (full: %.4f)", xgb_pca_ll, _REF_XGB_LL)

    # ── Random Forest in PCA space ─────────────────────────────────────────────
    logger.info("Training Random Forest on %d PCA components...", n_components_95)
    rf_pca = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf_pca.fit(X_train_pca, y_train_enc)
    rf_pca_ll = _evaluate_log_loss(y_test_enc, rf_pca.predict_proba(X_test_pca))
    logger.info("RF-PCA LL=%.4f  (full: %.4f)", rf_pca_ll, _REF_RF_LL)

    return {
        "n_components_95pct": n_components_95,
        "xgb_full_ll":   _REF_XGB_LL,
        "xgb_pca_ll":    round(xgb_pca_ll, 4),
        "rf_full_ll":    _REF_RF_LL,
        "rf_pca_ll":     round(rf_pca_ll, 4),
        "xgb_pca_delta": round(xgb_pca_ll - _REF_XGB_LL, 4),
        "rf_pca_delta":  round(rf_pca_ll  - _REF_RF_LL,  4),
    }


# ── Experiment 2: LDA Projection ──────────────────────────────────────────────


def run_lda_experiment(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_val_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> dict:
    """
    LDA 2D projection: scatter plot + LogReg and KNN in discriminant space.

    Preprocessing pipeline (fit on X_train only):
        SimpleImputer(median) → StandardScaler → LDA (solver='svd')

    LDA output (2D discriminant scores) has no NaN and is already a scaled
    linear transform — no further scaling needed for downstream models.
    """
    logger.info("=== Experiment 2: LDA Projection ===")

    # ── Preprocessing — fit on X_train only ──────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp   = imputer.transform(X_val)
    X_test_imp  = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled   = scaler.transform(X_val_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # ── LDA — fit on X_train, project all splits ──────────────────────────────
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(X_train_scaled, y_train_enc)

    X_train_lda = lda.transform(X_train_scaled)
    X_val_lda   = lda.transform(X_val_scaled)
    X_test_lda  = lda.transform(X_test_scaled)

    n_lda_components = X_train_lda.shape[1]
    logger.info("LDA: %d discriminant component(s)", n_lda_components)

    # ── Scatter plot (all splits combined, coloured by outcome) ───────────────
    X_all = np.vstack([X_train_lda, X_val_lda, X_test_lda])
    y_all = np.concatenate([y_train_enc, y_val_enc, y_test_enc])

    outcome_palette = {0: "steelblue", 1: "dimgray", 2: "crimson"}
    outcome_labels  = {0: "Home Win (H)", 1: "Draw (D)", 2: "Away Win (A)"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in [0, 1, 2]:
        mask = y_all == cls
        ax.scatter(
            X_all[mask, 0], X_all[mask, 1],
            c=outcome_palette[cls], alpha=0.4, s=15, rasterized=True,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=outcome_palette[cls], markersize=9,
               label=outcome_labels[cls])
        for cls in [0, 1, 2]
    ]
    ax.legend(handles=legend_handles, fontsize=10)
    ax.set_xlabel("LDA Component 1", fontsize=11)
    ax.set_ylabel("LDA Component 2", fontsize=11)
    ax.set_title(
        "LDA 2D Projection — Tier 1 Features\n"
        "Coloured by Match Outcome  (train 2022 + val 2023 + test 2024)",
        fontsize=11,
    )
    plt.tight_layout()
    scatter_path = _REPORTS_DIR / "lda_scatter.png"
    fig.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("LDA scatter saved to %s", scatter_path)

    # ── LogReg in 2D LDA space ─────────────────────────────────────────────────
    logger.info("Training LogReg (C=0.1) in 2D LDA space...")
    logreg_lda = LogisticRegression(
        C=0.1, solver="lbfgs", multi_class="multinomial",
        max_iter=1000, random_state=42,
    )
    logreg_lda.fit(X_train_lda, y_train_enc)
    logreg_lda_ll = _evaluate_log_loss(y_test_enc, logreg_lda.predict_proba(X_test_lda))
    logger.info("LogReg-LDA LL=%.4f  (full LASSO LR: %.4f)", logreg_lda_ll, _REF_LASSO_LL)

    # ── KNN in 2D LDA space ────────────────────────────────────────────────────
    logger.info("Training KNN (k=51) in 2D LDA space...")
    knn_lda = KNeighborsClassifier(n_neighbors=51, weights="distance")
    knn_lda.fit(X_train_lda, y_train_enc)
    knn_lda_ll = _evaluate_log_loss(y_test_enc, knn_lda.predict_proba(X_test_lda))
    logger.info("KNN-LDA LL=%.4f  (full KNN: %.4f)", knn_lda_ll, _REF_KNN_LL)

    return {
        "n_components":       n_lda_components,
        "logreg_full_ll":     _REF_LASSO_LL,
        "logreg_lda_ll":      round(logreg_lda_ll, 4),
        "knn_full_ll":        _REF_KNN_LL,
        "knn_lda_ll":         round(knn_lda_ll, 4),
        "logreg_lda_delta":   round(logreg_lda_ll - _REF_LASSO_LL, 4),
        "knn_lda_delta":      round(knn_lda_ll  - _REF_KNN_LL,   4),
    }


# ── Experiment 3: LASSO Feature Selection ─────────────────────────────────────


def run_lasso_selection_experiment(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    X_full: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    y_full_enc: np.ndarray,
    feature_names: list,
) -> dict:
    """
    LASSO feature selection: identify surviving features, train XGBoost on
    the subset, and cross-reference with XGBoost gain rankings.

    The LASSO selector is fit on X_full (train+val) — more data yields more
    stable feature selection. The final XGBoost is trained on X_train only
    to keep evaluation fair.
    """
    logger.info("=== Experiment 3: LASSO Feature Selection ===")

    # ── Fit LASSO on X_full to identify surviving features ────────────────────
    lasso_selector = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("lasso",  LogisticRegression(
            penalty="l1", C=0.1, solver="saga", multi_class="ovr",
            max_iter=2000, random_state=42,
        )),
    ])
    logger.info("Fitting LASSO on X_full (train+val) to identify surviving features...")
    lasso_selector.fit(X_full, y_full_enc)

    coef = lasso_selector.named_steps["lasso"].coef_  # shape (3, n_features) — OVR
    # Surviving: features where at least one OVR classifier has non-zero coef
    zero_mask      = np.all(np.abs(coef) < 1e-10, axis=0)
    surviving_mask = ~zero_mask
    surviving_features = [f for f, keep in zip(feature_names, surviving_mask) if keep]
    mean_abs_coef  = np.mean(np.abs(coef), axis=0)

    logger.info(
        "LASSO survivors: %d of %d features (%d zeroed)",
        len(surviving_features), len(feature_names), zero_mask.sum(),
    )

    # ── XGBoost gain ranking on all 34 features ───────────────────────────────
    logger.info("Fitting XGBoost on full features to compute gain rankings...")
    imputer_xgb = SimpleImputer(strategy="median")
    X_train_imp  = imputer_xgb.fit_transform(X_train)

    xgb_for_gain = _xgb_classifier()
    xgb_for_gain.fit(X_train_imp, y_train_enc)
    gain_scores = xgb_for_gain.feature_importances_  # gain importance (set in constructor)

    # Rank 1 = highest gain; features with zero gain share last rank
    gain_rank_arr = np.argsort(np.argsort(-gain_scores)) + 1
    gain_rank = {feature_names[i]: int(gain_rank_arr[i]) for i in range(len(feature_names))}

    # ── Print cross-reference table ────────────────────────────────────────────
    print("\nLASSO Survivors — XGBoost Gain Cross-Reference")
    print("-" * 72)
    print(f"  {'Feature':<36}  {'mean|coef|':>10}  {'XGB rank':>8}  {'Top-10?':>8}")
    print("-" * 72)

    feature_details: dict = {}
    # Sort survivors by descending mean |coef|
    survivors_sorted = sorted(
        surviving_features,
        key=lambda f: -mean_abs_coef[feature_names.index(f)],
    )
    for feat in survivors_sorted:
        feat_idx   = feature_names.index(feat)
        coef_val   = float(mean_abs_coef[feat_idx])
        xgb_rank   = gain_rank[feat]
        in_both_10 = "YES" if xgb_rank <= 10 else "—"
        print(f"  {feat:<36}  {coef_val:>10.4f}  {xgb_rank:>8}  {in_both_10:>8}")
        feature_details[feat] = {
            "mean_abs_coef": round(coef_val, 6),
            "xgb_gain_rank": xgb_rank,
        }
    print("-" * 72)

    # ── XGBoost on LASSO-selected feature subset ──────────────────────────────
    logger.info(
        "Training XGBoost on %d LASSO-selected features...", len(surviving_features),
    )
    X_train_sel = X_train[surviving_features].reset_index(drop=True)
    X_test_sel  = X_test[surviving_features].reset_index(drop=True)

    xgb_lasso_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    _xgb_classifier()),
    ])
    xgb_lasso_pipe.fit(X_train_sel, y_train_enc)
    xgb_lasso_ll = _evaluate_log_loss(
        y_test_enc, xgb_lasso_pipe.predict_proba(X_test_sel),
    )
    logger.info(
        "XGBoost-LASSO-sel LL=%.4f  (full XGB: %.4f)", xgb_lasso_ll, _REF_XGB_LL,
    )

    return {
        "n_surviving_features": len(surviving_features),
        "surviving_features":   surviving_features,
        "feature_details":      feature_details,
        "xgb_full_ll":          _REF_XGB_LL,
        "xgb_lasso_ll":         round(xgb_lasso_ll, 4),
        "xgb_lasso_delta":      round(xgb_lasso_ll - _REF_XGB_LL, 4),
    }


# ── Summary table ─────────────────────────────────────────────────────────────


def _print_summary_table(
    pca_results: dict,
    lda_results: dict,
    lasso_results: dict,
) -> None:
    n = pca_results["n_components_95pct"]
    sep  = "=" * 70
    dash = "-" * 70
    print(f"\n{sep}")
    print(f"  PHASE 7c DIMENSIONALITY REDUCTION — EPL 2024 TEST SEASON")
    print(f"{sep}")
    print(f"  {'Experiment':<24}  {'Method':<18}  {'Log Loss':>8}  {'Δ vs Full':>10}")
    print(dash)

    experiments = [
        (f"PCA (n={n})",     "XGBoost-PCA",   pca_results["xgb_pca_ll"],    pca_results["xgb_pca_delta"]),
        (f"PCA (n={n})",     "RF-PCA",         pca_results["rf_pca_ll"],     pca_results["rf_pca_delta"]),
        ("LDA (2D)",         "LogReg-LDA",     lda_results["logreg_lda_ll"], lda_results["logreg_lda_delta"]),
        ("LDA (2D)",         "KNN-LDA",        lda_results["knn_lda_ll"],    lda_results["knn_lda_delta"]),
        ("LASSO selection",  "XGB-LASSO-sel",  lasso_results["xgb_lasso_ll"], lasso_results["xgb_lasso_delta"]),
    ]
    for experiment, method, ll, delta in experiments:
        sign = "+" if delta >= 0 else ""
        print(f"  {experiment:<24}  {method:<18}  {ll:>8.4f}  {sign}{delta:>9.4f}")

    print(dash)
    references = [
        ("Reference", "Full XGBoost",  _REF_XGB_LL),
        ("Reference", "Full RF",       _REF_RF_LL),
        ("Reference", "Full LASSO LR", _REF_LASSO_LL),
        ("Reference", "Full KNN",      _REF_KNN_LL),
    ]
    for experiment, method, ll in references:
        print(f"  {experiment:<24}  {method:<18}  {ll:>8.4f}  {'—':>10}")
    print(f"{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _REPORTS_DIR.mkdir(exist_ok=True)

    (
        X_train, X_val, X_test, X_full,
        y_train_enc, y_val_enc, y_test_enc, y_full_enc,
        feature_names,
    ) = _load_and_prep_data()

    pca_results = run_pca_experiment(
        X_train, X_test,
        y_train_enc, y_test_enc,
        len(feature_names),
    )
    lda_results = run_lda_experiment(
        X_train, X_val, X_test,
        y_train_enc, y_val_enc, y_test_enc,
    )
    lasso_results = run_lasso_selection_experiment(
        X_train, X_val, X_test, X_full,
        y_train_enc, y_test_enc, y_full_enc,
        feature_names,
    )

    _print_summary_table(pca_results, lda_results, lasso_results)

    results = {
        "split":             {"train": 2022, "val": 2023, "test": 2024},
        "n_features_full":   len(feature_names),
        "pca":               pca_results,
        "lda":               lda_results,
        "lasso_selection":   lasso_results,
    }
    json_path = _REPORTS_DIR / "dim_reduction_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", json_path)
    logger.info("Phase 7c complete.")


if __name__ == "__main__":
    main()
