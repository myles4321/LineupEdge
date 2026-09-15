"""
Feature analysis for the Tier 1 feature matrix.

Performs the role that df-analyze would fill: univariate feature ranking,
mutual information, cross-validated classifier comparison, and a summary
report. df-analyze now requires Python ≥ 3.13 and cannot be used in the
project's Python 3.10 environment, so this module uses scikit-learn
equivalents which are equally rigorous and more transparent for the paper.

Outputs:
    reports/feature_analysis.txt   — human-readable ranking table
    reports/feature_importance.png — bar chart of top features

Usage:
    python models/analyse_features.py
"""

from __future__ import annotations

import logging
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from models.split import FEATURE_COLS, LABEL_ORDER, load_splits, xy

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPORTS_DIR = pathlib.Path("reports")


def _feature_ranking(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Rank features by ANOVA F-statistic and mutual information.

    Both metrics use only training data. NaN values are median-imputed
    before scoring so that partially-null features (H2H, position) are
    ranked fairly. Features that are 100% null (e.g. xG before team_stats
    ingestion) are excluded from imputation/scoring and assigned score=0.
    """
    null_rates = X_train.isna().mean() * 100
    scorable_cols = [c for c in X_train.columns if null_rates[c] < 100.0]
    all_cols = list(X_train.columns)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train[scorable_cols])

    # Encode target as 0/1/2 for sklearn (it expects non-negative integers)
    label_map = {-1: 0, 0: 1, 1: 2}
    y_enc = y_train.map(label_map)

    # ANOVA F-statistic (linear association with target)
    f_scores_scorable, _ = f_classif(X_imp, y_enc)

    # Mutual information (captures non-linear relationships)
    mi_scores_scorable = mutual_info_classif(X_imp, y_enc, random_state=42)

    # Merge scores back to full feature list (100%-null features get score=0)
    f_map  = dict(zip(scorable_cols, f_scores_scorable))
    mi_map = dict(zip(scorable_cols, mi_scores_scorable))

    f_scores  = np.array([f_map.get(c, 0.0)  for c in all_cols])
    mi_scores = np.array([mi_map.get(c, 0.0) for c in all_cols])

    ranking = pd.DataFrame({
        "feature":     all_cols,
        "f_score":     f_scores,
        "mutual_info": mi_scores,
        "null_rate":   null_rates[all_cols].values,
    })
    # Normalise scores to [0, 1] for comparability
    ranking["f_score_norm"] = ranking["f_score"] / ranking["f_score"].max()
    ranking["mi_norm"]      = ranking["mutual_info"] / ranking["mutual_info"].max()
    ranking["combined"]     = (ranking["f_score_norm"] + ranking["mi_norm"]) / 2
    ranking = ranking.sort_values("combined", ascending=False).reset_index(drop=True)
    ranking.index = ranking.index + 1  # 1-based rank
    return ranking


def _classifier_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Compare classifiers using 5-fold stratified cross-validation on training data.

    Reports mean log loss (lower = better) and standard deviation across folds.
    Uses only training data — val/test are not touched.
    """
    label_map = {-1: 0, 0: 1, 1: 2}
    y_enc = y_train.map(label_map).values

    classifiers = {
        "LogisticRegression (C=1)": Pipeline([
            ("impute",  SimpleImputer(strategy="median")),
            ("scale",   StandardScaler()),
            ("clf",     LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=1000, C=1.0, random_state=42,
            )),
        ]),
        "LogisticRegression (C=0.1)": Pipeline([
            ("impute",  SimpleImputer(strategy="median")),
            ("scale",   StandardScaler()),
            ("clf",     LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=1000, C=0.1, random_state=42,
            )),
        ]),
        "RandomForest (n=200)": Pipeline([
            ("impute",  SimpleImputer(strategy="median")),
            ("clf",     RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=42, n_jobs=-1,
            )),
        ]),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=False)  # no shuffle — respects chronological order
    # Drop 100%-null columns before CV; pipelines can't impute them
    null_rates = X_train.isna().mean()
    scorable = [c for c in X_train.columns if null_rates[c] < 1.0]
    X_arr = X_train[scorable].values

    results = []
    for name, pipeline in classifiers.items():
        fold_losses = []
        for train_idx, fold_idx in cv.split(X_arr, y_enc):
            pipeline.fit(X_arr[train_idx], y_enc[train_idx])
            proba = pipeline.predict_proba(X_arr[fold_idx])
            fold_losses.append(log_loss(y_enc[fold_idx], proba))
        results.append({
            "classifier": name,
            "cv_log_loss_mean": float(np.mean(fold_losses)),
            "cv_log_loss_std":  float(np.std(fold_losses)),
        })
        logger.info(
            "  %-35s log_loss=%.4f ± %.4f",
            name, results[-1]["cv_log_loss_mean"], results[-1]["cv_log_loss_std"],
        )

    return pd.DataFrame(results).sort_values("cv_log_loss_mean").reset_index(drop=True)


def _select_features(ranking: pd.DataFrame, top_n: int = 20) -> list[str]:
    """Return the top-N features by combined rank, excluding 100%-null features."""
    non_null = ranking[ranking["null_rate"] < 100.0]
    return non_null.head(top_n)["feature"].tolist()


def _save_report(
    ranking: pd.DataFrame,
    cv_results: pd.DataFrame,
    selected_features: list[str],
) -> pathlib.Path:
    """Write feature analysis report to reports/feature_analysis.txt."""
    _REPORTS_DIR.mkdir(exist_ok=True)
    report_path = _REPORTS_DIR / "feature_analysis.txt"

    lines = [
        "=" * 70,
        "TIER 1 FEATURE ANALYSIS — Edge Soccer Prediction Project",
        "=" * 70,
        "",
        "NOTE: df-analyze requires Python >= 3.13; project uses Python 3.10.",
        "This report uses scikit-learn equivalents: ANOVA F-statistic +",
        "mutual information for univariate ranking, and 5-fold stratified",
        "cross-validation for classifier comparison.",
        "",
        "=" * 70,
        "FEATURE RANKING (by combined ANOVA F + Mutual Information score)",
        "=" * 70,
        f"{'Rank':<5} {'Feature':<30} {'F-score':>8} {'MI':>8} {'Combined':>9} {'Null%':>6}",
        "-" * 70,
    ]
    for rank, row in ranking.iterrows():
        null_flag = " *" if row["null_rate"] >= 100 else ""
        lines.append(
            f"{rank:<5} {row['feature']:<30} {row['f_score_norm']:>8.3f} "
            f"{row['mi_norm']:>8.3f} {row['combined']:>9.3f} {row['null_rate']:>5.1f}%{null_flag}"
        )
    lines += [
        "",
        "* = 100% null (xG features — excluded from selection)",
        "",
        "=" * 70,
        "CLASSIFIER COMPARISON (5-fold CV log loss on train split, lower = better)",
        "=" * 70,
        f"{'Classifier':<40} {'Log Loss':>10} {'± Std':>8}",
        "-" * 70,
    ]
    for _, row in cv_results.iterrows():
        lines.append(
            f"{row['classifier']:<40} {row['cv_log_loss_mean']:>10.4f} {row['cv_log_loss_std']:>8.4f}"
        )
    lines += [
        "",
        "=" * 70,
        f"SELECTED FEATURES FOR MODEL TRAINING (top {len(selected_features)}, non-null)",
        "=" * 70,
    ]
    for i, f in enumerate(selected_features, 1):
        lines.append(f"  {i:2d}. {f}")
    lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def _save_importance_plot(ranking: pd.DataFrame, top_n: int = 20) -> pathlib.Path:
    """Save a horizontal bar chart of top features to reports/feature_importance.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _REPORTS_DIR.mkdir(exist_ok=True)
    plot_path = _REPORTS_DIR / "feature_importance_univariate.png"

    top = ranking[ranking["null_rate"] < 100].head(top_n).copy()
    top = top.sort_values("combined")  # ascending for horizontal bar (bottom = best)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(top["feature"], top["combined"], color="#2563eb", alpha=0.85)
    ax.set_xlabel("Combined score (normalised F-stat + MI) / 2", fontsize=11)
    ax.set_title("Tier 1 Feature Importance — Univariate Ranking\n(ANOVA F-statistic + Mutual Information)", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars with combined score
    for bar, score in zip(bars, top["combined"]):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path


def run_analysis(csv_path: str = "data/processed/tier1_features.csv") -> list[str]:
    """Full feature analysis pipeline. Returns the selected feature list.

    Only touches training data — val and test are never seen here.
    """
    logger.info("Loading training split...")
    train, _, _ = load_splits(csv_path)
    X_train, y_train = xy(train)

    logger.info("Running univariate feature ranking...")
    ranking = _feature_ranking(X_train, y_train)

    logger.info("Running 5-fold cross-validated classifier comparison...")
    cv_results = _classifier_comparison(X_train, y_train)

    selected = _select_features(ranking, top_n=20)
    logger.info("Selected %d features for training.", len(selected))

    report_path = _save_report(ranking, cv_results, selected)
    plot_path   = _save_importance_plot(ranking)

    logger.info("Report saved: %s", report_path)
    logger.info("Plot saved:   %s", plot_path)

    print("\n" + "=" * 60)
    print("TOP 10 FEATURES (by combined F + MI score)")
    print("=" * 60)
    for rank, row in ranking[ranking["null_rate"] < 100].head(10).iterrows():
        print(f"  {rank:2d}. {row['feature']:<30}  score={row['combined']:.3f}  null={row['null_rate']:.1f}%")

    print("\nCV LOG LOSS COMPARISON (train fold, lower = better):")
    for _, row in cv_results.iterrows():
        print(f"  {row['classifier']:<40}  {row['cv_log_loss_mean']:.4f} ± {row['cv_log_loss_std']:.4f}")
    print()

    return selected


if __name__ == "__main__":
    run_analysis()
