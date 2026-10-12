"""
Phase 7a — Full Model Tournament.

Trains and evaluates 9 classifiers (+QDA) on the Tier 1 feature matrix.
Each model is hyperparameter-tuned on the 2023 validation season, retrained
on the full train+val set (2022+2023), then evaluated on the 2024 test season.

5-fold stratified CV is run on train+val to measure generalisation variance.
All test predictions are logged to the database with tier=7.

Results are written to reports/tournament_results.json and a ranked
tournament table is printed to stdout.

Usage:
    python models/tournament.py
    python models/tournament.py --no-db
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sqlalchemy import create_engine, text

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from config import DATABASE_URL
from models.split import (
    INT_LABEL_TO_IDX,
    LABEL_ORDER,
    LABEL_TO_IDX,
    load_splits,
    xy,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPORTS_DIR = pathlib.Path("reports")
_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ── Evaluation helpers ────────────────────────────────────────────────────────


def _multiclass_brier(y_true_enc: np.ndarray, proba: np.ndarray) -> float:
    """Mean one-vs-rest Brier score across all three outcome classes."""
    scores = []
    for i in range(len(LABEL_ORDER)):
        y_binary = (y_true_enc == i).astype(float)
        scores.append(brier_score_loss(y_binary, proba[:, i]))
    return float(np.mean(scores))


def _per_class_log_loss(y_true_enc: np.ndarray, proba: np.ndarray) -> dict:
    """One-vs-rest log loss for each class {H, D, A}."""
    result = {}
    for i, label in enumerate(LABEL_ORDER):
        y_binary = (y_true_enc == i).astype(int)
        result[label] = float(log_loss(y_binary, proba[:, i]))
    return result


def _accuracy(y_true_enc: np.ndarray, proba: np.ndarray) -> float:
    y_pred = np.argmax(proba, axis=1)
    return float((y_pred == y_true_enc).mean())


def _evaluate(
    model_name: str,
    pipeline: Pipeline,
    X_full: pd.DataFrame,
    y_full_enc: np.ndarray,
    X_test: pd.DataFrame,
    y_test_enc: np.ndarray,
    best_params: dict,
) -> dict:
    """Compute all test metrics + 5-fold CV on train+val. Returns results dict."""
    proba = pipeline.predict_proba(X_test)

    ll       = float(log_loss(y_test_enc, proba, labels=[0, 1, 2]))
    brier    = _multiclass_brier(y_test_enc, proba)
    acc      = _accuracy(y_test_enc, proba)
    per_cls  = _per_class_log_loss(y_test_enc, proba)
    roc_auc  = float(roc_auc_score(y_test_enc, proba,
                                   multi_class="ovr", average="macro",
                                   labels=[0, 1, 2]))

    cv_scores       = cross_val_score(pipeline, X_full, y_full_enc,
                                      cv=_CV, scoring="neg_log_loss", n_jobs=-1)
    cv_mean_ll      = float(-cv_scores.mean())
    cv_std_ll       = float(cv_scores.std())

    logger.info(
        "  %-22s  LL=%.4f  Brier=%.4f  Acc=%.3f  AUC=%.4f  CV=%.4f±%.4f",
        model_name, ll, brier, acc, roc_auc, cv_mean_ll, cv_std_ll,
    )

    return {
        "log_loss":         ll,
        "brier_score":      brier,
        "accuracy":         acc,
        "roc_auc_macro":    roc_auc,
        "cv_mean_log_loss": cv_mean_ll,
        "cv_std_log_loss":  cv_std_ll,
        "per_class_log_loss": per_cls,
        "best_params":      best_params,
    }


# ── Database helpers ──────────────────────────────────────────────────────────


def _log_predictions(
    engine,
    test_df: pd.DataFrame,
    proba: np.ndarray,
    model_version: str,
    tier: int = 7,
) -> int:
    """Delete existing rows for model_version and bulk-insert fresh predictions.

    proba columns must be ordered [P(H), P(D), P(A)] — i.e. indices 0, 1, 2.
    """
    rows = []
    for i, row in test_df.iterrows():
        rows.append({
            "match_id":      int(row["match_id"]),
            "tier":          tier,
            "model_version": model_version,
            "prob_home":     float(proba[i, 0]),
            "prob_draw":     float(proba[i, 1]),
            "prob_away":     float(proba[i, 2]),
        })

    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM predictions WHERE model_version = :mv AND tier = :tier"),
            {"mv": model_version, "tier": tier},
        )
        conn.execute(
            text("""
                INSERT INTO predictions
                    (match_id, tier, model_version, prob_home, prob_draw, prob_away)
                VALUES
                    (:match_id, :tier, :model_version, :prob_home, :prob_draw, :prob_away)
            """),
            rows,
        )
        conn.commit()
    return len(rows)


def _load_bookmaker(match_ids: pd.Series, engine) -> pd.DataFrame | None:
    """Load B365 implied probabilities for test match IDs."""
    ids = tuple(int(m) for m in match_ids)
    if not ids:
        return None
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT match_id, implied_home, implied_draw, implied_away
                FROM odds
                WHERE match_id IN :ids AND bookmaker = 'B365'
            """),
            {"ids": ids},
        ).fetchall()
    if not result:
        return None
    return pd.DataFrame(result, columns=["match_id", "bm_home", "bm_draw", "bm_away"])


# ── Model implementations ─────────────────────────────────────────────────────


def _build_impute_scale_clf(clf) -> Pipeline:
    """StandardScaler + SimpleImputer pipeline around a classifier."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    clf),
    ])


def _build_impute_clf(clf) -> Pipeline:
    """SimpleImputer-only pipeline (no scaling needed for tree/NB models)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    clf),
    ])


def train_random_forest(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict]:
    """Random Forest — tune n_estimators by OOB error on train set."""
    print("Training [random_forest]...")

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    best_n, best_oob = 500, -1.0
    for n_est in [100, 200, 300, 400, 500]:
        rf = RandomForestClassifier(
            n_estimators=n_est,
            max_features="sqrt",
            class_weight="balanced",
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train_imp, y_train_enc)
        logger.info("  RF n_est=%d  oob_score=%.4f", n_est, rf.oob_score_)
        if rf.oob_score_ > best_oob:
            best_oob = rf.oob_score_
            best_n = n_est

    logger.info("Best RF n_estimators=%d  oob_score=%.4f", best_n, best_oob)

    final_clf = RandomForestClassifier(
        n_estimators=best_n,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=False,
        random_state=42,
        n_jobs=-1,
    )
    pipeline = _build_impute_clf(final_clf)
    pipeline.fit(X_full, y_full_enc)

    best_params = {"n_estimators": best_n, "max_features": "sqrt",
                   "class_weight": "balanced", "oob_score": best_oob}
    return pipeline, best_params


def train_svm(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict]:
    """SVM RBF — grid search C × gamma on validation log loss."""
    print("Training [svm]...")

    best_params_inner = {"C": 1.0, "gamma": "scale"}
    best_loss = float("inf")

    for c in [0.1, 1.0, 10.0]:
        for gamma in ["scale", "auto"]:
            pipe = _build_impute_scale_clf(
                SVC(C=c, gamma=gamma, kernel="rbf",
                    probability=True, random_state=42)
            )
            pipe.fit(X_train, y_train_enc)
            proba = pipe.predict_proba(X_val)
            ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
            logger.info("  SVM C=%-4s gamma=%-6s  val LL=%.4f", c, gamma, ll)
            if ll < best_loss:
                best_loss = ll
                best_params_inner = {"C": c, "gamma": gamma}

    logger.info("Best SVM params=%s  val LL=%.4f", best_params_inner, best_loss)

    final_clf = SVC(
        C=best_params_inner["C"],
        gamma=best_params_inner["gamma"],
        kernel="rbf",
        probability=True,
        random_state=42,
    )
    pipeline = _build_impute_scale_clf(final_clf)
    pipeline.fit(X_full, y_full_enc)
    return pipeline, best_params_inner


def train_knn(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict]:
    """KNN — tune k on validation log loss, StandardScaled features."""
    print("Training [knn]...")

    best_k, best_loss = 5, float("inf")

    for k in [5, 11, 21, 51]:
        pipe = _build_impute_scale_clf(
            KNeighborsClassifier(n_neighbors=k, weights="distance")
        )
        pipe.fit(X_train, y_train_enc)
        proba = pipe.predict_proba(X_val)
        ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
        logger.info("  KNN k=%-2d  val LL=%.4f", k, ll)
        if ll < best_loss:
            best_loss, best_k = ll, k

    logger.info("Best KNN k=%d  val LL=%.4f", best_k, best_loss)

    final_clf = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
    pipeline = _build_impute_scale_clf(final_clf)
    pipeline.fit(X_full, y_full_enc)
    return pipeline, {"n_neighbors": best_k, "weights": "distance"}


def train_gaussian_nb(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict]:
    """Gaussian Naive Bayes — tune var_smoothing on val log loss."""
    print("Training [gaussian_nb]...")

    best_vs, best_loss = 1e-9, float("inf")

    for vs in [1e-9, 1e-7, 1e-5]:
        pipe = _build_impute_clf(GaussianNB(var_smoothing=vs))
        pipe.fit(X_train, y_train_enc)
        proba = pipe.predict_proba(X_val)
        ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
        logger.info("  GNB var_smoothing=%.1e  val LL=%.4f", vs, ll)
        if ll < best_loss:
            best_loss, best_vs = ll, vs

    logger.info("Best GNB var_smoothing=%.1e  val LL=%.4f", best_vs, best_loss)

    pipeline = _build_impute_clf(GaussianNB(var_smoothing=best_vs))
    pipeline.fit(X_full, y_full_enc)
    return pipeline, {"var_smoothing": best_vs}


def train_mlp(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict]:
    """MLP — tune L2 regularisation alpha on val log loss."""
    print("Training [mlp]...")

    best_alpha, best_loss = 1e-4, float("inf")

    for alpha in [1e-5, 1e-4, 1e-3, 1e-2]:
        pipe = _build_impute_scale_clf(
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                alpha=alpha,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )
        )
        pipe.fit(X_train, y_train_enc)
        proba = pipe.predict_proba(X_val)
        ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
        logger.info("  MLP alpha=%.1e  val LL=%.4f", alpha, ll)
        if ll < best_loss:
            best_loss, best_alpha = ll, alpha

    logger.info("Best MLP alpha=%.1e  val LL=%.4f", best_alpha, best_loss)

    final_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=best_alpha,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    pipeline = _build_impute_scale_clf(final_clf)
    pipeline.fit(X_full, y_full_enc)
    return pipeline, {"alpha": best_alpha, "hidden_layer_sizes": (128, 64)}


def train_ridge_logreg(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict]:
    """Logistic Regression L2 (Ridge) — tune C on val log loss."""
    print("Training [ridge_logreg]...")

    best_c, best_loss = 1.0, float("inf")

    for c in [0.001, 0.01, 0.1, 1.0]:
        pipe = _build_impute_scale_clf(
            LogisticRegression(
                penalty="l2", C=c, solver="lbfgs",
                multi_class="multinomial", max_iter=1000,
                random_state=42,
            )
        )
        pipe.fit(X_train, y_train_enc)
        proba = pipe.predict_proba(X_val)
        ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
        logger.info("  Ridge LR C=%-5s  val LL=%.4f", c, ll)
        if ll < best_loss:
            best_loss, best_c = ll, c

    logger.info("Best Ridge LR C=%.4f  val LL=%.4f", best_c, best_loss)

    final_clf = LogisticRegression(
        penalty="l2", C=best_c, solver="lbfgs",
        multi_class="multinomial", max_iter=1000,
        random_state=42,
    )
    pipeline = _build_impute_scale_clf(final_clf)
    pipeline.fit(X_full, y_full_enc)
    return pipeline, {"C": best_c, "penalty": "l2"}


def train_lasso_logreg(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
    feature_names: list,
) -> tuple[Pipeline, dict, list]:
    """Logistic Regression L1 (LASSO) — tune C on val. Returns zero features."""
    print("Training [lasso_logreg]...")

    best_c, best_loss = 0.1, float("inf")

    for c in [0.001, 0.01, 0.1, 1.0]:
        pipe = _build_impute_scale_clf(
            LogisticRegression(
                penalty="l1", C=c, solver="saga",
                multi_class="ovr", max_iter=1000,
                random_state=42,
            )
        )
        pipe.fit(X_train, y_train_enc)
        proba = pipe.predict_proba(X_val)
        ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
        logger.info("  LASSO LR C=%-5s  val LL=%.4f", c, ll)
        if ll < best_loss:
            best_loss, best_c = ll, c

    logger.info("Best LASSO LR C=%.4f  val LL=%.4f", best_c, best_loss)

    final_clf = LogisticRegression(
        penalty="l1", C=best_c, solver="saga",
        multi_class="ovr", max_iter=1000,
        random_state=42,
    )
    pipeline = _build_impute_scale_clf(final_clf)
    pipeline.fit(X_full, y_full_enc)

    # Identify features zeroed out by L1 across ALL 3 OVR classifiers
    # coef_ shape: (n_classes, n_features)
    lasso_coef = pipeline["clf"].coef_
    zero_mask = np.all(np.abs(lasso_coef) < 1e-10, axis=0)
    zero_features = [feature_names[i] for i, z in enumerate(zero_mask) if z]
    n_zero = len(zero_features)
    logger.info("LASSO zeroed %d/%d features: %s", n_zero, len(feature_names),
                zero_features if n_zero <= 10 else zero_features[:10])

    return pipeline, {"C": best_c, "penalty": "l1"}, zero_features


def train_elastic_net(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict]:
    """Logistic Regression Elastic Net — tune C and l1_ratio on val."""
    print("Training [elastic_net]...")

    best_params_inner = {"C": 0.1, "l1_ratio": 0.5}
    best_loss = float("inf")

    for c in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            pipe = _build_impute_scale_clf(
                LogisticRegression(
                    penalty="elasticnet", C=c, l1_ratio=l1_ratio,
                    solver="saga", multi_class="ovr",
                    max_iter=1000, random_state=42,
                )
            )
            pipe.fit(X_train, y_train_enc)
            proba = pipe.predict_proba(X_val)
            ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
            logger.info("  ENet C=%-4s l1_ratio=%.1f  val LL=%.4f", c, l1_ratio, ll)
            if ll < best_loss:
                best_loss = ll
                best_params_inner = {"C": c, "l1_ratio": l1_ratio}

    logger.info("Best ENet params=%s  val LL=%.4f", best_params_inner, best_loss)

    final_clf = LogisticRegression(
        penalty="elasticnet",
        C=best_params_inner["C"],
        l1_ratio=best_params_inner["l1_ratio"],
        solver="saga",
        multi_class="ovr",
        max_iter=1000,
        random_state=42,
    )
    pipeline = _build_impute_scale_clf(final_clf)
    pipeline.fit(X_full, y_full_enc)
    return pipeline, {**best_params_inner, "penalty": "elasticnet"}


def train_lda_qda(
    X_train: pd.DataFrame, y_train_enc: np.ndarray,
    X_val: pd.DataFrame,   y_val_enc: np.ndarray,
    X_full: pd.DataFrame,  y_full_enc: np.ndarray,
    X_test: pd.DataFrame,  y_test_enc: np.ndarray,
) -> tuple[Pipeline, dict, Pipeline | None, dict | None]:
    """LDA (solver=svd) + QDA (tune reg_param on val). Returns (lda_pipe, lda_params, qda_pipe, qda_params)."""
    print("Training [lda]...")

    # LDA — no hyperparameter tuning needed
    lda_pipeline = _build_impute_clf(LinearDiscriminantAnalysis(solver="svd"))
    lda_pipeline.fit(X_full, y_full_enc)
    lda_params = {"solver": "svd"}

    print("Training [qda]...")

    # QDA — tune reg_param on val
    best_reg, best_loss = 0.0, float("inf")

    for reg in [0.0, 0.001, 0.01, 0.1]:
        try:
            pipe = _build_impute_clf(QuadraticDiscriminantAnalysis(reg_param=reg))
            pipe.fit(X_train, y_train_enc)
            proba = pipe.predict_proba(X_val)
            ll = log_loss(y_val_enc, proba, labels=[0, 1, 2])
            logger.info("  QDA reg_param=%.3f  val LL=%.4f", reg, ll)
            if ll < best_loss:
                best_loss, best_reg = ll, reg
        except Exception as exc:
            logger.warning("  QDA reg_param=%.3f failed: %s", reg, exc)

    if best_loss == float("inf"):
        logger.error("QDA failed for all reg_param values — skipping QDA.")
        return lda_pipeline, lda_params, None, None

    logger.info("Best QDA reg_param=%.4f  val LL=%.4f", best_reg, best_loss)

    try:
        qda_pipeline = _build_impute_clf(
            QuadraticDiscriminantAnalysis(reg_param=best_reg)
        )
        qda_pipeline.fit(X_full, y_full_enc)
        qda_params = {"reg_param": best_reg}
        return lda_pipeline, lda_params, qda_pipeline, qda_params
    except Exception as exc:
        logger.error("QDA final fit failed: %s — skipping QDA.", exc)
        return lda_pipeline, lda_params, None, None


# ── Tournament table printing ─────────────────────────────────────────────────


def _print_tournament(results: list[dict], bookmaker: dict | None) -> None:
    print("\n" + "=" * 100)
    print("  PHASE 7a MODEL TOURNAMENT — EPL 2024 TEST SEASON (380 matches)")
    print("=" * 100)
    print(f"  {'Rank':<5} {'Model':<28} {'Log Loss':>9} {'Brier':>8} {'Acc%':>7} "
          f"{'AUC':>7} {'CV LL ± σ':>16} {'Δ vs BM':>10}")
    print("  " + "-" * 96)

    sorted_results = sorted(results, key=lambda r: r["log_loss"])
    bm_ll = bookmaker["log_loss"] if bookmaker else None

    for rank, r in enumerate(sorted_results, start=1):
        delta = f"{r['log_loss'] - bm_ll:+.4f}" if bm_ll is not None else "—"
        cv    = f"{r['cv_mean_log_loss']:.4f} ± {r['cv_std_log_loss']:.4f}"
        print(f"  {rank:<5} {r['model']:<28} {r['log_loss']:>9.4f} "
              f"{r['brier_score']:>8.4f} {r['accuracy']*100:>7.2f} "
              f"{r['roc_auc_macro']:>7.4f} {cv:>16} {delta:>10}")

    if bookmaker:
        print("  " + "-" * 96)
        bm_delta = "0.0000"
        bm_cv    = "—"
        bm_acc   = bookmaker.get("accuracy", float("nan"))
        bm_auc   = bookmaker.get("roc_auc_macro", float("nan"))
        print(f"  {'—':<5} {'Bookmaker B365':<28} {bookmaker['log_loss']:>9.4f} "
              f"{bookmaker['brier_score']:>8.4f} {bm_acc*100:>7.2f} "
              f"{bm_auc:>7.4f} {bm_cv:>16} {bm_delta:>10}")

    print("=" * 100 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7a: Full model tournament")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip writing predictions to database")
    args = parser.parse_args()

    _REPORTS_DIR.mkdir(exist_ok=True)

    # ── Load splits ───────────────────────────────────────────────────────────
    logger.info("Loading train/val/test splits...")
    train, val, test = load_splits()

    X_train, y_train = xy(train)
    X_val,   y_val   = xy(val)
    X_test,  y_test  = xy(test)

    # Drop 100%-null features (e.g. xG columns until stats ingestion completes)
    null_in_train = X_train.isna().mean()
    scorable_cols = [c for c in X_train.columns if null_in_train[c] < 1.0]
    dropped_cols  = [c for c in X_train.columns if null_in_train[c] >= 1.0]
    if dropped_cols:
        logger.info("Dropping 100%%-null features: %s", dropped_cols)

    X_train = X_train[scorable_cols].reset_index(drop=True)
    X_val   = X_val[scorable_cols].reset_index(drop=True)
    X_test  = X_test[scorable_cols].reset_index(drop=True)

    feature_names = list(scorable_cols)
    n_features    = len(feature_names)

    # Encode integer labels to class indices 0/1/2
    y_train_enc = y_train.map(INT_LABEL_TO_IDX).values
    y_val_enc   = y_val.map(INT_LABEL_TO_IDX).values
    y_test_enc  = y_test.map(INT_LABEL_TO_IDX).values

    # Combined train+val for final model fitting and CV
    X_full     = pd.concat([X_train, X_val], ignore_index=True)
    y_full_enc = np.concatenate([y_train_enc, y_val_enc])

    test_reset = test.reset_index(drop=True)

    logger.info(
        "Train: %d  Val: %d  Test: %d  Full: %d  Features: %d",
        len(X_train), len(X_val), len(X_test), len(X_full), n_features,
    )

    engine = create_engine(DATABASE_URL)

    # ── Run all models ────────────────────────────────────────────────────────

    results = []
    lasso_zero_features = []

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf_pipe, rf_params = train_random_forest(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)
    rf_metrics = _evaluate("random_forest", rf_pipe, X_full, y_full_enc,
                           X_test, y_test_enc, rf_params)
    results.append({"model": "tier7_rf_v1", "model_version": "tier7_rf_v1", **rf_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, rf_pipe.predict_proba(X_test), "tier7_rf_v1")
        logger.info("Logged %d predictions for tier7_rf_v1", n)

    # ── SVM ───────────────────────────────────────────────────────────────────
    svm_pipe, svm_params = train_svm(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)
    svm_metrics = _evaluate("svm", svm_pipe, X_full, y_full_enc,
                            X_test, y_test_enc, svm_params)
    results.append({"model": "tier7_svm_v1", "model_version": "tier7_svm_v1", **svm_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, svm_pipe.predict_proba(X_test), "tier7_svm_v1")
        logger.info("Logged %d predictions for tier7_svm_v1", n)

    # ── KNN ───────────────────────────────────────────────────────────────────
    knn_pipe, knn_params = train_knn(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)
    knn_metrics = _evaluate("knn", knn_pipe, X_full, y_full_enc,
                            X_test, y_test_enc, knn_params)
    results.append({"model": "tier7_knn_v1", "model_version": "tier7_knn_v1", **knn_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, knn_pipe.predict_proba(X_test), "tier7_knn_v1")
        logger.info("Logged %d predictions for tier7_knn_v1", n)

    # ── Gaussian NB ───────────────────────────────────────────────────────────
    gnb_pipe, gnb_params = train_gaussian_nb(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)
    gnb_metrics = _evaluate("gaussian_nb", gnb_pipe, X_full, y_full_enc,
                            X_test, y_test_enc, gnb_params)
    results.append({"model": "tier7_gnb_v1", "model_version": "tier7_gnb_v1", **gnb_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, gnb_pipe.predict_proba(X_test), "tier7_gnb_v1")
        logger.info("Logged %d predictions for tier7_gnb_v1", n)

    # ── MLP ───────────────────────────────────────────────────────────────────
    mlp_pipe, mlp_params = train_mlp(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)
    mlp_metrics = _evaluate("mlp", mlp_pipe, X_full, y_full_enc,
                            X_test, y_test_enc, mlp_params)
    results.append({"model": "tier7_mlp_v1", "model_version": "tier7_mlp_v1", **mlp_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, mlp_pipe.predict_proba(X_test), "tier7_mlp_v1")
        logger.info("Logged %d predictions for tier7_mlp_v1", n)

    # ── Ridge Logistic Regression ─────────────────────────────────────────────
    ridge_pipe, ridge_params = train_ridge_logreg(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)
    ridge_metrics = _evaluate("ridge_logreg", ridge_pipe, X_full, y_full_enc,
                              X_test, y_test_enc, ridge_params)
    results.append({"model": "tier7_ridge_v1", "model_version": "tier7_ridge_v1", **ridge_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, ridge_pipe.predict_proba(X_test), "tier7_ridge_v1")
        logger.info("Logged %d predictions for tier7_ridge_v1", n)

    # ── LASSO Logistic Regression ─────────────────────────────────────────────
    lasso_pipe, lasso_params, lasso_zero_features = train_lasso_logreg(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc,
        feature_names)
    lasso_metrics = _evaluate("lasso_logreg", lasso_pipe, X_full, y_full_enc,
                              X_test, y_test_enc, lasso_params)
    results.append({"model": "tier7_lasso_v1", "model_version": "tier7_lasso_v1", **lasso_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, lasso_pipe.predict_proba(X_test), "tier7_lasso_v1")
        logger.info("Logged %d predictions for tier7_lasso_v1", n)

    # ── Elastic Net ───────────────────────────────────────────────────────────
    enet_pipe, enet_params = train_elastic_net(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)
    enet_metrics = _evaluate("elastic_net", enet_pipe, X_full, y_full_enc,
                             X_test, y_test_enc, enet_params)
    results.append({"model": "tier7_enet_v1", "model_version": "tier7_enet_v1", **enet_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, enet_pipe.predict_proba(X_test), "tier7_enet_v1")
        logger.info("Logged %d predictions for tier7_enet_v1", n)

    # ── LDA + QDA ─────────────────────────────────────────────────────────────
    lda_pipe, lda_params, qda_pipe, qda_params = train_lda_qda(
        X_train, y_train_enc, X_val, y_val_enc, X_full, y_full_enc, X_test, y_test_enc)

    lda_metrics = _evaluate("lda", lda_pipe, X_full, y_full_enc,
                            X_test, y_test_enc, lda_params)
    results.append({"model": "tier7_lda_v1", "model_version": "tier7_lda_v1", **lda_metrics})
    if not args.no_db:
        n = _log_predictions(engine, test_reset, lda_pipe.predict_proba(X_test), "tier7_lda_v1")
        logger.info("Logged %d predictions for tier7_lda_v1", n)

    if qda_pipe is not None:
        qda_metrics = _evaluate("qda", qda_pipe, X_full, y_full_enc,
                                X_test, y_test_enc, qda_params)
        results.append({"model": "tier7_qda_v1", "model_version": "tier7_qda_v1", **qda_metrics})
        if not args.no_db:
            n = _log_predictions(engine, test_reset, qda_pipe.predict_proba(X_test), "tier7_qda_v1")
            logger.info("Logged %d predictions for tier7_qda_v1", n)
    else:
        logger.warning("QDA training failed — omitted from tournament results.")

    # ── Bookmaker benchmark ───────────────────────────────────────────────────
    logger.info("Loading bookmaker B365 benchmark...")
    bm_raw = _load_bookmaker(test["match_id"], engine)
    bookmaker_entry = None

    if bm_raw is not None:
        test_bm = test.merge(bm_raw, on="match_id", how="left")
        bm_mask = test_bm["bm_home"].notna()
        bm_sub  = test_bm[bm_mask]
        bm_proba = bm_sub[["bm_home", "bm_draw", "bm_away"]].values.astype(float)
        y_bm_enc = bm_sub["result"].map(LABEL_TO_IDX).values

        bm_ll    = float(log_loss(y_bm_enc, bm_proba, labels=[0, 1, 2]))
        bm_brier = _multiclass_brier(y_bm_enc, bm_proba)
        bm_acc   = _accuracy(y_bm_enc, bm_proba)
        bm_auc   = float(roc_auc_score(y_bm_enc, bm_proba,
                                       multi_class="ovr", average="macro",
                                       labels=[0, 1, 2]))
        bookmaker_entry = {
            "log_loss":    bm_ll,
            "brier_score": bm_brier,
            "accuracy":    bm_acc,
            "roc_auc_macro": bm_auc,
            "n_matches":   int(bm_mask.sum()),
        }
        logger.info(
            "Bookmaker B365: LL=%.4f  Brier=%.4f  Acc=%.3f  (n=%d)",
            bm_ll, bm_brier, bm_acc, bm_mask.sum(),
        )
    else:
        logger.warning("No bookmaker odds found — omitting benchmark.")

    # ── Print tournament table ────────────────────────────────────────────────
    _print_tournament(results, bookmaker_entry)

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "split":    {"train": 2022, "val": 2023, "test": 2024},
        "n_features": n_features,
        "results":  results,
        "bookmaker": bookmaker_entry,
        "lasso_zero_features": lasso_zero_features,
    }
    json_path = _REPORTS_DIR / "tournament_results.json"
    json_path.write_text(json.dumps(output, indent=2))
    logger.info("Tournament results written to %s", json_path)
    logger.info("Phase 7a complete.")


if __name__ == "__main__":
    main()
