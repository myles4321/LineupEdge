"""
Phase 7b — Stacking / Super Learner.

Builds a two-layer stacking ensemble on top of the 8 Phase 7a base models.

Layer 1 — OOF generation:
  Each base model generates out-of-fold predictions on train+val (760 rows)
  via 5-fold stratified CV, producing a 760×24 OOF matrix
  (8 models × 3 class probabilities per model).

Layer 2 — Meta-learner:
  A multinomial logistic regression is trained on the OOF matrix.
  Its C hyperparameter is tuned via inner 5-fold CV on the OOF matrix.

Final evaluation:
  All 8 base models are refit on full train+val; their test predictions are
  assembled into a 380×24 matrix and passed through the meta-learner.
  Test log loss is compared against the best individual model (LASSO, 0.9964)
  and bookmaker B365 (0.9708).

Usage:
    python models/stack.py
    python models/stack.py --no-db
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
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
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

_REPORTS_DIR  = pathlib.Path("reports")
_MODEL_VERSION = "tier7_stack_v1"
_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Reference metrics from Phase 7a for comparison table
_LASSO_LOG_LOSS      = 0.9964244324974212
_BOOKMAKER_LOG_LOSS  = 0.9707576661637081


# ── Pipeline factory helpers ──────────────────────────────────────────────────


def _impute_scale_clf(clf) -> Pipeline:
    """Median imputation + standard scaling pipeline."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("clf",    clf),
    ])


def _impute_clf(clf) -> Pipeline:
    """Median imputation only (no scaling — tree/LDA models)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf",    clf),
    ])


def _base_model_templates() -> list[tuple[str, Pipeline]]:
    """Return fresh (name, unfitted Pipeline) pairs for all 8 base models.

    Hyperparameters are the Phase 7a val-tuned best values.
    Called once to define templates; sklearn.base.clone() is used to get
    fresh copies for each CV fold, preventing state leakage between folds.
    """
    return [
        ("rf", _impute_clf(
            RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
        )),
        ("svm", _impute_scale_clf(
            SVC(C=0.1, gamma="scale", kernel="rbf",
                probability=True, random_state=42)
        )),
        ("knn", _impute_scale_clf(
            KNeighborsClassifier(n_neighbors=51, weights="distance")
        )),
        ("mlp", _impute_scale_clf(
            MLPClassifier(
                hidden_layer_sizes=(128, 64), activation="relu",
                solver="adam", alpha=1e-4, max_iter=500,
                early_stopping=True, validation_fraction=0.1,
                random_state=42,
            )
        )),
        ("ridge", _impute_scale_clf(
            LogisticRegression(
                penalty="l2", C=0.01, solver="lbfgs",
                multi_class="multinomial", max_iter=1000, random_state=42,
            )
        )),
        ("lasso", _impute_scale_clf(
            LogisticRegression(
                penalty="l1", C=0.1, solver="saga",
                multi_class="ovr", max_iter=1000, random_state=42,
            )
        )),
        ("enet", _impute_scale_clf(
            LogisticRegression(
                penalty="elasticnet", C=0.1, l1_ratio=0.9,
                solver="saga", multi_class="ovr",
                max_iter=1000, random_state=42,
            )
        )),
        ("lda", _impute_clf(
            LinearDiscriminantAnalysis(solver="svd")
        )),
    ]


# ── OOF generation ────────────────────────────────────────────────────────────


def generate_oof_matrix(
    templates: list[tuple[str, Pipeline]],
    X_full: pd.DataFrame,
    y_full_enc: np.ndarray,
) -> np.ndarray:
    """Generate out-of-fold predictions for all base models.

    Returns ndarray of shape (n_samples, n_models * 3).
    Column order: rf_H, rf_D, rf_A, svm_H, svm_D, svm_A, ...
    """
    n_models  = len(templates)
    n_samples = len(X_full)
    oof       = np.zeros((n_samples, n_models * 3))

    for model_idx, (model_name, pipeline_template) in enumerate(templates):
        print(f"Generating OOF for [{model_name}]...")
        col_start = model_idx * 3

        for fold_train_idx, fold_val_idx in _CV.split(X_full, y_full_enc):
            fold_pipeline = clone(pipeline_template)
            fold_pipeline.fit(
                X_full.iloc[fold_train_idx],
                y_full_enc[fold_train_idx],
            )
            fold_proba = fold_pipeline.predict_proba(X_full.iloc[fold_val_idx])
            oof[fold_val_idx, col_start : col_start + 3] = fold_proba

        model_ll = log_loss(y_full_enc, oof[:, col_start : col_start + 3],
                            labels=[0, 1, 2])
        logger.info("  OOF %-6s  log_loss=%.4f", model_name, model_ll)

    return oof


# ── Meta-learner ──────────────────────────────────────────────────────────────


def tune_meta_learner(
    oof_matrix: np.ndarray,
    y_full_enc: np.ndarray,
) -> tuple[float, float]:
    """Tune meta-learner L2 C via inner 5-fold CV on the OOF matrix.

    Returns (best_C, best_cv_log_loss).
    """
    best_c, best_ll = 1.0, float("inf")

    for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
        meta = LogisticRegression(
            penalty="l2", C=c, solver="lbfgs",
            multi_class="multinomial", max_iter=1000, random_state=42,
        )
        cv_scores = cross_val_score(
            meta, oof_matrix, y_full_enc,
            cv=_CV, scoring="neg_log_loss", n_jobs=-1,
        )
        ll = float(-cv_scores.mean())
        logger.info("  Meta-LR C=%-6s  OOF CV log_loss=%.4f", c, ll)
        if ll < best_ll:
            best_ll, best_c = ll, c

    logger.info("Best meta C=%.4f  OOF CV log_loss=%.4f", best_c, best_ll)
    return best_c, best_ll


def fit_meta_learner(
    oof_matrix: np.ndarray,
    y_full_enc: np.ndarray,
    best_c: float,
) -> LogisticRegression:
    """Fit the final meta-learner on the full OOF matrix."""
    meta = LogisticRegression(
        penalty="l2", C=best_c, solver="lbfgs",
        multi_class="multinomial", max_iter=1000, random_state=42,
    )
    meta.fit(oof_matrix, y_full_enc)
    return meta


# ── Base model refitting ──────────────────────────────────────────────────────


def fit_base_models(
    templates: list[tuple[str, Pipeline]],
    X_full: pd.DataFrame,
    y_full_enc: np.ndarray,
) -> list[tuple[str, Pipeline]]:
    """Refit each base model on the full train+val dataset."""
    fitted = []
    for model_name, pipeline_template in templates:
        logger.info("Fitting final base model [%s] on train+val...", model_name)
        pipeline = clone(pipeline_template)
        pipeline.fit(X_full, y_full_enc)
        fitted.append((model_name, pipeline))
    return fitted


def build_test_meta_features(
    fitted_models: list[tuple[str, Pipeline]],
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Assemble (n_test, n_models * 3) meta-feature matrix for the test set."""
    n_models = len(fitted_models)
    meta     = np.zeros((len(X_test), n_models * 3))

    for model_idx, (model_name, pipeline) in enumerate(fitted_models):
        col_start = model_idx * 3
        meta[:, col_start : col_start + 3] = pipeline.predict_proba(X_test)

    return meta


# ── Evaluation helpers ────────────────────────────────────────────────────────


def _multiclass_brier(y_true_enc: np.ndarray, proba: np.ndarray) -> float:
    scores = []
    for i in range(len(LABEL_ORDER)):
        y_binary = (y_true_enc == i).astype(float)
        scores.append(brier_score_loss(y_binary, proba[:, i]))
    return float(np.mean(scores))


def _accuracy(y_true_enc: np.ndarray, proba: np.ndarray) -> float:
    return float((np.argmax(proba, axis=1) == y_true_enc).mean())


# ── Meta-learner weight extraction ────────────────────────────────────────────


def extract_base_model_weights(
    meta_learner: LogisticRegression,
    model_names: list[str],
) -> dict:
    """Extract per-base-model weights from the meta-learner coefficient matrix.

    meta_learner.coef_ shape: (3, n_models * 3)
      rows    = output classes (H=0, D=1, A=2)
      columns = input features grouped as [model_H, model_D, model_A] per model

    Returns dict: {model_name: {"H": w, "D": w, "A": w, "mean_abs": w}}
    where H/D/A are the coefficients for that model's corresponding class probability
    in the final H/D/A output class respectively.
    """
    coef = meta_learner.coef_  # shape (3, n_models * 3)
    weights = {}

    for model_idx, model_name in enumerate(model_names):
        col_start = model_idx * 3
        # coef[output_class, input_col] — diagonal gives the "self" class weight
        w_h = float(coef[0, col_start])      # H-output weight on H-input
        w_d = float(coef[1, col_start + 1])  # D-output weight on D-input
        w_a = float(coef[2, col_start + 2])  # A-output weight on A-input
        mean_abs = float(np.abs(coef[:, col_start : col_start + 3]).mean())
        weights[model_name] = {
            "H": round(w_h, 4),
            "D": round(w_d, 4),
            "A": round(w_a, 4),
            "mean_abs_coef": round(mean_abs, 4),
        }

    return weights


# ── DB helpers ────────────────────────────────────────────────────────────────


def _log_predictions(
    engine,
    test_df: pd.DataFrame,
    proba: np.ndarray,
    model_version: str,
    tier: int = 7,
) -> int:
    """Delete existing rows for model_version and insert fresh predictions."""
    rows = [
        {
            "match_id":      int(row["match_id"]),
            "tier":          tier,
            "model_version": model_version,
            "prob_home":     float(proba[i, 0]),
            "prob_draw":     float(proba[i, 1]),
            "prob_away":     float(proba[i, 2]),
        }
        for i, row in test_df.iterrows()
    ]

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


# ── Output ────────────────────────────────────────────────────────────────────


def _print_comparison(stack_ll: float, stack_brier: float, stack_acc: float) -> None:
    print("\n" + "=" * 70)
    print("  PHASE 7b STACKING RESULTS — EPL 2024 TEST SEASON (380 matches)")
    print("=" * 70)
    print(f"  {'Model':<30} {'Log Loss':>10} {'Brier':>8} {'Acc%':>7}")
    print("  " + "-" * 60)

    delta_vs_lasso = stack_ll - _LASSO_LOG_LOSS
    delta_vs_bm    = stack_ll - _BOOKMAKER_LOG_LOSS

    print(f"  {'Super Learner (stack)':<30} {stack_ll:>10.4f} {stack_brier:>8.4f} {stack_acc*100:>7.2f}")
    print(f"  {'LASSO LogReg (best T7a)':<30} {_LASSO_LOG_LOSS:>10.4f} {'—':>8} {'—':>7}  "
          f"(Δ = {delta_vs_lasso:+.4f} vs stack)")
    print(f"  {'Bookmaker B365':<30} {_BOOKMAKER_LOG_LOSS:>10.4f} {'—':>8} {'—':>7}  "
          f"(Δ = {delta_vs_bm:+.4f} vs stack)")
    print("=" * 70 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7b: Stacking / Super Learner")
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

    # Drop 100%-null features (xG columns absent until stats ingestion)
    null_in_train = X_train.isna().mean()
    scorable_cols = [c for c in X_train.columns if null_in_train[c] < 1.0]
    dropped_cols  = [c for c in X_train.columns if null_in_train[c] >= 1.0]
    if dropped_cols:
        logger.info("Dropping 100%%-null features: %s", dropped_cols)

    X_train = X_train[scorable_cols].reset_index(drop=True)
    X_val   = X_val[scorable_cols].reset_index(drop=True)
    X_test  = X_test[scorable_cols].reset_index(drop=True)

    n_features = len(scorable_cols)

    y_train_enc = y_train.map(INT_LABEL_TO_IDX).values
    y_val_enc   = y_val.map(INT_LABEL_TO_IDX).values
    y_test_enc  = y_test.map(INT_LABEL_TO_IDX).values

    # Combined train+val for OOF generation and final base model fitting
    X_full     = pd.concat([X_train, X_val], ignore_index=True)
    y_full_enc = np.concatenate([y_train_enc, y_val_enc])

    test_reset = test.reset_index(drop=True)
    X_test     = X_test.reset_index(drop=True)

    logger.info(
        "Train: %d  Val: %d  Test: %d  Full: %d  Features: %d",
        len(X_train), len(X_val), len(X_test), len(X_full), n_features,
    )

    engine = create_engine(DATABASE_URL)

    # ── Define base model templates ───────────────────────────────────────────
    templates   = _base_model_templates()
    model_names = [name for name, _ in templates]
    n_models    = len(templates)

    logger.info("=== Phase 1: OOF generation (%d base models, 5-fold CV) ===", n_models)
    oof_matrix = generate_oof_matrix(templates, X_full, y_full_enc)
    logger.info("OOF matrix shape: %s", oof_matrix.shape)

    oof_ll = log_loss(y_full_enc, oof_matrix[:, :3], labels=[0, 1, 2])
    logger.info("OOF matrix first-model (rf) log_loss sanity: %.4f", oof_ll)

    # ── Tune meta-learner ─────────────────────────────────────────────────────
    logger.info("=== Phase 2: Meta-learner C tuning (inner 5-fold CV on OOF) ===")
    best_meta_c, best_meta_ll = tune_meta_learner(oof_matrix, y_full_enc)

    # ── Fit final meta-learner on full OOF ───────────────────────────────────
    logger.info("=== Phase 3: Fitting meta-learner (C=%.4f) on full OOF ===", best_meta_c)
    meta_learner = fit_meta_learner(oof_matrix, y_full_enc, best_meta_c)

    # ── Refit base models on full train+val ───────────────────────────────────
    logger.info("=== Phase 4: Refitting base models on full train+val ===")
    fitted_models = fit_base_models(templates, X_full, y_full_enc)

    # ── Assemble test meta-features ───────────────────────────────────────────
    logger.info("=== Phase 5: Assembling test meta-features ===")
    test_meta_features = build_test_meta_features(fitted_models, X_test)
    stack_proba        = meta_learner.predict_proba(test_meta_features)

    logger.info("Stack proba shape: %s  (should be 380×3)", stack_proba.shape)

    # ── Evaluate stack on test set ────────────────────────────────────────────
    logger.info("=== Phase 6: Evaluating stack on 2024 test set ===")
    stack_ll    = float(log_loss(y_test_enc, stack_proba, labels=[0, 1, 2]))
    stack_brier = _multiclass_brier(y_test_enc, stack_proba)
    stack_acc   = _accuracy(y_test_enc, stack_proba)
    stack_auc   = float(roc_auc_score(y_test_enc, stack_proba,
                                      multi_class="ovr", average="macro",
                                      labels=[0, 1, 2]))

    logger.info(
        "Stack: LL=%.4f  Brier=%.4f  Acc=%.3f  AUC=%.4f",
        stack_ll, stack_brier, stack_acc, stack_auc,
    )

    # ── Bookmaker benchmark ───────────────────────────────────────────────────
    bm_raw = _load_bookmaker(test["match_id"], engine)
    bm_ll  = None
    if bm_raw is not None:
        test_bm  = test.merge(bm_raw, on="match_id", how="left")
        bm_mask  = test_bm["bm_home"].notna()
        bm_sub   = test_bm[bm_mask]
        bm_proba = bm_sub[["bm_home", "bm_draw", "bm_away"]].values.astype(float)
        y_bm_enc = bm_sub["result"].map(LABEL_TO_IDX).values
        bm_ll    = float(log_loss(y_bm_enc, bm_proba, labels=[0, 1, 2]))
        logger.info("Bookmaker B365 reference LL=%.4f", bm_ll)

    # ── Extract base model weights ────────────────────────────────────────────
    weights = extract_base_model_weights(meta_learner, model_names)
    logger.info("Meta-learner weights (mean |coef|):")
    for name, w in sorted(weights.items(), key=lambda x: -x[1]["mean_abs_coef"]):
        logger.info("  %-8s  mean|coef|=%.4f  (H=%.3f  D=%.3f  A=%.3f)",
                    name, w["mean_abs_coef"], w["H"], w["D"], w["A"])

    # ── Print comparison table ────────────────────────────────────────────────
    _print_comparison(stack_ll, stack_brier, stack_acc)

    # ── Log predictions to DB ─────────────────────────────────────────────────
    if not args.no_db:
        logger.info("=== Phase 7: Logging stack predictions to DB ===")
        n_logged = _log_predictions(engine, test_reset, stack_proba, _MODEL_VERSION)
        logger.info("Logged %d predictions for %s", n_logged, _MODEL_VERSION)

    # ── Write JSON ────────────────────────────────────────────────────────────
    output = {
        "split":         {"train": 2022, "val": 2023, "test": 2024},
        "n_features":    n_features,
        "n_base_models": n_models,
        "base_models":   model_names,
        "meta_learner": {
            "best_C":       best_meta_c,
            "cv_log_loss":  round(best_meta_ll, 4),
        },
        "stack": {
            "model_version": _MODEL_VERSION,
            "log_loss":      round(stack_ll, 4),
            "brier_score":   round(stack_brier, 4),
            "accuracy":      round(stack_acc, 4),
            "roc_auc_macro": round(stack_auc, 4),
        },
        "base_model_weights": weights,
        "comparison": {
            "lasso_log_loss":          round(_LASSO_LOG_LOSS, 4),
            "bookmaker_log_loss":      round(_BOOKMAKER_LOG_LOSS, 4),
            "stack_delta_vs_lasso":    round(stack_ll - _LASSO_LOG_LOSS, 4),
            "stack_delta_vs_bookmaker": round(stack_ll - _BOOKMAKER_LOG_LOSS, 4),
        },
    }

    json_path = _REPORTS_DIR / "stack_results.json"
    json_path.write_text(json.dumps(output, indent=2))
    logger.info("Stack results written to %s", json_path)
    logger.info("Phase 7b complete.")


if __name__ == "__main__":
    main()
