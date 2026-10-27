# Quality Report: Phase 7a & 7b — Model Tournament + Stacking

**Date:** 2026-10-12
**Reviewer:** Claude Code QA
**Files reviewed:** `models/tournament.py`, `models/stack.py`
**Test season:** EPL 2024 (380 matches)

---

## Build Status

| Check | Status |
|-------|--------|
| `py_compile models/tournament.py` | PASS |
| `py_compile models/stack.py` | PASS |
| `python models/tournament.py --no-db` | PASS |
| `python models/stack.py --no-db` | PASS |

---

## Code Quality Score: 9/10

---

## Critical Correctness Checks

### 1. Data Leakage — StandardScaler / SimpleImputer

| Check | Result |
|-------|--------|
| Scaler fit only on fold-train data | **PASS** — all models use `Pipeline([impute, scale, clf])`, sklearn fits each step on fold-train only |
| Imputer fit only on fold-train data | **PASS** — same pipeline guarantee |
| RF OOB tuning — imputer fit on X_train only | **PASS** — `imputer.fit_transform(X_train)` for OOB search; final pipeline refits cleanly on X_full |
| X_val / X_test never passed to `.fit()` | **PASS** — only ever passed to `.predict_proba()` |

### 2. OOF Correctness (stack.py)

| Check | Result |
|-------|--------|
| No row overlap across folds | **PASS** — `StratifiedKFold` produces disjoint val indices covering all 760 rows exactly once |
| OOF matrix shape (760×24) | **PASS** — 8 models × 3 classes = 24 columns |
| `sklearn.base.clone()` used per fold | **PASS** — fresh pipeline copy prevents state leakage between folds |
| OOF index alignment | **PASS** — `oof_matrix[val_idx, col_start:col_end] = proba` uses the exact fold val indices |

### 3. Meta-Learner Correctness

| Check | Result |
|-------|--------|
| Meta-learner fit on OOF (not test) | **PASS** — `meta_learner.fit(oof_matrix, y_full_enc)` |
| Test meta-features from models fit on full train+val | **PASS** — `fit_base_models()` refits on `X_full`; `build_test_meta_features()` uses those final models |
| No meta-learner C tuning leakage | **PASS** — inner CV over C ∈ {0.001, 0.01, 0.1, 1.0, 10.0} uses `cross_val_score` on OOF matrix (no test involvement) |

### 4. LASSO Zero-Feature Detection

| Check | Result |
|-------|--------|
| `coef_` shape | **PASS** — `(3, 34)` for OVR, one row per class |
| Zero-mask logic | **PASS** — `np.all(np.abs(coef_) < 1e-10, axis=0)` reduces over rows → `(34,)` per-feature mask |
| Threshold appropriateness | **PASS** — 1e-10 is well below any meaningful coefficient; LASSO zeroes to exact 0.0 in practice |
| Result: 14 features zeroed | **CONFIRMED** |

### 5. DB Idempotency

| Check | Result |
|-------|--------|
| DELETE before INSERT | **PASS** — `_log_predictions()` deletes by `(tier, model_version)` before bulk insert |
| Reruns produce same row count | **PASS** — deterministic; repeated runs yield exactly 380 rows per model_version |
| Tier constraint | **PASS** — constraint updated to `tier IN (1, 2, 7)` before scripts were run |

### 6. Probability Column Ordering [P(H), P(D), P(A)]

| Check | Result |
|-------|--------|
| Pipeline `classes_` order | **PASS** — `[0, 1, 2]` = H, D, A via `INT_LABEL_TO_IDX = {1:0, 0:1, -1:2}` |
| OOF column naming | **PASS** — `"{name}_H"`, `"{name}_D"`, `"{name}_A"` at indices 0, 1, 2 respectively |
| `prob_home`, `prob_draw`, `prob_away` DB columns | **PASS** — `proba[:, 0]`, `proba[:, 1]`, `proba[:, 2]` |
| `log_loss(labels=[0,1,2])` | **PASS** — explicit labels prevent auto-ordering surprises |

### 7. Metric Correctness

| Check | Result |
|-------|--------|
| `log_loss(labels=[0, 1, 2])` | **PASS** |
| `roc_auc_score(multi_class='ovr', average='macro')` | **PASS** |
| Brier (multiclass OVR mean) | **PASS** — one-vs-rest binarisation, mean over 3 classes |
| CV: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` | **PASS** — same seed used in both tournament.py and stack.py |

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line (pre-fix) | Status |
|---|----------|----------|-------------|---------------------|--------|
| 1 | Minor | Dead code | `X_val_imp = imputer.transform(X_val)` — computed but never used in RF OOB tuning loop | `tournament.py:230` | **Fixed** |
| 2 | Minor | Dead code | `qda_failed = False` — assigned but never read anywhere | `tournament.py:561` | **Fixed** |
| 3 | Minor | Dead code | `y_test_str = test[TARGET_STR_COL]` — string-label column extracted but never used | `tournament.py:646` | **Fixed** |
| 4 | Minor | Dead code | `X_test_reset = X_test.reset_index(drop=True)` — redundant; `X_test` already reset at line 657; all 20 usages replaced with `X_test` | `tournament.py:672` | **Fixed** |
| 5 | Minor | Dead code | `model_configs` list (10 tuples) defined but never iterated — leftover scaffold | `tournament.py:686–697` | **Fixed** |
| 6 | Minor | Dead import | `TARGET_STR_COL` imported in `stack.py` but never referenced | `stack.py:57` | **Fixed** |
| 6b | Minor | Dead import | `TARGET_STR_COL` import in `tournament.py` also orphaned after fix #3 | `tournament.py:53` | **Fixed** |

**No functional bugs found. No data leakage found. All correctness checks pass.**

---

## Results Verified (post-fix — identical to pre-fix)

### Phase 7a Tournament

| Rank | Model | Log Loss | Brier | Acc% | AUC | CV LL ± σ |
|------|-------|----------|-------|------|-----|-----------|
| 1 | LASSO LogReg | 0.9964 | 0.1991 | 52.63 | 0.6537 | 0.9791 ± 0.0280 |
| 2 | Elastic Net | 0.9978 | 0.1994 | 52.63 | 0.6525 | 0.9795 ± 0.0285 |
| 3 | KNN | 1.0131 | 0.2020 | 52.11 | 0.6485 | 0.9921 ± 0.0215 |
| 4 | SVM RBF | 1.0191 | 0.2034 | 52.11 | 0.6291 | 0.9949 ± 0.0252 |
| 5 | Random Forest | 1.0250 | 0.2035 | 50.00 | 0.6401 | 1.0086 ± 0.0501 |
| 6 | Ridge LogReg | 1.0327 | 0.2018 | 51.84 | 0.6447 | 0.9821 ± 0.0316 |
| 7 | LDA | 1.0798 | 0.2079 | 52.37 | 0.6271 | 1.0149 ± 0.0420 |
| 8 | MLP | 1.1173 | 0.2069 | 50.53 | 0.6369 | 1.0060 ± 0.0408 |
| 9 | QDA | 2.7030 | 0.2996 | 36.58 | 0.5807 | 1.9491 ± 0.1830 |
| 10 | Gaussian NB | 2.7266 | 0.2711 | 52.11 | 0.6310 | 2.3400 ± 0.2112 |
| — | **Bookmaker B365** | **0.9708** | **0.1929** | **53.95** | **0.6688** | — |

### Phase 7b Stack

| Model | Log Loss | Brier | Acc% |
|-------|----------|-------|------|
| Super Learner | 1.0079 | 0.2007 | 52.11 |
| LASSO (best T7a) | 0.9964 | — | — |
| Bookmaker B365 | 0.9708 | — | — |

Stack is **+0.0115 worse than LASSO** and **+0.0371 worse than bookmaker**.
OOF CV suggested 0.9782 vs test 1.0079 → 0.030 generalisation gap (meta-learner overfits on 760 OOF samples).

---

## Meta-Learner Weights (mean |coef| across H/D/A classifiers)

| Rank | Base Model | mean\|coef\| |
|------|-----------|------------|
| 1 | KNN | 0.1266 |
| 2 | RF | 0.1080 |
| 3 | MLP | 0.1064 |
| 4 | Ridge LR | 0.1058 |
| 5 | LDA | 0.1021 |
| 6 | SVM | 0.0931 |
| 7 | Elastic Net | 0.0868 |
| 8 | LASSO | 0.0845 |

Note: KNN receiving the highest meta-weight despite being 3rd individually is expected — its errors are less correlated with the logistic regression family, providing diversification. LASSO and Elastic Net receive the lowest weights despite being individually strongest, consistent with the meta-learner already having near-equivalent linear signal from Ridge.

---

## Summary

Both scripts are **functionally correct with no data leakage**. All critical properties verified: scaler isolation, OOF non-overlap, meta-learner fitting order, proba column ordering, LASSO zero-mask axis, and DB idempotency. Seven minor dead code items (6 in tournament.py, 1 in stack.py) were cleaned up with no impact on results, confirmed by re-running both scripts post-fix.

The stacking result (LL=1.0079 < LASSO LL=0.9964) is academically meaningful rather than a bug: all 8 base models share identical features, producing correlated errors that stacking cannot diversify away. The gap narrows to Tier 2 lineup features as the expected source of additional signal.

**Recommendation: APPROVE**
