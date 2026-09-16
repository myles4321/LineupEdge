# Quality Report: Phase 3 — Tier 1 Baseline Model

**Date:** 2026-09-15
**Reviewer:** Claude Code QA

---

## Build Status

| Check | Status |
|-------|--------|
| Python imports (all three modules) | PASS |
| Label encoding alignment (INT_LABEL_TO_IDX ↔ LABEL_TO_IDX) | PASS |
| Temporal leakage audit | PASS |
| `_log_predictions` index alignment | PASS |
| Probability sum-to-one (LR + XGBoost) | PASS |
| Bookmaker probability normalization | PASS |
| DB upsert key correctness | PASS |

---

## Code Quality Score: 9.0/10

---

## Focused Audit Results

### 1. Temporal Leakage — PASS

| Step | Data Used | Verdict |
|------|-----------|---------|
| 100%-null column drop | `X_train.isna().mean()` only | PASS |
| LogReg C grid search | `X_train` fit, `X_val` eval | PASS |
| LogReg final model | `train + val` combined, no test | PASS |
| XGBoost param grid search | `X_train` fit, `X_val` eval_set | PASS |
| XGBoost best_n_estimators | `tuning_model.best_iteration` from val early stopping | PASS |
| XGBoost final model | `train + val`, fixed n_estimators, no early stopping | PASS |
| Feature analysis (`run_analysis`) | Only `train` split — `train, _, _ = load_splits()` | PASS |

No step touches the 2024 test set during training or hyperparameter selection. The XGBoost n_estimators fix is correctly implemented: `best_n_estimators = tuning_model.best_iteration + 1`, with the final model re-trained on `train+val` without early stopping or any `eval_set`.

### 2. Label Encoding — PASS

```
LABEL_ORDER     = ["H", "D", "A"]
LABEL_TO_IDX    = {"H": 0, "D": 1, "A": 2}     # string labels → class index
INT_LABEL_TO_IDX = {1: 0, 0: 1, -1: 2}          # integer result_label → class index
```

Verified: `INT_LABEL_TO_IDX[1] == LABEL_TO_IDX["H"] == 0` ✓, `INT_LABEL_TO_IDX[0] == LABEL_TO_IDX["D"] == 1` ✓, `INT_LABEL_TO_IDX[-1] == LABEL_TO_IDX["A"] == 2` ✓

Usage is consistent throughout:
- Training fits use `y.map(INT_LABEL_TO_IDX)` on integer `result_label` ✓
- Evaluation uses `y_true_str.map(LABEL_TO_IDX)` on string `result` column ✓
- `_log_predictions` indexes proba columns via `LABEL_TO_IDX["H"]` etc. ✓
- Bookmaker log_loss uses `y_true_str.map(LABEL_TO_IDX)` ✓

### 3. `_log_predictions` Index Alignment — PASS

Called as `_log_predictions(engine, test.reset_index(drop=True), logreg_proba, ...)`.

After `reset_index(drop=True)`, `test_df.index` is `[0, 1, ..., 379]`. `iterrows()` yields `i ∈ {0, …, 379}`, exactly matching rows of the `proba` numpy array (shape `(380, 3)`). Confirmed programmatically: all 380 indices are in-bounds.

### 4. Probability Sum-to-One — PASS

- `LogisticRegression(multi_class="multinomial")` with softmax: guaranteed sum = 1.0 (verified: min=1.000000, max=1.000000)
- `XGBClassifier(objective="multi:softprob")`: softmax normalization, guaranteed sum = 1.0
- Bookmaker: `implied_home + implied_draw + implied_away = 1.0000` for all rows — the Phase 1 ingestion already normalizes implied probabilities to remove the vig. ✓

### 5. DB Upsert Key — PASS

```sql
ON CONFLICT (match_id, tier, model_version) DO UPDATE SET ...
```

Correct composite key: a match can have predictions from both tier=1 and tier=2, and multiple model versions per tier. This triple uniquely identifies a prediction row and ensures idempotent reruns.

### 6. Bookmaker Benchmark Alignment — PASS

`_load_bookmaker_benchmark` returns a DataFrame keyed by `match_id`. The main loop merges it with `test` via `test.merge(bm, on="match_id", how="left")`, preserving the left row order. `bm_mask` is a boolean series over `test_with_bm.index` (0-based sequential). `bm_positions` are used as `iloc` indices into `X_test` which has the same 0-based sequential index. Row alignment between model predictions and bookmaker probabilities is correct.

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line | Status |
|---|----------|----------|-------------|-----------|--------|
| 1 | Minor | Dead code | `FEATURE_COLS`, `TARGET_COL`, `VAL_SEASON` imported but never referenced in the file body. `xy()` uses `FEATURE_COLS` internally; `TARGET_COL` is not needed outside `split.py`; `VAL_SEASON` is handled by `load_splits()`. | `train_tier1.py:53-59` | Fixed |
| 2 | Minor | Code quality | No `__init__.py` in `models/`. Works via Python 3 namespace packages but is not idiomatic and can cause confusion for tools expecting explicit packages. | `models/` | Fixed |

---

## Design Notes (Not Bugs)

**`_tune_xgb` + `train_xgb` double-pass on best params**
`_tune_xgb` trains with `n_estimators=500` early stopping to select params. `train_xgb` then re-trains with `n_estimators=1000` early stopping using the same best params to get a better estimate of `best_n_estimators`. This is one extra val early stopping run but is methodologically sound — the 1000-tree budget gives a more reliable stopping point than 500. Acceptable for a training script of this size.

**Bookmaker comparison is on a subset**
`_print_eval` reports log loss on all 380 test matches; the bookmaker benchmark reports on only those matches that have B365 odds (confirmed 380/380 in current data). If some matches were missing odds, the model vs. bookmaker comparison would be on different subsets — worth monitoring if the odds table changes. Currently no mismatch.

---

## Summary

The Phase 3 implementation is methodologically clean. All seven focused audit areas pass without exception. The temporal split is sacred throughout: no val or test data is touched during training or hyperparameter selection, the XGBoost n_estimators fix is correctly implemented, label encoding is consistent across all code paths, `_log_predictions` row alignment is guaranteed by `reset_index(drop=True)`, and bookmaker probabilities are already vig-normalised from Phase 1 ingestion.

Two minor issues were found and fixed: three dead imports removed from `train_tier1.py` and an `__init__.py` added to `models/`.

**Recommendation: APPROVE** (fixes applied)
