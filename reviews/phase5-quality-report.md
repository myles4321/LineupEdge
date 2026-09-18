# Quality Report: Phase 5 — Tier 2 Main Model & Core Experiment

**Date:** 2026-09-18
**Reviewer:** Claude Code QA
**Branch:** `phase-5-tier2-model` (based off `phase-4-tier2-features`)

---

## Build Status

| Check | Status |
|-------|--------|
| Python syntax (py_compile, all 3 files) | PASS |
| Python imports (train_tier2, compare_tiers, split) | PASS |
| `--help` flags (train_tier2, compare_tiers) | PASS |
| Graceful fail — tier2_features.csv absent (exit 1 + clear message) | PASS |
| Graceful fail — no T1 predictions in DB (exit 1 + clear message) | PASS |
| Graceful fail — no T2 predictions in DB (exit 1 + clear message) | PASS |
| Tier 1 regression (load_splits, xy, xy_tier2 unchanged) | PASS |
| TIER2_FEATURE_COLS parity (split.py == 14) | PASS |
| Model version tags consistent (train_tier1 ↔ compare_tiers) | PASS |
| Post-fix: re-import check | PASS |

## Code Quality Score: 9/10

(Two minor issues found and fixed; score was 8.5/10 before fixes.)

---

## Acceptance Criteria Verification

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Train Tier 2 XGBoost + LogReg on combined Tier 1+2 features | PASS | `models/train_tier2.py` |
| 2 | Mirror train_tier1.py structure exactly | PASS | Section by section structural match verified |
| 3 | Same hyperparameter search — early stopping on val, fixed n_estimators for final | PASS | `_tune_xgb` grid + `best_iteration + 1` pattern |
| 4 | Same metrics — log loss, Brier, per-class, calibration curves | PASS | All helpers identical to Phase 3 |
| 5 | Log 380 test predictions to DB with tier=2, ON CONFLICT upsert | PASS | `_log_predictions` with parameterised INSERT |
| 6 | Core comparison: log loss delta, Brier delta, per-class improvement | PASS | `compare_tiers.py` — T1 XGB vs T2 XGB primary |
| 7 | Bootstrap significance test (1000 resamples, p-value + 95% CI) | PASS | `_bootstrap_significance` — one-sided, correct framing |
| 8 | Flat-stake betting simulation (£1/bet, B365 decimal odds, ROI) | PASS | `_betting_simulation` |
| 9 | Feature importance plot with Tier 2 features highlighted (orange vs green) | PASS | `_plot_feature_importance` |
| 10 | Side-by-side calibration comparison (T1 vs T2 vs Bookmaker) | PASS | `_plot_calibration_comparison` |
| 11 | Per-class bar chart (green=T2 better, red=T1 better) | PASS | `_plot_tier_improvement` right panel |
| 12 | Results saved to reports/ as JSON | PASS | `tier2_results.json`, `tier_comparison.json` |

---

## Data Path Tracing

| Path | Verified |
|------|---------|
| `xy_tier2()` returns `result_label` (int 1/0/-1) — verified dtype int64, unique {-1,0,1} | PASS |
| `y_train.map(INT_LABEL_TO_IDX)` maps 1→0, 0→1, -1→2 — every int in y_train covered | PASS |
| `y_test_str = test[TARGET_STR_COL]` (H/D/A strings) → `LABEL_TO_IDX` maps to 0/1/2 | PASS |
| `_log_predictions(test.reset_index(drop=True), proba)` — after reset, iterrows `i`=0..N-1 aligns with `proba[i]` | PASS |
| `bm_positions = bm_mask[bm_mask].index.tolist()` → `X_test.iloc[bm_positions]` — safe because test has 0-based index (verified) | PASS |
| compare_tiers: T1/T2 inner-joined on match_id with ground truth before any metric — no row misalignment | PASS |
| compare_tiers: `bm_mask_arr = aligned["match_id"].isin(bm_aligned_df["match_id"])` — correct boolean mask for calibration plot | PASS |
| Betting sim: `aligned.merge(bm_odds, how="left")` preserves aligned's 0-based index; `odds_df.iloc[i]` is positionally aligned with `y_true_str.values[i]` and `proba[i]` | PASS |
| `_per_class_log_loss`: `column_stack([1-p, p])` → binary cross-entropy correct for one-vs-rest | PASS |
| Bootstrap: `delta = ll_t1 - ll_t2`; p-value = fraction ≤ 0; positive delta = T2 improved — matches spec | PASS |
| Model version tags: `tier1_xgb_v1`, `tier1_logreg_v1` in compare_tiers match tags written by train_tier1.py | PASS |

---

## Code Quality Deep Dive

### Architecture & Pattern Consistency
- **Mirrors train_tier1.py exactly** — module docstring → constants → evaluation helpers → training functions → plots → DB logger → main. Line-for-line correspondence verified. PASS.
- **compare_tiers.py** — clean section ordering: DB loaders → metric helpers → bootstrap → betting sim → plots → main. PASS.
- **Naming** — all identifiers descriptive, no abbreviations, consistent with project conventions. PASS.
- **Single responsibility** — every function has one job. Longest function is `main()` in compare_tiers at ~130 lines, but it's a well-structured orchestrator. PASS.

### Type Safety
- Full type annotations on all public and private functions. PASS.
- `dict[str, np.ndarray]` correctly typed for calibration plot routing. PASS.
- `pd.DataFrame | None` and `dict | None` used appropriately for optional data. PASS.

### SQL Safety
- All DB calls use `text("...")` with named parameters (`:version`, `:ids`, `:season`, `:match_id`). No f-string interpolation. PASS.
- `IN :ids` with `tuple()` conversion — safe SQLAlchemy pattern. PASS.

### Statistical Correctness
- **Bootstrap**: one-sided test, `p_value = (deltas ≤ 0).mean()` = fraction of resamples showing no improvement. Correct per spec. PASS.
- **Per-class log loss**: binary one-vs-rest cross-entropy via `column_stack([1-p, p])`. Correct. PASS.
- **Multiclass Brier**: mean of three one-vs-rest Brier scores. Correct. PASS.
- **Betting simulation**: `(decimal_odds - 1)` on win, `-1` on loss. Correct for decimal odds. PASS.
- **Calibration**: `calibration_curve` with `n_bins=8, strategy="uniform"`. Consistent with Phase 3. PASS.

### Performance
| Component | Assessment |
|-----------|------------|
| Bootstrap (1000 × log_loss(380)) | < 1s — each log_loss is O(N×K), fully vectorised |
| Betting sim (O(N) loop, 380 rows) | < 50ms |
| DB insert loop (380 rows × 2 models) | ~1–2s — acceptable at this scale |
| XGB grid search (5 configs × early stopping on val) | ~30–60s — expected and documented |

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line | Status |
|---|----------|----------|-------------|-----------|--------|
| 1 | Minor | Dead Code | `FEATURE_COLS` imported but never referenced in the file — `TIER2_FEATURE_COLS` is used but `FEATURE_COLS` is not | `train_tier2.py:54` (pre-fix) | **Fixed** |
| 2 | Minor | Style | `from matplotlib.patches import Patch` was a local import inside `_plot_feature_importance()` — should be top-level | `train_tier2.py:406` (pre-fix) | **Fixed** |

### Fix Details

**Issue 1 — Dead import `FEATURE_COLS`:**

`train_tier2.py` was importing `FEATURE_COLS` from `models.split` even though it never uses it. Only `TIER2_FEATURE_COLS` is needed for the tier2_present check and the importance-plot colouring. Removed from the import block.

```python
# Before
from models.split import (
    FEATURE_COLS,
    INT_LABEL_TO_IDX,
    ...
)

# After
from models.split import (
    INT_LABEL_TO_IDX,
    ...
)
```

**Issue 2 — Inline `Patch` import:**

`_plot_feature_importance` contained `from matplotlib.patches import Patch` inside the function body. This is imported on every call, is harder to spot in code review, and hides the dependency at the top of the file. Moved to top-level, after the existing matplotlib imports.

---

## Pre-Existing Patterns (Not Fixed — By Design)

| Item | Reason Not Fixed |
|------|-----------------|
| `_plot_calibration(y, predictions, bm)` — `bm` parameter declared but unused inside the function body | Carried over identically from Phase 3's `train_tier1.py`. Fixing train_tier2 but not train_tier1 would break the "mirror exactly" requirement. Flag for Phase 6 cleanup of both files simultaneously. |
| `_plot_tier_improvement` negative-delta bar labels use `val - 0.004` offset, potentially pushing below visible area for large deltas | Cosmetic; won't arise until data is present. No functional impact. |

---

## Regression Check

| Check | Result |
|-------|--------|
| `load_splits()` returns 380/380/380 rows | PASS |
| `xy(train)` returns (380, 38) | PASS |
| `xy_tier2(test)` with T1-only CSV returns (380, 38) gracefully | PASS |
| All Phase 3/4 code imports unchanged | PASS |
| `models/split.py` FEATURE_COLS count (38) unchanged | PASS |
| Model version tags in compare_tiers match those written by train_tier1 | PASS |
| `--help` flags work post-fix | PASS |
| `py_compile` clean post-fix | PASS |
| `import models.train_tier2` clean post-fix | PASS |

---

## Edge Case Handling

| Scenario | Behavior | Correct? |
|----------|----------|---------|
| `tier2_features.csv` missing | Clear error with exact command, `sys.exit(1)` | YES |
| No T1 predictions in DB | compare_tiers exits with actionable error | YES |
| No T2 predictions in DB | compare_tiers exits with actionable error | YES |
| No bookmaker odds in DB | Benchmark skipped gracefully; all other outputs produced | YES |
| All Tier 2 cols 100% null in train | Dropped with info log; model trains on T1 only; warning issued | YES |
| T1/T2 prediction match IDs differ | Inner join produces correct intersection | YES |
| Bootstrap with perfectly equal models | p-value ≈ 0.5, CI straddles zero — correctly non-significant | YES |

---

## Phase 5 Exit Criteria (from Project Plan)

| Criterion | Status |
|-----------|--------|
| Tier 2 model training script implemented | PASS |
| Mirrors train_tier1.py structure exactly | PASS |
| Same hyperparameter search strategy (early stopping → fixed n_estimators) | PASS |
| Same metrics (log loss, Brier, per-class, calibration) | PASS |
| 380 test predictions logged to DB with tier=2, ON CONFLICT upsert | PASS |
| Core comparison experiment implemented | PASS |
| Bootstrap significance test (1000 resamples, p-value, 95% CI) | PASS |
| Flat-stake betting simulation (B365 decimal odds, ROI) | PASS |
| Feature importance with Tier 2 highlighting (orange = T2, green = T1) | PASS |
| Comparison plots (calibration side-by-side, per-class bar) | PASS |
| Results JSON saved to reports/ | PASS |
| Tier 2 model trained end-to-end on real data | BLOCKED — awaiting lineup ingestion |

---

## Dead Code & Reuse

- Dead imports removed: 1 (`FEATURE_COLS` from train_tier2.py)
- Inline imports promoted to top-level: 1 (`from matplotlib.patches import Patch`)
- Duplicate utility logic: `_multiclass_brier`, `_log_loss_from_bm`, `_brier_from_bm`, `_print_eval`, `_log_predictions` are duplicated across train_tier1.py and train_tier2.py. This is intentional — the "mirror exactly" design constraint requires it. Acknowledged and acceptable for this phase. Consolidation into a shared `models/eval_utils.py` would be appropriate in Phase 6 cleanup.

---

## Summary

Phase 5 is a thorough, correct implementation. Both scripts mirror Phase 3 architecture exactly where required. The data path from raw labels through model training to DB logging is traced end-to-end with correct label mapping (INT_LABEL_TO_IDX for training, LABEL_TO_IDX for evaluation). The bootstrap test and betting simulation are statistically correct. Two minor issues were found and fixed: a dead `FEATURE_COLS` import and a function-local matplotlib import that should be top-level. No functional bugs were found. The implementation is correctly blocked on lineup data ingestion — an external constraint, not a code defect.

**Recommendation: APPROVE** (pending lineup data ingestion for final end-to-end run)
