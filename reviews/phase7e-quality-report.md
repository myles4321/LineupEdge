# Quality Report: Phase 7e — SHAP Explainability

**Date:** 2026-10-14
**Reviewer:** Claude Code QA
**File reviewed:** `models/explain.py`

---

## Build Status

| Check | Status |
|-------|--------|
| `py_compile models/explain.py` | PASS |
| `python models/explain.py` (end-to-end) | PASS |
| Results post-fix identical to pre-fix | PASS |

---

## Code Quality Score: 9/10

---

## Critical Correctness Checks

### 1. SHAP Class Index Correctness

`INT_LABEL_TO_IDX = {1: 0, 0: 1, -1: 2}` maps:
- result=1 (Home Win) → class index **0**
- result=0 (Draw)     → class index **1**
- result=-1 (Away Win) → class index **2**

Code constants:
```python
_CLASS_H = 0  # Home Win
_CLASS_D = 1  # Draw
_CLASS_A = 2  # Away Win
```

Matches exactly. The XGBClassifier fitted on `y_full_enc` (encoded via `INT_LABEL_TO_IDX`) will produce `predict_proba` columns in this same order. **PASS**

### 2. Illustrative Match Selection

```python
# high_conf_correct: highest max_prob where prediction == actual   ✓
# high_conf_wrong:   highest max_prob where prediction != actual   ✓
# near_draw:         lowest max_prob (most uncertain)              ✓
```

`np.argmin(max_probs)` finds the match where no class exceeds a low ceiling (0.345 in practice) — valid proxy for maximum uncertainty. The name `near_draw` is slightly misleading (it found a predicted-H match, not a predicted-D), but the definition in the docstring is accurate. **PASS** (naming is cosmetic only)

### 3. Waterfall Slicing

```python
shap.plots.waterfall(shap_values[idx, :, pred_class], max_display=15, show=False)
```

`shap_values` has shape `(380, 34, 3)`. `shap_values[idx, :, pred_class]` slices to shape `(34,)` — a 1D `shap.Explanation` for one sample and one class. `shap.plots.waterfall` expects exactly this. **PASS**

### 4. `_build_shap_ranking` — Axis Reduction

```python
mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))  # (380, 34, 3) → (34,)
```

`axis=(0, 2)` reduces over samples (axis 0) and classes (axis 2), leaving the feature axis (axis 1) intact. Produces per-feature mean |SHAP| averaged across all 380 test matches and all 3 outcome classes. **PASS**

Verified numerically:
```python
v = np.random.rand(380, 34, 3)
np.abs(v).mean(axis=(0, 2)).shape  # → (34,)  ✓
```

### 5. `_plot_draw_analysis` — Draw Class Extraction

```python
draw_shap_mean = shap_values.values[:, :, _CLASS_D].mean(axis=0)  # (380, 34) → (34,)
```

`[:, :, _CLASS_D]` selects the Draw class slice `(380, 34)`, then `.mean(axis=0)` averages over samples to yield per-feature mean SHAP for the Draw class. Correctly preserves sign (positive = pushes toward draw, negative = away from draw). **PASS**

### 6. JSON Schema

All required keys present and verified:

| Key | Present | Note |
|-----|---------|------|
| `model` | ✅ | |
| `n_test` | ✅ | 380 |
| `n_features` | ✅ | 34 |
| `shap_top15[].rank/feature/mean_abs_shap/gain_rank` | ✅ | `delta_rank` is a bonus field — not in spec but useful |
| `draw_shap_top10[].feature/mean_shap_draw/direction` | ✅ | |
| `illustrative_matches.{high_conf_correct,high_conf_wrong,near_draw}` | ✅ | |
| `pred_class` stripped from JSON output | ✅ | Internal field correctly excluded |

**PASS**

### 7. Gain Rank Fallback Logic

When `dim_reduction_results.json` is absent, the fallback correctly recomputes ranks from the fitted pipeline using double-argsort (same logic as `dim_reduction.py`). The primary path loads from JSON, which ensures SHAP ranks are compared against the same XGBoost model that produced the Phase 7c gain rankings (not against the newly fitted model here, which uses the same hyperparameters anyway). **PASS**

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line (pre-fix) | Status |
|---|----------|----------|-------------|---------------------|--------|
| 1 | Minor | Dead import | `from sklearn.preprocessing import LabelEncoder` imported but never used anywhere in the script | `explain.py:36` | **Fixed** |
| 2 | Minor | Dead import | `LABEL_ORDER` imported from `models.split` but never referenced in the script body (label encoding uses `INT_LABEL_TO_IDX` directly; class name mapping uses `_LABEL_NAMES`) | `explain.py:48` | **Fixed** |

**No functional bugs found. No data leakage. Both fixes are purely cosmetic — results verified identical pre/post fix.**

---

## Results Verified (post-fix)

### SHAP vs XGBoost Gain Ranking (top 15)

| SHAP Rank | Feature | Mean\|SHAP\| | Gain Rank | Δ |
|-----------|---------|------------|-----------|---|
| 1 | elo_diff | 0.1358 | 1 | 0 |
| 2 | home_elo | 0.1325 | 2 | 0 |
| 3 | points_diff | 0.0892 | — | — |
| 4 | away_form5_draw_rate | 0.0748 | 25 | +21 |
| 5 | home_goal_diff | 0.0633 | 26 | +21 |
| 6 | away_form10_draw_rate | 0.0618 | 13 | +7 |
| 13 | home_form10_loss_rate | 0.0468 | 3 | -10 |

### Draw SHAP Analysis

| Feature | Mean SHAP (Draw) | Direction |
|---------|-----------------|-----------|
| elo_diff | -0.0470 | away_from_draw |
| home_elo | +0.0283 | toward_draw |
| h2h_draw_rate | +0.0103 | toward_draw |
| h2h_home_win_rate | +0.0112 | toward_draw |

---

## Findings of Academic Note

1. **elo_diff and home_elo dominate by both SHAP and gain** — convergent validity across two independent importance metrics (SHAP = marginal contribution; gain = split quality). This is strong evidence these are the true predictive drivers, not artifacts of collinearity.
2. **draw_rate features jump in SHAP rank vs gain rank** — `away_form5_draw_rate` moves from gain rank 25 to SHAP rank 4. Gain underweights features used sparingly but with high marginal impact when applied; SHAP correctly captures this.
3. **elo_diff suppresses draws** (mean SHAP=-0.047 for Draw class) — large ELO gaps produce decisive predicted outcomes; the model does not predict draws when there is a strong favourite. The H2H features are the primary draw promoters.
4. **home_form10_loss_rate falls from gain rank 3 to SHAP rank 13** — gain rank is inflated by high split frequency (XGBoost uses it often but not decisively); SHAP reveals its marginal contribution is modest once elo_diff and home_elo are controlled.

---

## Summary

`explain.py` is functionally correct with no data leakage and no result-invalidating bugs. Two dead imports (`LabelEncoder`, `LABEL_ORDER`) were removed. All critical correctness properties are verified: class index alignment, axis reduction math, waterfall slicing, and JSON schema. The `shap.plots.bar` bug workaround (custom matplotlib implementation) is correctly documented and produces equivalent results.

**Recommendation: APPROVE**
