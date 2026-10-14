# Quality Report: Phase 7f — Dashboard & Report Integration

**Date:** 2026-10-14
**Reviewer:** Claude Code QA
**Files reviewed:**
- `dashboard/pages/4_🏆_Model_Tournament.py` (NEW)
- `dashboard/pages/5_🔎_Explainability.py` (NEW)
- `reports/generate_report.py` (UPDATED — sections 13–17)
- `reports/plot_final_figures.py` (UPDATED — 3 new figures)

---

## Build Status

| Check | Status |
|-------|--------|
| `py_compile dashboard/pages/4_🏆_Model_Tournament.py` | PASS |
| `py_compile dashboard/pages/5_🔎_Explainability.py` | PASS |
| `py_compile reports/generate_report.py` | PASS |
| `py_compile reports/plot_final_figures.py` | PASS |
| `python reports/generate_report.py` (end-to-end) | PASS |
| `python reports/plot_final_figures.py --no-db` (end-to-end) | PASS |
| All 3 new PNGs generated | PASS |

---

## Code Quality Score: 8/10

---

## Critical Correctness Checks

### 1. JSON Key Accuracy — `_section_dim_reduction` (3 bugs found and fixed)

**Bug 1 — Wrong LDA log-loss key (pre-fix):**
```python
lda_ll = lda.get("xgb_lda_ll", float("nan"))  # key does not exist
```
Actual key in `dim_reduction_results.json`: `logreg_lda_ll`. The wrong key always returned `nan`,
producing `| LDA (2D) | Max discriminant directions | nan | +nan |` in the Markdown table.

**Fix:**
```python
lda_ll = lda.get("logreg_lda_ll", float("nan"))
```
Verified: now produces `1.6713`.

---

**Bug 2 — Wrong LASSO count key (pre-fix):**
```python
lasso_k = lasso.get("n_survivors", "—")  # key does not exist
```
Actual key: `n_surviving_features`. The wrong key always returned `"—"`, causing:
- The table row to show `"—" features survive` instead of `20`.
- The arithmetic `34 - (lasso_k if isinstance(lasso_k, int) else 34)` to evaluate to `0`
  (the `isinstance(lasso_k, int)` guard correctly caught this, but reported `"The 0 zeroed features"`
  instead of `"The 14 zeroed features"`).

**Fix:**
```python
lasso_k = lasso.get("n_surviving_features", "—")
```
Verified: now produces `20`.

---

**Bug 3 — Wrong LASSO top-feature key (pre-fix):**
```python
lasso_top   = lasso.get("top_features", [])   # key does not exist
lasso_rank1 = lasso_top[0] if lasso_top else "—"
```
Actual key: `surviving_features`. However, that list is stored in column order (not by coefficient
magnitude), so `surviving_features[0]` is `home_elo`, not `elo_diff`. The report text reads:
"Top survivor: `{lasso_rank1}` — convergent validity with XGBoost gain rank 1 (elo_diff)."
Using `home_elo` as the top survivor would have been factually wrong.

**Fix:** Sort `feature_details` by `mean_abs_coef` descending to identify the true #1 survivor:
```python
fd          = lasso.get("feature_details", {})
lasso_rank1 = (
    max(fd, key=lambda f: fd[f].get("mean_abs_coef", 0))
    if fd else "—"
)
```
Verified: now produces `elo_diff` (mean_abs_coef=0.231), correctly matching the claim of
convergent validity with XGBoost gain rank 1.

---

### 2. Dead Variable — `_section_stacking`

```python
delta_bm = comp.get("stack_delta_vs_bookmaker", float("nan"))
```
`delta_bm` was computed but never referenced in the f-string. **Removed.**

---

### 3. Deprecated Streamlit Parameter — `5_🔎_Explainability.py`

```python
st.image(str(wf_path), use_column_width=True)   # deprecated since Streamlit 1.16
```
**Fix:**
```python
st.image(str(wf_path), use_container_width=True)
```
All other `st.image()` calls in the file already correctly use `use_container_width` — this was
the only stale instance.

---

### 4. JSON Key Access Safety — All Other Sections

Verified all dict accesses in the 5 new `_section_*` functions:

| Section | Access pattern | Safe? |
|---------|---------------|-------|
| `_section_tournament` | All via `.get()` with defaults or top-level keys confirmed present | PASS |
| `_section_stacking` | All `.get()` with nan/empty-dict defaults | PASS |
| `_section_dim_reduction` | Fixed (bugs 1–3 above) | PASS (post-fix) |
| `_section_unsupervised` | All `.get()`; cluster dict access uses keys confirmed present in JSON | PASS |
| `_section_shap` | All `.get()` with list defaults; `top15[0]` guarded by `if top15` | PASS |

**One edge case noted (not a crash, minor):** In `_section_unsupervised`, the KMeans cluster
sort uses `x[1]["mean_elo"]` without `.get()`. If a cluster entry were missing `mean_elo`, this
would raise a `KeyError`. Confirmed the JSON always includes `mean_elo` in every cluster entry —
acceptable risk for a research script that controls its own data.

---

### 5. `_section_tournament` — Failed Model Handling

Both QDA and Gaussian NB appear in the ranked table with their actual log-loss values (2.7+) and
CV scores, with a `⚠ catastrophic failure` note appended in the table row. The `best` model
selection correctly skips failed models:
```python
best = next(
    r for r in results
    if _MODEL_DISPLAY.get(r["model_version"], r["model_version"]) not in _FAILED_MODELS
)
```
The tournament chart (`_plot_final_tournament`) correctly filters them out via `_FAILED_PLOT`.
Both the report and the chart are consistent. **PASS**

---

### 6. `_plot_final_tournament` — Chart Correctness

- Models sorted `reverse=True` (worst LL at top, best at bottom) → correct for `barh` with
  y-positions in ascending order: lowest bar = best model at bottom. **PASS**
- Bookmaker shown as `axvline` (vertical dashed line), not as a bar. **PASS**
- Colour logic: `"#9333ea"` for "Stack" models, `"#2563eb"` for all others. No Super Learner
  row is present in `tournament_results.json` — the colour guard is harmless. **PASS**
- Figures saved at 300 dpi. **PASS**

---

### 7. Duplicate Model Display Dicts

`_MODEL_DISPLAY_PLOT` in `plot_final_figures.py` is a duplicate of `_MODEL_DISPLAY` in
`generate_report.py`. Since the two files have different entry points and no shared module,
this duplication is unavoidable without refactoring into a shared constants file — acceptable
at current project scale.

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line (pre-fix) | Status |
|---|----------|----------|-------------|---------------------|--------|
| 1 | Major | Wrong JSON key | `lda.get("xgb_lda_ll")` → key absent → `nan` in Markdown table | `generate_report.py:786` | **Fixed** |
| 2 | Major | Wrong JSON key | `lasso.get("n_survivors")` → key absent → `"—"` instead of `20`, zeroed-count arithmetic produces `0` | `generate_report.py:789` | **Fixed** |
| 3 | Major | Wrong JSON key + wrong sort | `lasso.get("top_features")` → key absent → `"—"`; even with correct key, list is in column order not coefficient order, so `home_elo` would have been reported instead of `elo_diff` | `generate_report.py:790–791` | **Fixed** |
| 4 | Minor | Dead variable | `delta_bm` computed but never used in f-string | `generate_report.py:716` | **Fixed** |
| 5 | Minor | Deprecated API | `use_column_width=True` deprecated in Streamlit ≥ 1.16 | `5_🔎_Explainability.py:146` | **Fixed** |

**Bugs 1–3 were silent data correctness errors: the scripts would run without crashing but produce
wrong values in the report. Bug 3 is the most impactful — `elo_diff` would not have been
identified as the top LASSO survivor, undermining the convergent validity claim.**

---

## Regression Check

| Check | Result |
|-------|--------|
| `generate_report.py` runs end-to-end post-fix | PASS |
| `plot_final_figures.py --no-db` runs end-to-end post-fix | PASS |
| All 5 new report sections present in `evaluation_report.md` | PASS |
| `final_tournament.png` generated (176 KB, 300 dpi) | PASS |
| `final_shap_summary.png` generated (595 KB, copied from beeswarm) | PASS |
| `final_clusters.png` generated (175 KB, copied from kmeans) | PASS |
| Existing sections 1–12 unaffected (pre-fix and post-fix output identical) | PASS |
| `py_compile` on all 4 files post-fix | PASS |

---

## UI/UX Audit (Streamlit Pages)

| Area | Score | Notes |
|------|-------|-------|
| Empty-state handling | 5/5 | All JSON-absent cases show `st.warning()` with run instructions |
| Data display completeness | 5/5 | All sections (table + image + metric) present |
| Image loading safety | 5/5 | All `st.image()` calls guarded by `if path.exists()` |
| Caching | 5/5 | `@st.cache_data` on all JSON loaders |
| Deprecated API usage | 4/5 | One `use_column_width` fixed; all others correct |

### Page 4 (Model Tournament)
- Rankings tab: styled DataFrame with gold (bookmaker), green (best model), red (failed models) row
  highlighting — visually clear benchmark hierarchy.
- Charts tab: two matplotlib figures side-by-side (Log Loss bar + LL vs Brier grouped bar).
  Failed models excluded from charts with caption noting the exclusion.
- CV tab: error-bar chart with bookmaker reference line. ✓
- All three tabs load from cached JSON — no DB dependency. ✓

### Page 5 (Explainability)
- SHAP section: table + 3 images (beeswarm, bar, draw analysis) with existence guards. ✓
- Illustrative matches: 3-column layout, `st.metric()` cards, correct/wrong coloured with
  `st.success`/`st.error`, waterfall PNGs guarded. ✓
- Team clustering: side-by-side image + cluster table. ✓
- Match context clustering: side-by-side dendrogram + hierarchical table. ✓

---

## End-to-End Flow

These are read-only display scripts with no user interactions, DB writes, or mutations.
All data flows are: JSON file → `@st.cache_data` loader → Streamlit display widget.

| Flow | Result | Notes |
|------|--------|-------|
| All JSON present → full report generated | PASS | Verified end-to-end |
| Missing JSON → graceful warning shown (Streamlit) | PASS | `st.warning()` with run instructions |
| Missing JSON → graceful stub section (report) | PASS | Each `_section_*` returns a "not found" stub |
| Missing PNG → image silently skipped (Streamlit) | PASS | All `st.image()` guarded by `path.exists()` |

---

## Summary

`generate_report.py` contained three silent data-correctness bugs in `_section_dim_reduction`:
all three stemmed from wrong JSON key names that did not match the actual keys in
`dim_reduction_results.json`. Bugs 1 and 2 caused `nan` and `"—"` to appear in the Markdown
table; bug 3 would have misidentified `home_elo` as the top LASSO survivor instead of `elo_diff`,
undermining the convergent-validity claim that is a key academic finding of the project. All three
are fixed and verified. The remaining two issues (dead variable, deprecated Streamlit parameter)
are cosmetic. The new Streamlit pages and `plot_final_figures.py` additions are otherwise clean,
well-guarded, and follow existing codebase patterns exactly.

**Recommendation: APPROVE**
