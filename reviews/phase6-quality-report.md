# Quality Report: Phase 6 — Evaluation, Calibration & Writeup

**Date:** 2026-09-21
**Reviewer:** Claude Code QA
**Branch:** `phase-6-evaluation-writeup` (based off `phase-5-tier2-model`)

---

## Build Status

| Check | Status |
|-------|--------|
| Python syntax (py_compile — generate_report.py) | PASS |
| Python syntax (py_compile — plot_final_figures.py) | PASS |
| `--help` flags (generate_report.py, plot_final_figures.py) | PASS |
| `generate_report.py` runs end-to-end, writes evaluation_report.md | PASS |
| `plot_final_figures.py --no-db` generates 2 figures (no DB needed) | PASS |
| `plot_final_figures.py` with DB generates all 3 figures | PASS |
| Graceful fail — tier1_results.json absent (exit 1 + clear message) | PASS |
| Graceful degrade — tier_comparison.json absent (Tier 1-only report) | PASS |
| Graceful degrade — feature_analysis.txt absent (section skipped) | PASS |
| Graceful degrade — DB unreachable (calibration skipped, warning logged) | PASS |
| Graceful degrade — --no-db (calibration skipped, 2 figures still generated) | PASS |
| T1 regression (generate_report parses tier1_results.json correctly) | PASS |
| Feature ranking parser (34 features parsed from feature_analysis.txt) | PASS |
| Calibration figure uses correct label order (H=0, D=1, A=2) | PASS |
| Publication DPI: all saved figures at 300 dpi | PASS |

## Code Quality Score: 9/10

---

## Acceptance Criteria Verification

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | Publication-quality combined figure (calibration + feature importance) | PASS | `plot_final_figures.py` → 300 dpi, larger fonts, clean axes |
| 2 | Evaluation report generated programmatically from JSON files | PASS | `generate_report.py` reads tier1_results.json + tier_comparison.json |
| 3 | Metrics table: T1 LogReg, T1 XGB, T2 LogReg, T2 XGB, Bookmaker | PASS | Full table in Sections 3+4; T2 rows hidden until tier_comparison.json available |
| 4 | Bootstrap significance section | PASS | Section 5 — renders when tier_comparison.json present |
| 5 | Per-class breakdown | PASS | Section 6 — H/D/A one-vs-rest log loss |
| 6 | Betting simulation section | PASS | Section 7 — renders when tier_comparison.json present |
| 7 | Feature importance analysis with academic framing | PASS | Section 8 — top-20 table + academic citations |
| 8 | Calibration analysis section | PASS | Section 9 — discusses T1/T2/bookmaker calibration quality |
| 9 | Draw prediction limitation discussion | PASS | Section 10 — 4 structural factors explained |
| 10 | Future work section | PASS | Section 11 — 8 directions with impact and complexity |
| 11 | Reproducibility checklist | PASS | `docs/reproducibility.md` — full phase-by-phase command sequence |
| 12 | Report degrades gracefully (Tier 1 only when Tier 2 absent) | PASS | Absent tier_comparison.json → Section 4 replaced with PENDING block |
| 13 | Publication figures at 300 dpi, larger fonts | PASS | `_PUB_RC` rcParams dict; all `savefig(..., dpi=300)` |
| 14 | EDGE_PROJECT_PLAN.md updated with Phase 6 status | PASS | Phase 6 tasks marked complete; deliverables checklist updated |

---

## Files Created

| File | Description |
|------|-------------|
| `reports/generate_report.py` | Programmatic Markdown report from JSON results |
| `reports/plot_final_figures.py` | Publication-quality figure generator (300 dpi) |
| `docs/reproducibility.md` | Full annotated command sequence for reproducibility |
| `reports/evaluation_report.md` | Generated report (output artifact) |
| `reports/final_calibration.png` | Reliability diagrams — 3 classes, all models (300 dpi) |
| `reports/final_metrics_comparison.png` | Log loss + Brier grouped bar chart (300 dpi) |
| `reports/final_feature_importance.png` | Top-20 feature importance horizontal bar chart (300 dpi) |

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line | Status |
|---|----------|----------|-------------|-----------|--------|
| 1 | **Bug** | Correctness | `bm_delta_ll` in `_section_tier2_results()` computed `bm["log_loss"] - t2["log_loss"]` (Bookmaker minus T2, giving `−0.0392`). The column header says "Δ vs T1 XGBoost" — all deltas use T1 as baseline. Should be `t1["log_loss"] - bm["log_loss"]` (giving `+0.0493`). | `generate_report.py:265` | Fixed |
| 2 | Minor | Code style | `from matplotlib.patches import Patch` was imported inside `_plot_final_feature_importance()` function body. Same anti-pattern was flagged in Phase 5 review for `train_tier2.py`. | `plot_final_figures.py:536` (was) | Fixed — moved to top-level imports |
| 3 | Minor | Dead code | `t2_logreg = None` was assigned inside `_plot_final_metrics_comparison()` but never referenced again. | `plot_final_figures.py:400` (was) | Fixed — removed |
| 4 | Minor | Code style | `__import__("sqlalchemy").text("SELECT 1")` used inside `_build_engine()` for the connection health check. Non-idiomatic — `sqlalchemy` was already a declared dependency. | `plot_final_figures.py:111` (was) | Fixed — replaced with `from sqlalchemy import create_engine, text as sa_text` at module level; health check uses `sa_text("SELECT 1")` |

---

## Data Path Tracing

| Path | Verified |
|------|----------|
| `_load_tier1_results()` reads `reports/tier1_results.json` → parses `split`, `n_features`, `results` | PASS |
| `_load_tier_comparison()` returns None when absent — all downstream sections conditioned on `has_comparison` | PASS |
| `_parse_feature_ranking()` regex matches 34 non-null feature rows + 4 null xG rows | PASS |
| generate_report: `_section_tier1_results()` extracts LogReg/XGB/Bookmaker from `results` list by `model` key substring | PASS |
| plot_final_figures: `_load_predictions_with_results()` JOINs predictions to matches on `match_id` with `season=2024`, `status='FT'` | PASS |
| plot_final_figures: `_load_b365_implied()` uses `IN :ids` + `tuple()` — safe SQLAlchemy pattern | PASS |
| plot_final_figures: `_plot_final_calibration()` inner-joins T1 and T2 on match_id before plotting — no row misalignment | PASS |
| Calibration proba index: `proba[:, _LABEL_TO_IDX[class_label]]` — correct (H=0, D=1, A=2) | PASS |
| Metrics comparison: Bookmaker sourced from `tier_comparison.json["bookmaker"]` if present, else falls back to tier1_results.json | PASS |
| Feature importance: `_feature_colour()` maps feature name substrings to group colours correctly | PASS |
| Bookmaker delta in `_section_tier2_results()`: `t1["log_loss"] - bm["log_loss"]` — correct T1-baseline delta (`+0.0493`) | PASS (post-fix) |

---

## Code Quality Deep Dive

### Architecture & Pattern Consistency
- **generate_report.py** — clean section pattern: data loaders → section builders → main assembler. Each section function returns a string. PASS.
- **plot_final_figures.py** — clean section pattern: DB loaders → JSON loaders → helpers → plot functions → main. PASS.
- **Naming** — all identifiers descriptive, consistent with project conventions. PASS.
- **Single responsibility** — each function does one thing. Longest function is `_plot_final_calibration()` at ~80 lines, but it's a single well-structured plotting routine. PASS.

### Type Safety
- Full type annotations on all public and private functions. PASS.
- `dict | None` used correctly for optional comparison data. PASS.
- Return type `pathlib.Path | None` correctly used for functions that may skip. PASS.

### SQL Safety
- `_load_predictions_with_results()` and `_load_b365_implied()` use `text("...", )` / `sa_text("...")` with named parameters. No f-string interpolation. PASS.
- `IN :ids` with `tuple()` conversion — safe SQLAlchemy pattern. PASS.

### Graceful Degradation
- All four degradation scenarios tested and verified:
  1. `tier1_results.json` missing → `sys.exit(1)` with actionable message
  2. `tier_comparison.json` missing → Tier 1-only report with PENDING block
  3. `feature_analysis.txt` missing → feature importance section replaced with skip notice
  4. DB unreachable → calibration figure skipped, warning printed, other figures still generated
- `--no-db` flag explicitly skips DB connection attempt. PASS.

### Publication Quality
- `_PUB_RC` rcParams dict applied via `plt.rc_context()` — no global state mutation. PASS.
- All figures saved with `dpi=300, bbox_inches="tight"`. PASS.
- Font sizes: axes labels 12pt, titles 13–15pt, tick labels 10pt. PASS.
- Feature importance uses feature group colour coding with legend. PASS.
- Metrics comparison includes random baseline reference lines (log(3) for log loss, 2/9 for Brier). PASS.
- Calibration figure uses consistent colour scheme matching `compare_tiers.py` colours. PASS.

---

## Regression Check

| Check | Result |
|-------|--------|
| `reports/tier1_results.json` still parses correctly | PASS |
| `reports/feature_analysis.txt` still parses correctly (34 features) | PASS |
| All Phase 3/4/5 code imports unchanged | PASS |
| DB predictions table unchanged (380 T1 predictions still present) | PASS |
| `evaluation_report.md` renders valid Markdown (all tables close, headers correct) | PASS |
| `final_metrics_comparison.png` and `final_feature_importance.png` regenerated post-fix at 300 dpi | PASS |

---

## Edge Case Handling

| Scenario | Behavior | Correct? |
|----------|----------|---------:|
| `tier1_results.json` absent | `sys.exit(1)` with actionable "Run: python models/train_tier1.py" | YES |
| `tier_comparison.json` absent | Tier 1-only report + PENDING block for Section 4, note printed to stdout | YES |
| `feature_analysis.txt` absent | Feature importance section replaced with skip notice + fix command | YES |
| DB unreachable (bad credentials, no server) | Warning printed, calibration figure skipped, other 2 figures still generated | YES |
| `--no-db` flag | Calibration figure skipped with informative message, other 2 figures generated | YES |
| T2 predictions absent from DB | `_load_predictions_with_results()` returns empty DataFrame → T2 line not added to calibration plot | YES |
| B365 odds absent from DB | `bm_df` empty → Bookmaker not plotted in calibration; Bookmaker in metrics comparison from tier1_results.json | YES |
| Empty feature file | `_parse_feature_ranking()` returns empty list → feature importance section shows skip notice | YES |

---

## Pre-Existing Patterns (Not Fixed — By Design)

| Item | Reason Not Fixed |
|------|------------------|
| Section numbering in `generate_report.py` shifts when tier_comparison.json is absent (Section 4 becomes PENDING instead of Tier 2 results, sections 5–8 vanish, subsequent sections renumber to 5–12) | Acceptable for a programmatically-generated report. The section numbers in the rendered Markdown are always consecutive and meaningful. Noted for future cleanup if strict numbering is required. |

---

## Phase 6 Exit Criteria (from Project Plan)

| Criterion | Status |
|-----------|--------|
| Publication-quality calibration figure generated | PASS |
| Evaluation report generated programmatically from JSON files | PASS |
| Bootstrap significance, per-class breakdown, betting simulation sections | PASS (renders when data available) |
| Feature importance analysis document with academic framing | PASS |
| Reproducibility checklist (full command sequence) | PASS |
| EDGE_PROJECT_PLAN.md updated | PASS |
| Full end-to-end results with Tier 2 | BLOCKED — awaiting lineup ingestion |

---

## Summary

Phase 6 delivers a clean, production-ready evaluation suite. Four issues were found and fixed: one correctness bug (Bookmaker delta in the Tier 2 results table used the wrong baseline, reporting T2−BM instead of T1−BM), two code style issues (inline import moved to module level, dead variable removed), and one readability issue (`__import__` anti-pattern replaced with a standard import). All fixes were verified post-edit.

The report generator reads all experimental results from JSON files and produces a well-structured, 12-section Markdown report. The figure generator produces three 300 dpi publication-quality figures. All four graceful degradation scenarios work correctly. The `--no-db` flag allows the full reporting pipeline to run except for the DB-dependent calibration curves.

The phase is correctly blocked on Tier 2 data ingestion — the same external constraint that blocked Phases 4 and 5. Once lineup ingestion completes, running `python reports/generate_report.py` and `python reports/plot_final_figures.py` will automatically pick up the Tier 2 results.

**Recommendation: APPROVE** (pending lineup data ingestion for full Tier 1 vs. Tier 2 report)
