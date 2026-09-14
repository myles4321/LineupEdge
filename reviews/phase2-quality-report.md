# Quality Report: Phase 2 — Tier 1 Feature Engineering

**Date:** 2026-09-14
**Reviewer:** Claude Code QA

---

## Build Status

| Check | Status |
|-------|--------|
| Python imports | PASS |
| Leakage audit (6 checks) | PASS |
| Feature export end-to-end | PASS |
| Feature matrix written | PASS |

---

## Code Quality Score: 8.5/10

---

## Leakage Audit Results

| Check | Result |
|-------|--------|
| ELO = 1500 on team's first match (24 teams) | PASS |
| Form features NaN on team's first match (24 teams) | PASS |
| Rest days NaN on team's first match (24 teams) | PASS |
| Points = 0 on matchday 1 (30 rows) | PASS |
| Spot-check 20 sampled matches (form5, rest, ELO) | PASS |
| History slices contain only pre-match dates (52 sampled) | PASS |

All checks passed both before and after fixes.

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line | Status |
|---|----------|----------|-------------|-----------|--------|
| 1 | **Major** | Correctness | `position = 20` when no prior season matches. Semantically ambiguous: position=20 could mean "last in the league" OR "no prior matches". XGBoost would see 30 matchday-1 rows where both teams are position=20 with 0 points — no information, but also no falsely injected signal. **Fix:** changed fallback to `np.nan`, consistent with how all other cold-start features handle absence of data. Now shows 2.6% null. | `tier1.py:285` | Fixed |
| 2 | **Major** | Correctness (latent) | ELO access used `iloc[idx]` guarded by `isinstance(idx, int)`. Since `home_elo`/`away_elo` are indexed by `matches.index`, `.loc[idx]` is always correct. With default 0-based index these are equivalent, but if the DataFrame had non-sequential indices (e.g. after a `.query()` or `.dropna()` call upstream), `iloc` would silently return the wrong ELO for every row. Confirmed divergence with a unit test. **Fix:** replaced all three occurrences with `.loc[idx]`. | `tier1.py:423-426` | Fixed |
| 3 | Minor | Code smell | `_season_table_stats` had a `history: pd.DataFrame` parameter that was never referenced in the function body (confirmed via AST analysis). The function uses `all_team_histories` directly. **Fix:** removed the parameter and updated both call sites. | `tier1.py:253` | Fixed |
| 4 | Minor | Performance | `home_prior_all` and `away_prior_all` were computed once, but form5, form10, and xG then re-filtered `home_hist`/`away_hist` from scratch (6 redundant full-array boolean filters per match iteration = 6,840 extra pandas filter operations over 1,140 matches). **Fix:** replaced with `home_prior_all.tail(n)` which is an O(1) view on the already-filtered result. Runtime improvement ~15%. | `tier1.py:375-382` | Fixed |
| 5 | Minor | Dead code | `result` field stored in every team history record in `_build_team_histories` (lines 146, 164) but never read by any downstream feature function (confirmed via AST analysis of all 6 feature functions). **Fix:** removed the field from both home and away record dicts. Reduces history DataFrame memory footprint by ~1 column. | `tier1.py:146,164` | Fixed |

---

## Data Characteristic Finding (Not a Bug)

**`home_points > (matchday-1)*3` for 16 rows** — These are EPL fixture postponements: matches with a stale matchday label (e.g., matchday=7) that were physically played much later in the season (Jan–April 2023 for 2022-season matches). By the time these games were played, the home team had accumulated far more than 6×3=18 points. The feature engineering is **correct** — `_season_table_stats` uses actual match dates (`date < before_date`), not matchday labels, so the points and position values are accurate. This is a data characteristic worth noting in the methodology: `matchday` is not a reliable proxy for "how many matches a team has played" in the EPL due to rescheduling. For model training, use `home_points` as the authoritative measure of season progress, not `matchday`.

---

## Feature Distribution Verification

| Feature Group | Finding | Status |
|---------------|---------|--------|
| ELO | Range 1289–1748, mean ≈1512, std ≈77. Reasonable convergence: top teams earn ~250 points above baseline over a season. | PASS |
| Form rates | Win+draw+loss rates sum to exactly 1.0 for all non-NaN rows | PASS |
| H2H rates | Home win + draw + away win rates sum to exactly 1.0 for all non-NaN rows | PASS |
| Rest days | No negative values. NaN only on first match. | PASS |
| Points | Max = 89 pts. Season max is 99 (33×3) for 38-game season — plausible. | PASS |
| Position | 1–20 for all non-NaN rows. NaN only when no prior season matches. | PASS |
| Target | H: 45.1% / D: 23.0% / A: 31.9% — realistic EPL home advantage | PASS |

---

## Leakage Audit Completeness Assessment

The 6 existing checks are sound. Additional checks that could be added in future:

1. **ELO after 2nd match**: Verify ELO has diverged from 1500 in the direction of the first-match result (winner > 1500, loser < 1500). Currently only verifies first match = 1500.
2. **Season table cross-team consistency**: Verify that on any given matchday, the sum of all teams' points equals `(matches_played * 3) - (draws * 1)` per league rules.
3. **`matchday` unreliability warning**: Add a check that logs how many rows have `home_points > (matchday-1)*3` as a data quality note (16 rows due to postponements).

These are enhancements, not blockers. Current audit is sufficient to confirm temporal correctness.

---

## Exit Criteria Verification

| Criterion | Met? | Evidence |
|-----------|------|---------|
| All Tier 1 features computed and stored | YES | 1,140 rows × 46 columns in `data/processed/tier1_features.csv` |
| Leakage audit passes — no post-match data in pre-match features | YES | 6/6 checks pass |
| Feature distributions look reasonable (ELO convergence, form variation) | YES | ELO std=77, form rates vary sensibly, all rate groups sum to 1.0 |

**Phase 2 exit criteria: ALL MET.**

---

## Summary

The Phase 2 feature engineering is algorithmically correct and temporally clean. The ELO implementation (K=20, standard expected-score formula, pre-match values stored before updating) is correct. All rolling features use strict `date < match_date` filtering. H2H lookup correctly captures all prior meetings regardless of venue. The leakage audit is comprehensive for the implemented feature set.

Two **major** issues were found and fixed: a misleading `position=20` cold-start value (now `NaN`) and a latent `iloc` vs `loc` fragility on ELO access. Three **minor** issues were also fixed: a dead function parameter, redundant filtering in the main loop, and a dead `result` field in history records.

The feature matrix is ready for Phase 3 model training.

**Recommendation: APPROVE** (after fixes applied — all fixes are now in place)
