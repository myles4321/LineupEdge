# Reproducibility Guide — Edge Soccer Prediction System

This document provides the full, ordered command sequence to reproduce every
result in the Edge project from scratch. Each step notes its prerequisites,
expected outputs, and whether it is currently blocked on data ingestion.

---

## Prerequisites

### 1. Clone / locate the repository

All commands below are run from the **project root**:
```
/path/to/LineupEdge/
```

### 2. Python environment

The project uses Python 3.10 with `uv` as the package manager.
The system `pip` is broken on macOS Darwin 25.2.0; use `uv` for all
package management.

```bash
# Create and activate the virtual environment
uv venv .venv
source .venv/bin/activate

# Install all dependencies
uv pip install -r requirements.txt
```

### 3. Environment variables

Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
```

Required variables:
| Variable | Description |
|----------|-------------|
| `API_FOOTBALL_KEY` | API-Football free plan key (100 req/day) |
| `DB_USER` | PostgreSQL username |
| `DB_PASSWORD` | PostgreSQL password (leave blank if none) |
| `DB_HOST` | Database host (default: `localhost`) |
| `DB_PORT` | Database port (default: `5432`) |
| `DB_NAME` | Database name (default: `edge`) |

### 4. PostgreSQL database

The `edge` database with all 9 tables must exist before ingestion. The schema
is managed by `db/schema.sql`:
```bash
psql -h localhost -U $DB_USER -c "CREATE DATABASE edge;"
psql -h localhost -U $DB_USER -d edge -f db/schema.sql
```

---

## Phase 0 — Setup ✓ COMPLETE

No commands required beyond Prerequisites above. The repository, DB schema,
and environment are already in place.

---

## Phase 1 — Historical Data Ingestion

### Step 1.1 — Load fixture data (match results, dates)

**Quota cost:** ~5 requests (< 1 day)
```bash
python pipeline/ingest.py --step fixtures
```

**Expected output:** 1,140 rows in the `matches` table
(EPL seasons 2022, 2023, 2024 — 380 matches × 3 seasons).

### Step 1.2 — Load bookmaker odds

**Quota cost:** 0 requests (free CSV download from Football-Data.co.uk)
```bash
python pipeline/load_odds.py
```

**Expected output:** ~1,102 rows in the `odds` table with `bookmaker = 'B365'`.

### Step 1.3 — Load lineups ⚠ BLOCKED — INGESTION RUNNING

**Quota cost:** ~1,900 requests over ~20 days at 95 req/day
```bash
python pipeline/ingest.py --step lineups
```

**Expected output:** rows in `lineups` and `lineup_players` tables for all
1,140 matches. The script is rate-limited to 95 req/day with automatic
checkpoint/resume — safe to interrupt and restart.

### Step 1.4 — Load player statistics ⚠ BLOCKED — DEPENDS ON STEP 1.3

**Quota cost:** ~1,900 requests over ~20 days
```bash
python pipeline/ingest.py --step player_stats
```

**Expected output:** rows in `player_stats` (minutes, ratings per match).
Required for Tier 2 lineup strength and fatigue features.

### Step 1.5 — Data validation

```bash
python pipeline/validate.py
```

**Expected output:** coverage report printed to stdout. Key checks:
- Match coverage: 1,140 rows across 3 seasons
- Lineup coverage: >95% of matches have lineup data
- Odds coverage: >97% of matches have B365 odds
- Null rates per feature column

---

## Phase 2 — Tier 1 Feature Engineering ✓ COMPLETE

```bash
python features/export_tier1.py
```

**Expected output:** `data/processed/tier1_features.csv`
- 1,140 rows × 46 columns
- 38 model-input features + 6 metadata columns + 2 target columns

**Already complete.** The CSV is present and verified.

---

## Phase 3 — Tier 1 Baseline Model ✓ COMPLETE

### Step 3.1 — Feature analysis

```bash
python models/analyse_features.py
```

**Expected output:**
- `reports/feature_analysis.txt` — univariate ranking + CV comparison
- `reports/feature_importance_univariate.png`

### Step 3.2 — Train Tier 1 models

```bash
python models/train_tier1.py
```

**Expected output:**
- `reports/tier1_results.json` — log loss and Brier scores
- `reports/calibration_curves.png`
- `reports/feature_importance_xgb.png`
- `reports/model_comparison.png`
- 760 rows in `predictions` table (`tier1_xgb_v1`: 380, `tier1_logreg_v1`: 380)

**Already complete.** All outputs are present.

**Tier 1 results (2024 test season, 380 matches):**

| Model | Log Loss | Brier Score |
|-------|----------|-------------|
| Logistic Regression | 1.0327 | 0.2018 |
| XGBoost | 1.0201 | 0.2034 |
| Bookmaker B365 | 0.9708 | 0.1929 |

---

## Phase 4 — Tier 2 Feature Engineering ⚠ BLOCKED

**Blocked on:** Steps 1.3 (lineups) and 1.4 (player_stats).

Once lineup ingestion is complete:

### Step 4.1 — Replay watcher (verify coverage)

```bash
python pipeline/lineup_watcher.py replay
```

**Expected output:** Coverage summary — fraction of matches with lineup data.

### Step 4.2 — Leakage audit

```bash
python features/leakage_audit_tier2.py
```

**Expected output:** All 4 temporal checks PASS. Any FAIL indicates a
data pipeline bug that must be fixed before continuing.

### Step 4.3 — Export Tier 2 feature matrix

```bash
python features/export_tier2.py
```

**Expected output:** `data/processed/tier2_features.csv`
- 1,140 rows × 60 columns (46 Tier 1 columns + 14 Tier 2 feature columns)
- 2 formation metadata columns (not model inputs)

---

## Phase 5 — Tier 2 Model & Core Experiment ⚠ BLOCKED

**Blocked on:** Phase 4 (requires `tier2_features.csv`).

### Step 5.1 — Train Tier 2 models

```bash
python models/train_tier2.py
```

**Expected output:**
- `reports/tier2_results.json`
- `reports/calibration_curves_tier2.png`
- `reports/feature_importance_tier2.png` (orange = T2 features, green = T1)
- 760 rows added to `predictions` table (`tier2_xgb_v1`: 380, `tier2_logreg_v1`: 380)

### Step 5.2 — Core comparison experiment

```bash
python models/compare_tiers.py
```

**Expected output:**
- `reports/tier_comparison.json` — full comparison metrics + bootstrap results
- `reports/calibration_comparison.png`
- `reports/tier_improvement.png`

---

## Phase 6 — Evaluation Report & Publication Figures

### Step 6.1 — Generate evaluation report

**Requires:** `reports/tier1_results.json` (present).
Degrades gracefully if `reports/tier_comparison.json` is absent.

```bash
python reports/generate_report.py
```

**Expected output:** `reports/evaluation_report.md`

### Step 6.2 — Generate publication figures

**No DB required for feature importance and metrics comparison:**
```bash
python reports/plot_final_figures.py --no-db
```

**With DB (full calibration figure):**
```bash
python reports/plot_final_figures.py
```

**Expected outputs:**
- `reports/final_feature_importance.png` — top-20 features, 300 dpi
- `reports/final_metrics_comparison.png` — log loss + Brier comparison, 300 dpi
- `reports/final_calibration.png` — reliability diagrams, 300 dpi (requires DB)

---

## Current State Summary

| Phase | Status | Blocker |
|-------|--------|---------|
| 0 — Setup | ✓ Complete | — |
| 1.1 Fixtures | ✓ Complete | — |
| 1.2 Odds | ✓ Complete | — |
| 1.3 Lineups | ⏳ Running | API quota (~20 days) |
| 1.4 Player stats | ⏳ Pending | Depends on 1.3 |
| 2 — Tier 1 features | ✓ Complete | — |
| 3 — Tier 1 model | ✓ Complete | — |
| 4 — Tier 2 features | ⏳ Pending | Depends on 1.3, 1.4 |
| 5 — Tier 2 model | ⏳ Pending | Depends on Phase 4 |
| 6 — Evaluation | ✓ Partial | Tier 2 pending |

---

## Verification Commands

After running each phase, use these to verify the expected state:

```bash
# Count predictions per model version
psql -h localhost -U $DB_USER -d edge \
  -c "SELECT model_version, tier, COUNT(*) FROM predictions GROUP BY 1, 2 ORDER BY 1;"

# Count matches per season
psql -h localhost -U $DB_USER -d edge \
  -c "SELECT season, COUNT(*) FROM matches GROUP BY season ORDER BY season;"

# Check odds coverage
psql -h localhost -U $DB_USER -d edge \
  -c "SELECT COUNT(*) FROM odds WHERE bookmaker = 'B365';"

# Check lineup coverage
psql -h localhost -U $DB_USER -d edge \
  -c "SELECT COUNT(DISTINCT match_id) FROM lineups;"

# Verify feature matrix sizes
python -c "import pandas as pd; df = pd.read_csv('data/processed/tier1_features.csv'); print(df.shape)"
```

---

## Reproducing from an Existing Database

If `predictions` and `matches` tables are already populated (e.g. after
restoring a DB snapshot), skip ingestion and start at Phase 3:

```bash
# Already have tier1_features.csv and predictions in DB
python models/train_tier1.py     # re-trains and re-logs predictions
python reports/generate_report.py
python reports/plot_final_figures.py
```

Note: `train_tier1.py` uses `ON CONFLICT DO UPDATE` upserts, so re-running it
is safe — it will not create duplicate rows.

---

## Known Environment Notes

- **Python 3.10:** Required. df-analyze (if used) now requires ≥ 3.13, so the
  project uses equivalent scikit-learn analysis in `models/analyse_features.py`.
- **uv:** All package management must use `uv pip`, not `pip`. The system pip
  is non-functional on macOS Darwin 25.2.0.
- **xG features:** `home_xg5_for_avg`, `home_xg5_against_avg`,
  `away_xg5_for_avg`, `away_xg5_against_avg` are 100% null until
  `python pipeline/ingest.py --step stats` completes. XGBoost handles NaN
  natively; these features are excluded from Logistic Regression training.
- **H2H features:** ~50% null for matches involving teams with no prior
  EPL meetings. Expected and handled correctly by XGBoost.
