# Edge

Soccer match outcome prediction — EPL 1X2 classification using a two-tier feature pipeline.

**Research question:** Does confirmed pre-match lineup data meaningfully improve win/draw/loss probability estimates over team-level features alone?

---

## Setup

### 1. Python environment (requires Python 3.10 for df-analyze compatibility)

This project uses [uv](https://github.com/astral-sh/uv) for Python/package management.
It handles Python 3.10 installation automatically — no manual version management needed.

```bash
# Create Python 3.10 venv (uv downloads it if not present)
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install df-analyze (STFX AutoML framework)
uv pip install git+https://github.com/stfxecutables/df-analyze.git
```

### 2. Environment variables

```bash
cp .env.example .env
# Edit .env — add your API_FOOTBALL_KEY and DB_USER
```

### 3. Database

```bash
python db/init_db.py
```

### 4. Verify setup

```bash
python -c "from pipeline.api_client import APIFootballClient; print('API client OK')"
python -c "from pipeline.football_data import FootballDataLoader; print('FD loader OK')"
```

---

## Data Ingestion (Phase 1)

Run each step daily. The pipeline resumes from checkpoints automatically.
**Start immediately — ingestion runs in the background over 3–4 weeks.**

```bash
# Step 1: fixtures (~5 requests — run once, all seasons)
python pipeline/ingest.py --step fixtures

# Step 2: lineups (~1,900 requests — run daily, ~19 days)
python pipeline/ingest.py --step lineups

# Step 3: match statistics (~1,900 requests — run daily after lineups complete)
python pipeline/ingest.py --step stats

# Step 4: player stats (~1,900 requests — run daily after stats complete)
python pipeline/ingest.py --step player_stats

# Odds: zero API quota — download from Football-Data.co.uk (run after fixtures)
python pipeline/load_odds.py

# Data quality report (run anytime to check ingestion progress)
python pipeline/validate.py
```

**Daily budget:** 95 requests/day (free plan cap = 100, 5 headroom).
The script stops automatically when the budget is reached and logs how to resume.

**Priority order:** Fixtures → Lineups → Odds → Stats → Player stats.
Lineups and odds are the most critical for Phase 2/4 feature engineering.

---

## Project Structure

```
LineupEdge/
├── config.py               Central config (loaded from .env)
├── db/
│   ├── schema.sql          Full PostgreSQL DDL
│   └── init_db.py          DB initialisation script
├── pipeline/
│   ├── api_client.py       Rate-limited API-Football client + checkpoints
│   ├── football_data.py    Football-Data.co.uk CSV loader (odds)
│   └── ingest.py           Ingestion orchestrator
├── features/               Feature engineering (Phase 2 & 4)
├── models/                 Trained model artifacts (Phase 3 & 5)
├── eval/                   Evaluation utilities (Phase 6)
├── notebooks/              Exploratory analysis
└── data/
    ├── raw/                Raw API responses / CSVs
    └── processed/          Feature matrices ready for modelling
```

---

## Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Setup & data access | In progress |
| 1 | Historical data pipeline | Not started |
| 2 | Tier 1 feature engineering | Not started |
| 3 | Tier 1 baseline model | Not started |
| 4 | Tier 2 features + lineup pipeline | Not started |
| 5 | Tier 2 model + core experiment | Not started |
| 6 | Evaluation + seminar paper | Not started |
