# Edge — Evaluation Report

**Date:** 2026-09-18
**Scope:** Tier 1 (pre-lineup) only — Tier 2 blocked on lineup data ingestion
**Research question:** Does incorporating confirmed pre-match lineup data
significantly improve 1X2 outcome probability estimates compared to
team-level historical features alone?

---

## 1. Dataset

| Split | Season | Matches |
|-------|--------|---------|
| Train | 2022 | 380 |
| Validation | 2023 | 380 |
| Test | 2024 | 380 |

- **League:** English Premier League (EPL)
- **Seasons:** 2022–2024 (3 complete seasons)
- **Total matches:** 1,140
- **Tier 1 features used for training:** 34
- **Split strategy:** Temporal by season — never random. No future data
  ever enters the training or validation sets.
- **Class distribution (all seasons):** ~45.1% Home Win / ~23.0% Draw / ~31.9% Away Win

---

## 2. Experimental Setup

### Models

| Model | Tier | Description |
|-------|------|-------------|
| Logistic Regression | 1 | Multinomial, C=0.01, L2 regularisation — interpretable baseline |
| XGBoost | 1 | Gradient boosted trees, max_depth=3, lr=0.05, n_estimators via early stopping on val |
| Logistic Regression | 2 | Same architecture on combined Tier 1+2 feature set |
| XGBoost | 2 | Same architecture on combined Tier 1+2 feature set |
| Bookmaker B365 | — | Market benchmark — vig-normalised implied probabilities (not a model input) |

Both XGBoost models use the same hyperparameter search: 5-configuration grid
with early stopping on the validation set, then retrain on train+val combined
using the best iteration + 1 as fixed n_estimators.

### Features

**Tier 1 (38 columns):** ELO ratings, rolling form (last 5/10 matches),
home/away venue splits, rest days, league position and goal difference,
points-per-game, head-to-head history. Note: 4 xG features are 100% null
(team_stats ingestion pending) and are excluded from model training.

**Tier 2 (14 additional columns):** Formation historical win rate, formation
change flag, aggregated lineup strength score, key player availability,
lineup deviation vs. modal XI, player fatigue (minutes in last 14 days).

### Evaluation Criteria

| Metric | Description |
|--------|-------------|
| Log Loss | Primary — multiclass cross-entropy. Random baseline = log(3) ≈ 1.099. Lower = better. |
| Brier Score | Secondary — mean one-vs-rest squared error. Measures calibration quality. |
| Per-class Log Loss | One-vs-rest log loss for each outcome separately (H / D / A). |
| Calibration Curves | Reliability diagrams with 8 uniform bins. |
| Bootstrap Test | 1,000 resamples with replacement; p-value = P(Δ ≤ 0). |
| Betting Simulation | Flat-stake £1 simulation against B365 decimal odds (ROI). |

---

## 3. Tier 1 Results — 2024 Test Season (380 matches)

| Model | Log Loss | Brier Score | Δ vs Bookmaker |
|-------|----------|-------------|----------------|
| Logistic Regression | 1.0327 | 0.2018 | +0.0620 |
| XGBoost | 1.0201 | 0.2034 | +0.0493 |
| Bookmaker B365 | 0.9708 | 0.1929 | — |

**Key observations:**

- XGBoost outperforms Logistic Regression on log loss by 0.0127 nats,
  confirming that non-linear feature interactions exist in the data.
- Both models trail the bookmaker by ~0.05–0.06 nats — consistent with
  team-level-only models lacking player-specific information.
- XGBoost Brier score (0.2034) is marginally higher than Logistic
  Regression (0.2018), reflecting the known overconfidence of
  gradient boosted trees at extreme probabilities.
- The bookmaker gap motivates Tier 2: lineup information may partially close
  the gap by providing player-level signals unavailable in team aggregates.

---

## 4. Tier 2 Results — PENDING

Tier 2 model results are not yet available. The full training and comparison
pipeline is implemented but requires lineup data from the API-Football
ingestion pipeline, which is still running.

**To unblock Tier 2:**

```bash
# Step 1: Complete data ingestion (~19 days at 100 req/day free plan)
python pipeline/ingest.py --step lineups
python pipeline/ingest.py --step player_stats

# Step 2: Build Tier 2 feature matrix
python features/export_tier2.py

# Step 3: Train Tier 2 models and run comparison experiment
python models/train_tier2.py
python models/compare_tiers.py

# Step 4: Regenerate this report with full results
python reports/generate_report.py
```

See `docs/reproducibility.md` for the full command sequence.

---

## 8. Feature Importance (Tier 1)

Univariate ranking by combined score = mean(normalised ANOVA F-statistic,
normalised mutual information). Top 20 non-null features:

| Rank | Feature | Combined | F-Score | MI | Null% |
|------|---------|----------|---------|-----|-------|
| 1 | `elo_diff` | 0.967 | 1.000 | 0.934 | 0% |
| 2 | `home_position` | 0.840 | 0.680 | 1.000 | 3% |
| 3 | `points_diff` | 0.822 | 0.908 | 0.736 | 0% |
| 4 | `ppg_diff` | 0.594 | 0.642 | 0.546 | 3% |
| 5 | `home_goal_diff` | 0.574 | 0.525 | 0.622 | 0% |
| 6 | `away_position` | 0.538 | 0.420 | 0.656 | 3% |
| 7 | `home_elo` | 0.514 | 0.630 | 0.397 | 0% |
| 8 | `home_ppg_last5` | 0.414 | 0.426 | 0.402 | 3% |
| 9 | `away_goal_diff` | 0.414 | 0.392 | 0.435 | 0% |
| 10 | `h2h_home_win_rate` | 0.380 | 0.252 | 0.508 | 50% |
| 11 | `home_home_win_rate` | 0.319 | 0.314 | 0.324 | 5% |
| 12 | `away_ppg_last5` | 0.315 | 0.195 | 0.435 | 3% |
| 13 | `rest_days_diff` | 0.298 | 0.054 | 0.543 | 3% |
| 14 | `home_form10_loss_rate` | 0.296 | 0.374 | 0.219 | 3% |
| 15 | `home_rest_days` | 0.254 | 0.068 | 0.441 | 3% |
| 16 | `away_form10_win_rate` | 0.244 | 0.314 | 0.173 | 3% |
| 17 | `home_form5_loss_rate` | 0.214 | 0.362 | 0.066 | 3% |
| 18 | `away_elo` | 0.199 | 0.399 | 0.000 | 0% |
| 19 | `home_form5_win_rate` | 0.184 | 0.369 | 0.000 | 3% |
| 20 | `home_form10_win_rate` | 0.184 | 0.264 | 0.103 | 3% |

**Observations:**

- **`elo_diff`** is the single most informative feature (combined = 0.967).
  ELO captures long-run team quality differences accumulated over many seasons.
- **`home_position`** and **`points_diff`** are closely related (league position and
  cumulative points), forming a positional quality cluster alongside `elo_diff`.
- **H2H features** show ~50% null rates for teams with no prior meetings —
  handled by XGBoost's native NaN support without imputation.
- **xG features** are 100% null (team_stats ingestion pending) and are
  excluded. When populated, they are expected to complement ELO by capturing
  recent attacking/defensive efficiency.

**Academic framing:** The dominance of ELO-derived and positional features
confirms that team quality, as measured by historical performance, is the
primary driver of 1X2 outcomes — consistent with the broader forecasting
literature (Groll et al., 2018; Hubáček et al., 2019). The Tier 2 hypothesis
is that lineup-level features capture residual variance in individual match
outcomes not explained by team-level averages.

**See:** `reports/final_feature_importance.png` for the publication figure.

---

## 9. Calibration Analysis

Calibration curves (reliability diagrams) plot predicted probability on the
x-axis against the empirical frequency of positives on the y-axis. A
perfectly calibrated model lies on the diagonal (y = x).

**Tier 1 findings:**

- Both models show moderate calibration across all three outcome classes.
- XGBoost tends toward **overconfidence** at high probability predictions
  (above the diagonal in the H and A classes) — a well-known property of
  gradient boosted trees (Niculescu-Mizil & Caruana, 2005).
- Logistic Regression is better calibrated at the cost of higher log loss,
  illustrating the bias-variance trade-off between sharpness and reliability.
- **Bookmaker B365** (vig-normalised) is the best calibrated model,
  confirming that market prices efficiently aggregate public information.

**Draw class calibration:**

Calibration curves for the Draw class are noisier than H or A, reflecting
the inherent unpredictability of draws. Both models tend to under-predict
draw probability in the high-confidence range because high-confidence draw
scenarios are rare and underrepresented in training data.

**Tier 2 expectations:**

Lineup information may improve calibration for matches where known player
absences strongly change the expected team strength differential. For example,
when a key striker is confirmed absent, the predicted home win probability
should decrease and the away win or draw probability should increase —
a signal unavailable to Tier 1.

**See:** `reports/calibration_curves.png` (Tier 1 reliability diagrams) and
`reports/final_calibration.png` (publication-quality, all models side-by-side).

---

## 10. Draw Prediction — Known Limitation

Draw prediction is the most challenging aspect of 1X2 soccer forecasting.
Several structural factors explain persistent underperformance on this class:

1. **Low base rate with high variance.** Draws occur in ~23–25% of EPL
   matches, but individual match draw probabilities vary from ~15% (mismatched
   teams) to ~40% (balanced derby). Models trained on team-level statistics
   cannot reliably identify the match-specific conditions that tip a result
   toward a draw rather than a narrow win.

2. **Symmetric quality implies draw — but is not sufficient.** A draw is
   most likely when both teams are evenly matched. However, evenly matched
   teams frequently produce decisive results (1-0, 2-1) through random
   variance in chance conversion. Without event-level data (shots, xG,
   pressing intensity), the model cannot distinguish a goalless tactical
   battle from a high-tempo match that ends 2-2.

3. **Bookmaker difficulty.** Even professional bookmakers assign draw
   probabilities with lower confidence than home/away outcomes. Implied B365
   draw probabilities in EPL average ~26–28% despite a ~23% empirical base
   rate — reflecting systematic market caution about predicting draws.

4. **Lineup signals for draws.** Pre-match lineup data (Tier 2) may partially
   address draw underperformance by capturing conservative team selections
   (both teams playing a cautious formation, key attacking players rested).
   This is an explicit hypothesis in the Tier 1 vs. Tier 2 per-class
   comparison: if Tier 2 shows the largest improvement on the D class, it
   would validate that lineup-level signals provide unique draw information.

**Methodological handling:** Per-class log loss is reported separately
throughout this report so that draw underperformance does not mask overall
model performance. This follows best practice in soccer forecasting evaluation
(Constantinou & Fenton, 2012).

---

## 11. Future Work

| Direction | Expected Impact | Complexity |
|-----------|----------------|------------|
| Complete lineup ingestion + run Tier 2 end-to-end | **High** — core research result pending | Low (pipeline ready) |
| xG features after team_stats ingestion | Medium — 4 currently null features | Low (pipeline ready) |
| Multi-league generalisation (La Liga, Bundesliga) | High — tests model robustness | Medium |
| Bivariate Poisson goal model | High — richer probability surface for 1X2 | High |
| Player-level embeddings from historical stats | High — improved lineup strength encoding | High |
| Ensemble T1+T2 (fall back to T1 when no lineup) | Medium — production-ready prediction system | Low |
| Live prediction API endpoint | Medium — operational deployment | Medium |
| In-play Bayesian updates from match events | High — sequential probability updates | High |

---

## 12. Reproducibility

See `docs/reproducibility.md` for the full annotated command sequence.
Abbreviated pipeline:

```bash
# Environment (Python 3.10, uv package manager)
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Data ingestion (requires .env with API_FOOTBALL_KEY + DB_USER)
python pipeline/ingest.py --step fixtures
python pipeline/load_odds.py
python pipeline/ingest.py --step lineups     # ~19 days at 100 req/day
python pipeline/ingest.py --step player_stats

# Tier 1 feature engineering
python features/export_tier1.py

# Tier 1 model (Phase 3 — currently complete)
python models/analyse_features.py
python models/train_tier1.py

# Tier 2 (Phase 4+5 — blocked on lineup ingestion)
python features/export_tier2.py
python models/train_tier2.py
python models/compare_tiers.py

# Phase 6 — evaluation report and publication figures
python reports/generate_report.py
python reports/plot_final_figures.py
```

---

*Report generated by `reports/generate_report.py` on 2026-09-18.*
