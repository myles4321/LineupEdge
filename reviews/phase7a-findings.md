# Phase 7a — Model Tournament Findings
**Date:** 2026-10-12
**Script:** `models/tournament.py`
**Test season:** EPL 2024 (380 matches)
**Features used:** 34 (38 Tier 1 minus 4 xG columns that are 100% null until stats ingestion)

---

## Tournament Table (ranked by test log loss)

| Rank | Model | Log Loss | Brier | Acc% | AUC | CV LL ± σ | Best Params |
|------|-------|----------|-------|------|-----|-----------|-------------|
| 1 | LASSO LogReg | 0.9964 | 0.1991 | 52.6 | 0.654 | 0.979 ± 0.028 | C=0.1, L1 |
| 2 | Elastic Net | 0.9978 | 0.1994 | 52.6 | 0.652 | 0.979 ± 0.029 | C=0.1, l1_ratio=0.9 |
| 3 | KNN | 1.0131 | 0.2020 | 52.1 | 0.649 | 0.992 ± 0.021 | k=51, distance-weighted |
| 4 | SVM RBF | 1.0191 | 0.2034 | 52.1 | 0.629 | 0.995 ± 0.025 | C=0.1, gamma=scale |
| 5 | Random Forest | 1.0250 | 0.2035 | 50.0 | 0.640 | 1.009 ± 0.050 | n_est=500, balanced |
| 6 | Ridge LogReg | 1.0327 | 0.2018 | 51.8 | 0.645 | 0.982 ± 0.032 | C=0.01, L2 |
| 7 | LDA | 1.0798 | 0.2079 | 52.4 | 0.627 | 1.015 ± 0.042 | SVD solver |
| 8 | MLP | 1.1173 | 0.2069 | 50.5 | 0.637 | 1.006 ± 0.041 | (128,64), α=1e-4 |
| 9 | QDA | 2.7030 | 0.2996 | 36.6 | 0.581 | 1.949 ± 0.183 | reg=0.1 |
| 10 | Gaussian NB | 2.7266 | 0.2711 | 52.1 | 0.631 | 2.340 ± 0.211 | var_smooth=1e-5 |
| — | **Bookmaker B365** | **0.9708** | **0.1929** | **53.9** | **0.669** | — | — |

---

## Key Findings

### 1. L1 Regularisation Dominates

LASSO and Elastic Net are the best-performing models, narrowly ahead of all others. The Elastic Net's best l1_ratio=0.9 confirms the signal is sparse: L1 does the heavy lifting while L2 adds slight numerical stability.

This is academically significant — it suggests that of the 34 active Tier 1 features, only a subset carry independent predictive signal. The rest add noise that hurts tree/kernel/distance-based methods more than it helps.

### 2. LASSO Feature Elimination (14 of 34 features zeroed)

The following 14 features were eliminated by LASSO (zero coefficient across all 3 OVR classifiers):

| Feature | Group | Interpretation |
|---------|-------|---------------|
| `away_elo` | ELO | Redundant with `elo_diff` |
| `home_form5_draw_rate` | Form | Draw rate weakly predictive |
| `away_form5_win_rate` | Form | Redundant with away form loss rate |
| `away_form5_loss_rate` | Form | — |
| `home_form10_win_rate` | Form | Short window dominates |
| `away_form10_loss_rate` | Form | — |
| `away_away_win_rate` | Venue | Possibly captured by ELO |
| `home_rest_days` | Rest | Rest asymmetry (diff) more informative |
| `home_points` | Table | `points_diff` captures relative standing |
| `away_points` | Table | Same |
| `points_diff` | Table | Correlated with ELO diff |
| `away_ppg_last5` | PPG | Away short-run form redundant |
| `ppg_diff` | PPG | Correlated with ELO/form |
| `h2h_total` | H2H | Count of meetings, not outcome |

**Implication:** The 20 surviving features represent the minimally sufficient signal in Tier 1. ELO diff, home form, H2H rates, and PPG survive; raw standings and redundant venue splits do not.

### 3. Models Cluster Tightly (Ranks 1–6)

All competitive models (LASSO → Ridge) are within 0.036 log-loss of each other (0.996–1.033). This suggests the ceiling for Tier 1 features is around log-loss ≈ 0.99–1.00. The remaining gap to the bookmaker (0.97) is approximately 0.02–0.03 log-loss units — expected to partially close when Tier 2 lineup features are added.

### 4. CV vs Test Alignment

CV log-loss (train+val, 5-fold) tracks well with test for all viable models:
- LASSO: CV=0.979, Test=0.996 (+0.017 generalisation gap — good)
- KNN: CV=0.992, Test=1.013 (+0.021 — good)
- RF: CV=1.009, Test=1.025 (+0.016 — good)
- MLP: CV=1.006, Test=1.117 (+0.111 — **overfit on test**)

The MLP gap is the largest, suggesting the (128,64) architecture and early stopping were insufficient to prevent overfitting on 760 train+val samples. A smaller architecture or stronger regularisation is warranted.

### 5. QDA and Gaussian NB Failure

QDA (log-loss 2.70) and GNB (log-loss 2.73) perform catastrophically. Both assume feature independence and Gaussian distributions — neither holds for correlated ELO/form/standings features. QDA additionally warns about collinear variables during CV folds. These results are included for completeness and academic contrast; they are not viable for production use.

### 6. Gap to Bookmaker

The bookmaker B365 log-loss (0.9708) remains the performance ceiling. All 10 models fall short by at least 0.026 log-loss units. The bookmaker benefits from:
- Real-time team news (injuries, suspensions) — i.e., Tier 2 lineup signals
- Market efficiency from sharp bettors
- Information not encoded in public tabular statistics

**Expected path to close the gap:** Tier 2 lineup features (Phase 4/5) should reduce this gap, as formation and player availability are primary sources of bookmaker edge.

---

## Hyperparameter Tuning Outcomes

| Model | Tuned Param | Best Value | Val LL |
|-------|-------------|-----------|--------|
| RF | n_estimators (OOB) | 500 | OOB=0.513 |
| SVM | C × gamma grid | C=0.1, γ=scale | 0.993 |
| KNN | k ∈ {5,11,21,51} | k=51 | 0.971 |
| GNB | var_smoothing | 1e-5 | 2.777 |
| MLP | alpha (L2) | 1e-4 | 0.9999 |
| Ridge LR | C | 0.01 | 0.970 |
| LASSO LR | C | 0.1 | 0.957 |
| Elastic Net | C × l1_ratio | C=0.1, l1=0.9 | 0.958 |
| LDA | — | SVD | — |
| QDA | reg_param | 0.1 | 3.893 |

Notable: KNN with k=51 (best val LL) is effectively using 51/380 = 13% of the training set as neighbours — a very smooth, high-bias estimator. This is consistent with the noisy nature of EPL match outcomes.

---

## Database Verified

```sql
SELECT model_version, COUNT(*) FROM predictions WHERE tier=7 GROUP BY model_version;
-- 10 rows × 380 = 3,800 predictions logged
```

Schema change: `predictions_tier_check` extended from `tier IN (1, 2)` to `tier IN (1, 2, 7)`.

---

## Output Files

| File | Description |
|------|-------------|
| `reports/tournament_results.json` | Full metrics for all 10 models + bookmaker |
| `models/tournament.py` | Self-contained training script |
| DB `predictions` table | 3,800 rows at tier=7 |

---

## Conclusions for Seminar Report

1. **LASSO is the best single model** on Tier 1 features — L1 regularisation performs implicit feature selection, yielding both the best test log-loss and interpretable feature importance via zero coefficients.
2. **14 of 34 features are redundant** given the others — the minimally sufficient feature set for Tier 1 is approximately 20 features centred on ELO diff, home/away form rates, H2H win rates, and PPG form.
3. **The gap to bookmaker accuracy is ~0.026 log-loss** and is expected to close with lineup data (Tier 2), which captures real-time team news that all Tier 1 models are blind to.
4. **Probabilistic models outperform distribution-assumption models** (QDA, GNB) by a large margin on this dataset, confirming that EPL feature correlations violate naive statistical assumptions.
