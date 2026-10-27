"""
Phase 6 — Evaluation Report Generator.

Reads logged experimental results from reports/tier1_results.json
(and reports/tier_comparison.json if present), plus the feature ranking
from reports/feature_analysis.txt, and writes a comprehensive Markdown
evaluation report to reports/evaluation_report.md.

The report degrades gracefully: when tier_comparison.json is absent
(Tier 2 training blocked on lineup data ingestion), the report covers
Tier 1 results only and notes what is pending.

Usage:
    python reports/generate_report.py
    python reports/generate_report.py --output reports/my_report.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import date


_REPORTS_DIR = pathlib.Path("reports")
_DEFAULT_OUTPUT = _REPORTS_DIR / "evaluation_report.md"
_TODAY = date.today().isoformat()

# Canonical class labels: index 0=H, 1=D, 2=A
_CLASS_LABELS = ["H", "D", "A"]
_CLASS_NAMES  = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


# ── Data loaders ──────────────────────────────────────────────────────────────


def _load_tier1_results(reports_dir: pathlib.Path) -> dict:
    """Load tier1_results.json — required; exits 1 if absent."""
    path = reports_dir / "tier1_results.json"
    if not path.exists():
        print(f"[ERROR] {path} not found.", file=sys.stderr)
        print("Run:  python models/train_tier1.py", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def _load_tier_comparison(reports_dir: pathlib.Path) -> dict | None:
    """Load tier_comparison.json — optional; returns None if absent."""
    path = reports_dir / "tier_comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _parse_feature_ranking(reports_dir: pathlib.Path) -> list[dict]:
    """Parse feature_analysis.txt into a list of ranked feature dicts.

    Each dict has keys: rank, name, f_score, mi, combined, null_pct.
    """
    path = reports_dir / "feature_analysis.txt"
    if not path.exists():
        return []

    features: list[dict] = []
    for line in path.read_text().splitlines():
        # Match lines like: "1     elo_diff   1.000   0.934   0.967   0.0%"
        m = re.match(
            r"^\s*(\d+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%",
            line,
        )
        if m:
            rank, name, fscore, mi, combined, null_pct = m.groups()
            features.append(
                {
                    "rank":     int(rank),
                    "name":     name,
                    "f_score":  float(fscore),
                    "mi":       float(mi),
                    "combined": float(combined),
                    "null_pct": float(null_pct),
                }
            )
    return features


# ── Report sections ───────────────────────────────────────────────────────────


def _section_header(has_comparison: bool) -> str:
    tier_note = (
        "Tier 1 (pre-lineup) and Tier 2 (post-lineup)"
        if has_comparison
        else "Tier 1 (pre-lineup) only — Tier 2 blocked on lineup data ingestion"
    )
    return f"""\
# Edge — Evaluation Report

**Date:** {_TODAY}
**Scope:** {tier_note}
**Research question:** Does incorporating confirmed pre-match lineup data
significantly improve 1X2 outcome probability estimates compared to
team-level historical features alone?

---
"""


def _section_dataset(tier1: dict) -> str:
    split = tier1["split"]
    return f"""\
## 1. Dataset

| Split | Season | Matches |
|-------|--------|---------|
| Train | {split['train']} | 380 |
| Validation | {split['val']} | 380 |
| Test | {split['test']} | 380 |

- **League:** English Premier League (EPL)
- **Seasons:** {split['train']}–{split['test']} (3 complete seasons)
- **Total matches:** 1,140
- **Tier 1 features used for training:** {tier1['n_features']}
- **Split strategy:** Temporal by season — never random. No future data
  ever enters the training or validation sets.
- **Class distribution (all seasons):** ~45.1% Home Win / ~23.0% Draw / ~31.9% Away Win

---
"""


def _section_experimental_setup() -> str:
    return """\
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
"""


def _section_tier1_results(tier1: dict) -> str:
    results = tier1["results"]
    bm = next((r for r in results if "Bookmaker" in r["model"]), None)

    rows = []
    for r in results:
        if bm and r["model"] != bm["model"]:
            delta = r["log_loss"] - bm["log_loss"]
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "—"
        rows.append(
            f"| {r['model']} | {r['log_loss']:.4f} | {r['brier']:.4f} | {delta_str} |"
        )

    table = "\n".join(rows)

    logreg = next(r for r in results if "Logistic" in r["model"])
    xgb    = next(r for r in results if "XGBoost" in r["model"])
    ll_gap = logreg["log_loss"] - xgb["log_loss"]

    return f"""\
## 3. Tier 1 Results — 2024 Test Season (380 matches)

| Model | Log Loss | Brier Score | Δ vs Bookmaker |
|-------|----------|-------------|----------------|
{table}

**Key observations:**

- XGBoost outperforms Logistic Regression on log loss by {ll_gap:.4f} nats,
  confirming that non-linear feature interactions exist in the data.
- Both models trail the bookmaker by ~0.05–0.06 nats — consistent with
  team-level-only models lacking player-specific information.
- XGBoost Brier score ({xgb['brier']:.4f}) is marginally higher than Logistic
  Regression ({logreg['brier']:.4f}), reflecting the known overconfidence of
  gradient boosted trees at extreme probabilities.
- The bookmaker gap motivates Tier 2: lineup information may partially close
  the gap by providing player-level signals unavailable in team aggregates.

---
"""


def _section_pending_tier2() -> str:
    return """\
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
"""


def _section_tier2_results(comparison: dict) -> str:
    t1 = comparison["tier1_xgb"]
    t2 = comparison["tier2_xgb"]
    bm = comparison.get("bookmaker")

    delta_ll    = t1["log_loss"] - t2["log_loss"]
    delta_brier = t1["brier"] - t2["brier"]
    t2_better   = delta_ll > 0

    bm_row = ""
    if bm:
        # Delta vs T1 XGBoost: positive = this model is better than T1
        bm_delta_ll = t1["log_loss"] - bm["log_loss"]
        bm_row = (
            f"| Bookmaker B365 | {bm['log_loss']:.4f} | {bm['brier']:.4f} | "
            f"{bm_delta_ll:+.4f} |"
        )

    direction = "improvement" if t2_better else "regression"

    return f"""\
## 4. Tier 2 Results — 2024 Test Season ({comparison['n_matches']} matches)

| Model | Log Loss | Brier Score | Δ vs T1 XGBoost |
|-------|----------|-------------|-----------------|
| Tier 1 XGBoost | {t1['log_loss']:.4f} | {t1['brier']:.4f} | — |
| Tier 2 XGBoost | {t2['log_loss']:.4f} | {t2['brier']:.4f} | {delta_ll:+.4f} |
{bm_row}

**Core result:** Tier 2 XGBoost {'outperforms' if t2_better else 'underperforms'} Tier 1
XGBoost by {abs(delta_ll):.4f} nats log loss ({direction}).
Brier score delta: {delta_brier:+.4f}
({'Tier 2 better calibrated' if delta_brier > 0 else 'Tier 1 better calibrated'}).

---
"""


def _section_bootstrap(comparison: dict) -> str:
    bs = comparison["bootstrap"]

    if bs["significant_at_05"]:
        sig_text = (
            "**statistically significant** (p < 0.05). "
            "Tier 2 significantly outperforms Tier 1 at the α = 5% level."
        )
    else:
        sig_text = (
            "**not statistically significant** (p ≥ 0.05). "
            "The improvement cannot be distinguished from random sampling variation."
        )

    return f"""\
## 5. Bootstrap Significance Test

| Parameter | Value |
|-----------|-------|
| Observed Δ log loss (T1 − T2) | {bs['observed_delta_log_loss']:+.4f} |
| p-value (one-sided) | {bs['p_value']:.4f} |
| 95% CI on Δ | [{bs['ci_95_lower']:+.4f}, {bs['ci_95_upper']:+.4f}] |
| Bootstrap resamples | {bs['n_bootstrap']:,} |
| Significant at α = 5%? | {'YES' if bs['significant_at_05'] else 'NO'} |

**Interpretation:** The result is {sig_text}

**Procedure:** {bs['n_bootstrap']:,} resamples (with replacement, size N =
{comparison['n_matches']}) are drawn from the test set. For each resample,
Δ = log_loss(T1) − log_loss(T2) is computed. The p-value is the fraction of
resamples where Δ ≤ 0 — i.e. the probability of observing no improvement
from Tier 2 under the null hypothesis that both models are equivalent.

---
"""


def _section_per_class(comparison: dict) -> str:
    pc1   = comparison["per_class_t1"]
    pc2   = comparison["per_class_t2"]
    delta = comparison["per_class_delta"]

    rows = []
    for c in _CLASS_LABELS:
        d      = delta[c]
        winner = "T2 better" if d > 0 else "T1 better"
        rows.append(
            f"| {_CLASS_NAMES[c]} ({c}) | {pc1[c]:.4f} | {pc2[c]:.4f} | {d:+.4f} | {winner} |"
        )
    table = "\n".join(rows)

    return f"""\
## 6. Per-Class Breakdown

One-vs-rest log loss per outcome class (lower = better):

| Class | T1 XGBoost | T2 XGBoost | Δ (T1−T2) | Winner |
|-------|-----------|-----------|-----------|--------|
{table}

**Draw class note:** The Draw class is consistently the hardest to predict.
Both models show higher log loss on draws than on home or away wins. This is
expected — see Section 10 for a detailed discussion of draw prediction
limitations.

---
"""


def _section_betting_sim(comparison: dict) -> str:
    bets = comparison.get("betting_simulation", [])
    if not bets:
        return """\
## 7. Betting Simulation

No betting simulation data available (requires B365 odds in the database).

---
"""

    rows = []
    for b in bets:
        rows.append(
            f"| {b['model']} | {b['n_bets']} | {b['n_wins']} | "
            f"{b['win_rate_pct']:.1f}% | "
            f"£{b['total_pnl']:+.2f} | {b['roi_pct']:+.1f}% |"
        )
    table = "\n".join(rows)

    return f"""\
## 7. Betting Simulation

Flat-stake simulation (£1 per bet) against B365 decimal odds. Strategy: bet
on the outcome with the highest predicted probability (argmax).

| Model | Bets | Wins | Win Rate | P&L | ROI |
|-------|------|------|----------|-----|-----|
{table}

**Interpretation:** A negative ROI is expected — the bookmaker margin (vig)
is built into decimal odds. The comparison between Tier 1 and Tier 2 ROI
shows whether lineup information produces meaningful improvements in the
model's bet selection quality above and beyond the bookmaker's edge.

---
"""


def _section_feature_importance(features: list[dict]) -> str:
    if not features:
        return """\
## 8. Feature Importance (Tier 1)

Feature ranking data not available (`reports/feature_analysis.txt` not found).
Run: `python models/analyse_features.py`

---
"""

    # Top 20 non-null features only
    top_20 = [f for f in features if f["null_pct"] < 100][:20]

    rows = [
        f"| {f['rank']} | `{f['name']}` | {f['combined']:.3f} | "
        f"{f['f_score']:.3f} | {f['mi']:.3f} | {f['null_pct']:.0f}% |"
        for f in top_20
    ]
    table = "\n".join(rows)

    top3 = [f["name"] for f in top_20[:3]]

    return f"""\
## 8. Feature Importance (Tier 1)

Univariate ranking by combined score = mean(normalised ANOVA F-statistic,
normalised mutual information). Top 20 non-null features:

| Rank | Feature | Combined | F-Score | MI | Null% |
|------|---------|----------|---------|-----|-------|
{table}

**Observations:**

- **`{top3[0]}`** is the single most informative feature (combined = {top_20[0]['combined']:.3f}).
  ELO captures long-run team quality differences accumulated over many seasons.
- **`{top3[1]}`** and **`{top3[2]}`** are closely related (league position and
  cumulative points), forming a positional quality cluster alongside `{top3[0]}`.
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
"""


def _section_calibration() -> str:
    return """\
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
"""


def _section_draw_limitations() -> str:
    return """\
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
"""


def _section_future_work() -> str:
    return """\
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
"""


def _section_reproducibility() -> str:
    return f"""\
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

*Report generated by `reports/generate_report.py` on {_TODAY}.*
"""


# ── Phase 7 section helpers ───────────────────────────────────────────────────


def _load_json_optional(reports_dir: pathlib.Path, filename: str) -> dict | None:
    """Load a JSON file; return None if absent (graceful degradation)."""
    path = reports_dir / filename
    if not path.exists():
        return None
    return json.loads(path.read_text())


_MODEL_DISPLAY = {
    "tier7_rf_v1":    "Random Forest",
    "tier7_svm_v1":   "SVM RBF",
    "tier7_knn_v1":   "KNN",
    "tier7_gnb_v1":   "Gaussian NB",
    "tier7_mlp_v1":   "MLP",
    "tier7_ridge_v1": "Ridge LogReg",
    "tier7_lasso_v1": "LASSO LogReg",
    "tier7_enet_v1":  "Elastic Net",
    "tier7_lda_v1":   "LDA",
    "tier7_qda_v1":   "QDA",
}

_FAILED_MODELS = {"Gaussian NB", "QDA"}


# ── Phase 7a: Model Tournament ────────────────────────────────────────────────


def _section_tournament(reports_dir: pathlib.Path) -> str:
    data = _load_json_optional(reports_dir, "tournament_results.json")
    if data is None:
        return """\
## 13. Model Tournament (Phase 7a)

`tournament_results.json` not found. Run `python models/tournament.py`.

---
"""

    results  = sorted(data["results"], key=lambda r: r["log_loss"])
    bm       = data["bookmaker"]
    best     = next(
        r for r in results
        if _MODEL_DISPLAY.get(r["model_version"], r["model_version"]) not in _FAILED_MODELS
    )
    best_name = _MODEL_DISPLAY.get(best["model_version"], best["model_version"])
    gap       = best["log_loss"] - bm["log_loss"]

    rows = []
    for i, r in enumerate(results, 1):
        name  = _MODEL_DISPLAY.get(r["model_version"], r["model_version"])
        note  = " ⚠ catastrophic failure" if name in _FAILED_MODELS else ""
        rows.append(
            f"| {i} | {name} | {r['log_loss']:.4f} | {r['brier_score']:.4f} | "
            f"{r['accuracy']*100:.1f}% | {r['roc_auc_macro']:.4f} | "
            f"{r['cv_mean_log_loss']:.4f} ± {r['cv_std_log_loss']:.4f}{note} |"
        )
    # Bookmaker row
    rows.append(
        f"| ★ | Bookmaker B365 | {bm['log_loss']:.4f} | {bm['brier_score']:.4f} | "
        f"{bm['accuracy']*100:.1f}% | {bm['roc_auc_macro']:.4f} | — |"
    )
    table = "\n".join(rows)

    lasso_zeros = data.get("lasso_zero_features", [])
    lasso_note = (
        f"LASSO zeroed {len(lasso_zeros)}/34 features: {', '.join(lasso_zeros[:5])}"
        + (" …" if len(lasso_zeros) > 5 else ".")
        if lasso_zeros else ""
    )

    return f"""\
## 13. Model Tournament (Phase 7a)

10 classifiers evaluated on the EPL 2024 test season (380 matches).
All models trained on 2022+2023, tuned on 2023 validation season.

| Rank | Model | Log Loss | Brier | Acc% | AUC | CV LL ± σ |
|------|-------|----------|-------|------|-----|-----------|
{table}

**Key findings:**

- **Winner: {best_name}** — log loss {best["log_loss"]:.4f}, gap to bookmaker: {gap:+.4f} nats.
- **QDA and Gaussian NB fail catastrophically** (LL > 2.7) due to collinearity among Tier 1
  features. QDA's covariance inversion is numerically unstable; GNB's feature-independence
  assumption is severely violated.
- The ~0.026-nat gap to B365 is expected for team-level-only models — lineup data (Tier 2)
  is the anticipated source of the remaining signal.
{('- ' + lasso_note) if lasso_note else ''}

---
"""


# ── Phase 7b: Stacking / Super Learner ────────────────────────────────────────


def _section_stacking(reports_dir: pathlib.Path) -> str:
    data = _load_json_optional(reports_dir, "stack_results.json")
    if data is None:
        return """\
## 14. Stacking / Super Learner (Phase 7b)

`stack_results.json` not found. Run `python models/stack.py`.

---
"""

    stack = data["stack"]
    comp  = data.get("comparison", {})
    lasso_ll    = comp.get("lasso_log_loss", float("nan"))
    bm_ll       = comp.get("bookmaker_log_loss", float("nan"))
    delta_lasso = comp.get("stack_delta_vs_lasso", float("nan"))

    _SHORT_TO_DISPLAY = {
        "rf": "Random Forest", "svm": "SVM RBF", "knn": "KNN",
        "mlp": "MLP", "ridge": "Ridge LogReg", "lasso": "LASSO LogReg",
        "enet": "Elastic Net", "lda": "LDA",
    }
    weights = data.get("base_model_weights", {})
    top3_weights = sorted(
        weights.items(), key=lambda x: -x[1].get("mean_abs_coef", 0)
    )[:3]
    weight_rows = "\n".join(
        f"| {_SHORT_TO_DISPLAY.get(k, k)} | {v.get('mean_abs_coef', float('nan')):.4f} |"
        for k, v in top3_weights
    )

    return f"""\
## 14. Stacking / Super Learner (Phase 7b)

A Super Learner (stacked ensemble) was trained using 8 base classifiers
(excluding QDA and Gaussian NB) and a Logistic Regression meta-learner.

| Model | Log Loss | Note |
|-------|----------|------|
| LASSO LogReg (best individual) | {lasso_ll:.4f} | Best Phase 7a model |
| Super Learner (stack) | {stack['log_loss']:.4f} | Meta-learner ensemble |
| Bookmaker B365 | {bm_ll:.4f} | Market benchmark |

**Stack vs LASSO:** {delta_lasso:+.4f} nats (positive = stack worse than LASSO).

**Top 3 meta-learner feature weights:**

| Base Model | Weight |
|------------|--------|
{weight_rows}

**Academic note:** The Super Learner underperforms LASSO (+{delta_lasso:.4f} nats) because
all 8 base models share identical Tier 1 features, producing highly correlated errors.
Stacking requires diverse base learners to add value — a theoretical prediction of the
stacking literature (Wolpert, 1992; Van der Laan et al., 2007) confirmed empirically here.
The stack does not significantly diversify away from the LASSO's already-sparse
feature representation.

---
"""


# ── Phase 7c: Dimensionality Reduction ────────────────────────────────────────


def _section_dim_reduction(reports_dir: pathlib.Path) -> str:
    data = _load_json_optional(reports_dir, "dim_reduction_results.json")
    if data is None:
        return """\
## 15. Dimensionality Reduction (Phase 7c)

`dim_reduction_results.json` not found. Run `python models/dim_reduction.py`.

---
"""

    pca   = data.get("pca", {})
    lda   = data.get("lda", {})
    lasso = data.get("lasso_selection", {})

    pca_n        = pca.get("n_components_95pct", "—")
    pca_full_ll  = pca.get("xgb_full_ll", float("nan"))
    pca_pca_ll   = pca.get("xgb_pca_ll", float("nan"))
    pca_delta    = pca_pca_ll - pca_full_ll

    lda_ll       = lda.get("logreg_lda_ll", float("nan"))
    lda_n        = lda.get("n_components", "—")

    lasso_k      = lasso.get("n_surviving_features", "—")
    fd           = lasso.get("feature_details", {})
    lasso_rank1  = (
        max(fd, key=lambda f: fd[f].get("mean_abs_coef", 0))
        if fd else "—"
    )

    return f"""\
## 15. Dimensionality Reduction (Phase 7c)

| Method | Summary | XGBoost Log Loss | Δ vs Full |
|--------|---------|-----------------|-----------|
| Baseline (full 34 features) | All Tier 1 features | {pca_full_ll:.4f} | — |
| PCA ({pca_n} components) | {pca_n} components retain 95% variance | {pca_pca_ll:.4f} | {pca_delta:+.4f} |
| LDA ({lda_n}D) | Max discriminant directions | {lda_ll:.4f} | {lda_ll - pca_full_ll:+.4f} |
| LASSO selection | {lasso_k} features survive at C=0.1 | — | — |

**Key findings:**

- **PCA:** 34 features compress to {pca_n} components for 95% variance retention.
  XGBoost on PCA space shows Δ = {pca_delta:+.4f} nats vs. full features — confirming
  that feature redundancy exists but raw features are preferable for tree models.
- **LDA** projects to {lda_n} discriminant dimensions. Log loss = {lda_ll:.4f}.
  LDA's linear assumption is more restrictive than XGBoost's native feature handling.
- **LASSO** selection: {lasso_k}/34 features survive with non-zero coefficients.
  Top survivor: `{lasso_rank1}` — convergent validity with XGBoost gain rank 1 (elo_diff).
  The {34 - (lasso_k if isinstance(lasso_k, int) else 34)} zeroed features are
  redundant form rates, standings, and rest-day variants.

---
"""


# ── Phase 7d: Unsupervised Analysis ───────────────────────────────────────────


def _section_unsupervised(reports_dir: pathlib.Path) -> str:
    data = _load_json_optional(reports_dir, "unsupervised_results.json")
    if data is None:
        return """\
## 16. Unsupervised Analysis (Phase 7d)

`unsupervised_results.json` not found. Run `python models/unsupervised.py`.

---
"""

    km   = data.get("kmeans", {})
    hier = data.get("hierarchical", {})

    best_k   = km.get("best_k", "—")
    sil      = km.get("silhouette_scores", {})
    best_sil = sil.get(str(best_k), float("nan"))

    km_clusters = km.get("clusters", {})
    km_rows = []
    for c_id, info in sorted(km_clusters.items(), key=lambda x: -x[1]["mean_elo"]):
        km_rows.append(
            f"| {c_id} | {info['size']} | {info['mean_elo']:.0f} | "
            f"{info['home_win_rate']*100:.1f}% | {info['away_win_rate']*100:.1f}% |"
        )
    km_table = "\n".join(km_rows)

    hier_clusters = hier.get("clusters", {})
    hier_rows = []
    for c_id, info in sorted(
        hier_clusters.items(), key=lambda x: -x[1]["mean_elo_diff"]
    ):
        op = info.get("outcome_pct", {})
        hier_rows.append(
            f"| {info.get('label', c_id)} | {info['size']} | "
            f"{info['mean_elo_diff']:+.1f} | "
            f"{op.get('H', '—')} | {op.get('D', '—')} | {op.get('A', '—')} |"
        )
    hier_table = "\n".join(hier_rows)

    return f"""\
## 16. Unsupervised Analysis (Phase 7d)

### K-Means Team Clustering

Best k = **{best_k}** (silhouette = {best_sil:.3f}).

| Cluster | Size | Mean ELO | Home W% | Away W% |
|---------|------|----------|---------|---------|
{km_table}

ELO bands: ~1635 elite (top-6), ~1542 upper-mid, ~1477 mid-table, ~1430 relegation.

### Ward Hierarchical Match-Context Clustering (n = {hier.get('n_matches', '—')} matches)

| Context Label | Size | ELO Δ | H% | D% | A% |
|---------------|------|-------|----|----|-----|
{hier_table}

**Key findings:**

- K = 4 is clearly optimal (silhouette {best_sil:.3f} vs k=3: see JSON) — matches the
  well-known four-tier EPL structure (elite / upper-mid / mid-table / relegation).
- **Away Favoured** is the largest match-context cluster (n ≈ 543, 48%) — most EPL
  matches involve a stronger visiting side, reflecting that top-6 clubs frequently
  travel to mid/lower clubs.
- **Strong Home** (ELO diff > +190, H% ≈ 69%) is rare (n ≈ 94) but the most
  predictable context in the dataset.

---
"""


# ── Phase 7e: SHAP Explainability ─────────────────────────────────────────────


def _section_shap(reports_dir: pathlib.Path) -> str:
    data = _load_json_optional(reports_dir, "shap_results.json")
    if data is None:
        return """\
## 17. SHAP Explainability (Phase 7e)

`shap_results.json` not found. Run `python models/explain.py`.

---
"""

    top15   = data.get("shap_top15", [])
    draw10  = data.get("draw_shap_top10", [])

    top5_rows = []
    for f in top15[:5]:
        gain_rank = f.get("gain_rank", "—")
        delta     = f.get("delta_rank", "—")
        top5_rows.append(
            f"| {f['rank']} | `{f['feature']}` | {f['mean_abs_shap']:.4f} | "
            f"{gain_rank} | {delta} |"
        )
    top5_table = "\n".join(top5_rows)

    draw_pushers = [d for d in draw10 if d["direction"] == "toward_draw"][:3]
    draw_suppress = [d for d in draw10 if d["direction"] == "away_from_draw"][:3]

    push_str = ", ".join(f"`{d['feature']}`" for d in draw_pushers) or "—"
    supp_str = ", ".join(f"`{d['feature']}`" for d in draw_suppress) or "—"

    top1 = top15[0] if top15 else {}
    convergent = top1.get("delta_rank", None) == 0

    return f"""\
## 17. SHAP Explainability (Phase 7e)

TreeExplainer SHAP values computed for XGBoost on the EPL 2024 test set
({data.get('n_test', 380)} matches, {data.get('n_features', 34)} features).

### Top 5 Features by Mean |SHAP| (all classes)

| SHAP Rank | Feature | Mean |SHAP| | Gain Rank | Δ Rank |
|-----------|---------|------------|-----------|--------|
{top5_table}

**Convergent validity:** `{top1.get('feature', '—')}` is rank 1 by both SHAP and XGBoost
gain — {'✓ confirmed' if convergent else '⚠ diverges'} across two independent importance
metrics. SHAP captures marginal contribution; gain captures split-quality frequency.
Their agreement on `{top1.get('feature', '—')}` validates it as the primary predictive driver.

### Draw SHAP Analysis

- **Pushers toward draw:** {push_str}
- **Suppressors of draw:** {supp_str}

`elo_diff` suppresses draws (large ELO gaps → decisive predicted outcomes).
H2H features (`h2h_draw_rate`, `h2h_home_win_rate`) are the primary draw promoters —
historical match context information unavailable from team-level aggregates alone.

**Notable rank divergence:** Several draw-rate form features (e.g. `away_form5_draw_rate`)
jump from low gain rank to high SHAP rank, because gain underweights features used
infrequently but with high marginal impact when applied.

---
"""


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Edge evaluation report from logged results."
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT,
        help=f"Output path for the Markdown report (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--reports-dir",
        type=pathlib.Path,
        default=_REPORTS_DIR,
        help=f"Directory containing JSON results files (default: {_REPORTS_DIR})",
    )
    args = parser.parse_args()

    tier1      = _load_tier1_results(args.reports_dir)
    comparison = _load_tier_comparison(args.reports_dir)
    features   = _parse_feature_ranking(args.reports_dir)

    has_comparison = comparison is not None

    sections: list[str] = [
        _section_header(has_comparison),
        _section_dataset(tier1),
        _section_experimental_setup(),
        _section_tier1_results(tier1),
    ]

    if has_comparison:
        sections += [
            _section_tier2_results(comparison),
            _section_bootstrap(comparison),
            _section_per_class(comparison),
            _section_betting_sim(comparison),
        ]
    else:
        sections.append(_section_pending_tier2())

    sections += [
        _section_feature_importance(features),
        _section_calibration(),
        _section_draw_limitations(),
        _section_future_work(),
        _section_reproducibility(),
        _section_tournament(args.reports_dir),
        _section_stacking(args.reports_dir),
        _section_dim_reduction(args.reports_dir),
        _section_unsupervised(args.reports_dir),
        _section_shap(args.reports_dir),
    ]

    report_text = "\n".join(sections)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")
    print(f"Report written to: {args.output}")

    if not has_comparison:
        print(
            "\nNOTE: Tier 2 comparison is absent — tier_comparison.json not found.\n"
            "Regenerate this report after completing lineup ingestion and Phase 5 training."
        )


if __name__ == "__main__":
    main()
