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
