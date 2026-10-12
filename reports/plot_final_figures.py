"""
Phase 6 — Publication-Quality Figure Generator.

Generates three final figures for the evaluation report:

  1. reports/final_calibration.png
     Reliability diagrams for all three outcome classes (H / D / A).
     Shows T1 XGBoost, T2 XGBoost (if predictions exist), and Bookmaker B365
     (if odds available). Requires database access — skip with --no-db.

  2. reports/final_metrics_comparison.png
     Grouped bar chart of log loss and Brier score across all models.
     Reads from reports/tier1_results.json and reports/tier_comparison.json
     (if present). No database required.

  3. reports/final_feature_importance.png
     Horizontal bar chart of the top-20 Tier 1 features ranked by combined
     univariate score (ANOVA F + mutual information).
     Reads from reports/feature_analysis.txt. No database required.

Usage:
    python reports/plot_final_figures.py           # all three figures
    python reports/plot_final_figures.py --no-db   # skip calibration figure
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPORTS_DIR = pathlib.Path("reports")

# ── Publication-quality rcParams ──────────────────────────────────────────────

_PUB_RC = {
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.dpi":        150,   # screen preview — saved at 300 dpi
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
}

# Canonical label order: index 0=H, 1=D, 2=A
_LABEL_ORDER  = ["H", "D", "A"]
_LABEL_TO_IDX = {label: i for i, label in enumerate(_LABEL_ORDER)}
_CLASS_NAMES  = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

# Consistent colour scheme across all figures
_COLOURS = {
    "T1 LogReg":      "#6366f1",  # indigo
    "T1 XGBoost":     "#2563eb",  # blue
    "T2 LogReg":      "#f97316",  # orange
    "T2 XGBoost":     "#e85d04",  # dark orange
    "Bookmaker B365": "#dc2626",  # red
}

# Model version tags — must match train_tier1.py / train_tier2.py
_T1_XGB_VERSION    = "tier1_xgb_v1"
_T2_XGB_VERSION    = "tier2_xgb_v1"
_TEST_SEASON       = 2024
_CALIBRATION_BINS  = 8


# ── Database loader ───────────────────────────────────────────────────────────


def _build_engine():
    """Create SQLAlchemy engine from .env / environment variables.

    Returns None if environment is not configured.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        db_user = os.environ["DB_USER"]
        db_password = os.getenv("DB_PASSWORD", "")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "edge")

        from sqlalchemy import create_engine, text as sa_text
        url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(url)
        # Verify connectivity with a lightweight query
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        return engine
    except Exception as exc:
        print(f"[WARNING] Cannot connect to database: {exc}", file=sys.stderr)
        print("[WARNING] Calibration figure will be skipped.", file=sys.stderr)
        return None


def _load_predictions_with_results(engine, model_version: str) -> pd.DataFrame:
    """Load predictions joined to match results for the 2024 test season.

    Returns a DataFrame with columns:
        match_id, prob_home, prob_draw, prob_away, result
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT p.match_id,
                       p.prob_home, p.prob_draw, p.prob_away,
                       m.result
                FROM predictions p
                JOIN matches m ON p.match_id = m.match_id
                WHERE p.model_version = :version
                  AND m.season        = :season
                  AND m.status        = 'FT'
                  AND m.result IS NOT NULL
                ORDER BY p.match_id ASC
            """),
            conn,
            params={"version": model_version, "season": _TEST_SEASON},
        )
    return df


def _load_b365_implied(engine, match_ids: list[int]) -> pd.DataFrame:
    """Load B365 vig-normalised implied probabilities.

    Returns columns: match_id, implied_home, implied_draw, implied_away, result.
    """
    from sqlalchemy import text

    if not match_ids:
        return pd.DataFrame()

    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT o.match_id,
                       o.implied_home, o.implied_draw, o.implied_away,
                       m.result
                FROM odds o
                JOIN matches m ON o.match_id = m.match_id
                WHERE o.match_id IN :ids
                  AND o.bookmaker  = 'B365'
                  AND m.season     = :season
                  AND m.status     = 'FT'
            """),
            conn,
            params={"ids": tuple(match_ids), "season": _TEST_SEASON},
        )
    return df


# ── JSON loaders ──────────────────────────────────────────────────────────────


def _load_tier1_results() -> dict | None:
    path = _REPORTS_DIR / "tier1_results.json"
    if not path.exists():
        print(f"[WARNING] {path} not found — metrics comparison figure skipped.",
              file=sys.stderr)
        return None
    return json.loads(path.read_text())


def _load_tier_comparison() -> dict | None:
    path = _REPORTS_DIR / "tier_comparison.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _parse_feature_ranking() -> list[dict]:
    path = _REPORTS_DIR / "feature_analysis.txt"
    if not path.exists():
        print(f"[WARNING] {path} not found — feature importance figure skipped.",
              file=sys.stderr)
        return []

    features: list[dict] = []
    for line in path.read_text().splitlines():
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


# ── Feature group colouring ───────────────────────────────────────────────────

def _feature_colour(name: str) -> str:
    """Assign a colour to a feature based on its group."""
    if "elo" in name:
        return "#2563eb"   # blue — ELO
    if "form" in name:
        return "#16a34a"   # green — rolling form
    if "xg" in name:
        return "#9ca3af"   # grey — xG (currently null)
    if "position" in name or "points" in name or "goal_diff" in name or "ppg" in name:
        return "#7c3aed"   # purple — league table
    if "rest" in name:
        return "#d97706"   # amber — rest days
    if "h2h" in name:
        return "#be185d"   # pink — head-to-head
    if "home_win" in name or "away_win" in name:
        return "#0891b2"   # teal — venue splits
    return "#6b7280"       # default grey


# ── Figure 1: Publication-quality calibration curves ─────────────────────────


def _plot_final_calibration(engine) -> pathlib.Path | None:
    """Generate publication-quality reliability diagrams for all three classes.

    Plots T1 XGBoost, T2 XGBoost (if available), and Bookmaker B365 (if available).
    Returns the output path, or None if no predictions are available.
    """
    print("Loading predictions for calibration figure...")

    # T1 XGBoost (required)
    t1_df = _load_predictions_with_results(engine, _T1_XGB_VERSION)
    if t1_df.empty:
        print(
            f"[WARNING] No predictions found for {_T1_XGB_VERSION} — "
            "calibration figure skipped.",
            file=sys.stderr,
        )
        return None

    # T2 XGBoost (optional)
    t2_df = _load_predictions_with_results(engine, _T2_XGB_VERSION)

    # B365 implied (optional)
    bm_df = _load_b365_implied(engine, t1_df["match_id"].tolist())

    # Build model dict: {display_name: (y_true_str, proba_array)}
    model_data: dict[str, tuple[pd.Series, np.ndarray]] = {
        "Tier 1 XGBoost": (
            t1_df["result"],
            t1_df[["prob_home", "prob_draw", "prob_away"]].values.astype(float),
        ),
    }

    if not t2_df.empty:
        # Inner-join on match_id so row counts match
        aligned = t1_df[["match_id", "result"]].merge(
            t2_df.rename(columns={
                "prob_home": "t2_home",
                "prob_draw": "t2_draw",
                "prob_away": "t2_away",
            }),
            on="match_id",
            how="inner",
        ).sort_values("match_id").reset_index(drop=True)

        model_data["Tier 2 XGBoost"] = (
            aligned["result"],
            aligned[["t2_home", "t2_draw", "t2_away"]].values.astype(float),
        )
        # Use the aligned subset for T1 as well so rows match
        t1_aligned_proba = t1_df.set_index("match_id").loc[
            aligned["match_id"],
            ["prob_home", "prob_draw", "prob_away"],
        ].values.astype(float)
        model_data["Tier 1 XGBoost"] = (
            aligned["result"],
            t1_aligned_proba,
        )

    if not bm_df.empty:
        bm_clean = bm_df.dropna(
            subset=["implied_home", "implied_draw", "implied_away"]
        ).reset_index(drop=True)
        if not bm_clean.empty:
            model_data["Bookmaker B365"] = (
                bm_clean["result"],
                bm_clean[["implied_home", "implied_draw", "implied_away"]].values.astype(float),
            )

    # ── Plot ──────────────────────────────────────────────────────────────────
    with plt.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        for ax_idx, class_label in enumerate(_LABEL_ORDER):
            ax = axes[ax_idx]
            class_idx = _LABEL_TO_IDX[class_label]

            for model_name, (y_true_str, proba) in model_data.items():
                y_binary   = (y_true_str == class_label).astype(float).values
                class_prob = proba[:, class_idx]

                try:
                    frac_pos, mean_pred = calibration_curve(
                        y_binary, class_prob,
                        n_bins=_CALIBRATION_BINS,
                        strategy="uniform",
                    )
                except ValueError:
                    continue

                colour = _COLOURS.get(model_name, "#374151")
                ax.plot(
                    mean_pred, frac_pos,
                    marker="o", markersize=5,
                    linewidth=2.0,
                    label=model_name,
                    color=colour,
                )

            # Perfect calibration reference
            ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.6,
                    label="Perfect calibration", zorder=0)

            ax.set_title(f"Outcome: {_CLASS_NAMES[class_label]}", fontsize=13)
            ax.set_xlabel("Mean predicted probability")
            if ax_idx == 0:
                ax.set_ylabel("Fraction of positives")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.legend(loc="upper left", fontsize=9)

        n_matches = len(next(iter(model_data.values()))[0])
        fig.suptitle(
            f"Reliability Diagrams — 2024 Test Season ({n_matches} matches)",
            fontsize=15, y=1.02,
        )
        plt.tight_layout()

        out = _REPORTS_DIR / "final_calibration.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved: {out}")
    return out


# ── Figure 2: Metrics comparison bar chart ────────────────────────────────────


def _plot_final_metrics_comparison() -> pathlib.Path | None:
    """Grouped bar chart of log loss and Brier score across all available models."""
    tier1      = _load_tier1_results()
    comparison = _load_tier_comparison()

    if tier1 is None:
        return None

    # Build an ordered list of (display_name, log_loss, brier) tuples
    models: list[tuple[str, float, float]] = []

    # Tier 1 results from tier1_results.json
    t1_results = tier1["results"]
    for r in t1_results:
        if "Bookmaker" in r["model"]:
            continue  # add Bookmaker last
        if "Logistic" in r["model"]:
            models.append(("T1 LogReg", r["log_loss"], r["brier"]))
        elif "XGBoost" in r["model"]:
            models.append(("T1 XGBoost", r["log_loss"], r["brier"]))

    # Tier 2 results from tier_comparison.json (if available)
    if comparison:
        # tier_comparison.json stores T1 XGB + T2 XGB by default.
        # If logreg_metrics were appended, they would also appear in the JSON.
        models.append((
            "T2 XGBoost",
            comparison["tier2_xgb"]["log_loss"],
            comparison["tier2_xgb"]["brier"],
        ))

    # Bookmaker — from tier1_results.json if not in comparison
    bm_source = comparison.get("bookmaker") if comparison else None
    if bm_source is None:
        bm_source = next(
            (r for r in t1_results if "Bookmaker" in r["model"]), None
        )
    if bm_source:
        models.append(("Bookmaker B365", bm_source["log_loss"], bm_source["brier"]))

    names    = [m[0] for m in models]
    ll_vals  = [m[1] for m in models]
    br_vals  = [m[2] for m in models]
    colours  = [_COLOURS.get(n, "#6b7280") for n in names]

    x = np.arange(len(names))

    with plt.rc_context(_PUB_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # ── Left: Log Loss ────────────────────────────────────────────────────
        bars1 = ax1.bar(x, ll_vals, color=colours, alpha=0.87, width=0.6,
                        edgecolor="white", linewidth=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=20, ha="right")
        ax1.set_title("Log Loss — 2024 Test Season\n(lower is better)", fontsize=13)
        ax1.set_ylabel("Log Loss")

        # Random baseline reference line (log(3))
        random_ll = float(np.log(3))
        ax1.axhline(random_ll, color="#6b7280", linestyle=":", linewidth=1.2,
                    label=f"Random baseline ({random_ll:.3f})")
        ax1.legend(fontsize=9)

        for bar, val in zip(bars1, ll_vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

        # ── Right: Brier Score ────────────────────────────────────────────────
        bars2 = ax2.bar(x, br_vals, color=colours, alpha=0.87, width=0.6,
                        edgecolor="white", linewidth=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=20, ha="right")
        ax2.set_title("Brier Score — 2024 Test Season\n(lower is better)", fontsize=13)
        ax2.set_ylabel("Brier Score")

        # Random baseline (1/3 per class → mean Brier ≈ 2/9 ≈ 0.222)
        random_brier = 2.0 / 9.0
        ax2.axhline(random_brier, color="#6b7280", linestyle=":", linewidth=1.2,
                    label=f"Random baseline ({random_brier:.3f})")
        ax2.legend(fontsize=9)

        for bar, val in zip(bars2, br_vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9,
            )

        fig.suptitle("Model Comparison — Log Loss and Brier Score", fontsize=15)
        plt.tight_layout()

        out = _REPORTS_DIR / "final_metrics_comparison.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved: {out}")
    return out


# ── Figure 3: Feature importance horizontal bar chart ─────────────────────────


def _plot_final_feature_importance() -> pathlib.Path | None:
    """Horizontal bar chart of top-20 Tier 1 features by combined univariate score."""
    features = _parse_feature_ranking()
    if not features:
        return None

    # Top 20, excluding 100%-null features, sorted by combined score
    top = sorted(
        [f for f in features if f["null_pct"] < 100],
        key=lambda f: f["combined"],
        reverse=True,
    )[:20]

    names    = [f["name"] for f in top]
    scores   = [f["combined"] for f in top]
    colours  = [_feature_colour(n) for n in names]

    # Reverse so highest is at top of chart
    names_rev   = names[::-1]
    scores_rev  = scores[::-1]
    colours_rev = colours[::-1]

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(names_rev))
        bars  = ax.barh(y_pos, scores_rev, color=colours_rev,
                        alpha=0.85, edgecolor="white", linewidth=0.8, height=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names_rev, fontsize=10)
        ax.set_xlabel("Combined Score (normalised ANOVA F + Mutual Information)", fontsize=11)
        ax.set_title(
            "Tier 1 Feature Importance — Univariate Ranking\n"
            "(combined = mean of normalised ANOVA F-statistic and mutual information)",
            fontsize=13,
        )
        ax.set_xlim(0, 1.08)

        # Value labels on bars
        for bar, val in zip(bars, scores_rev):
            ax.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center", fontsize=8.5,
            )

        # Legend for feature groups
        legend_elements = [
            Patch(facecolor="#2563eb", label="ELO ratings"),
            Patch(facecolor="#16a34a", label="Rolling form"),
            Patch(facecolor="#7c3aed", label="League table / PPG"),
            Patch(facecolor="#d97706", label="Rest days"),
            Patch(facecolor="#be185d", label="Head-to-head"),
            Patch(facecolor="#0891b2", label="Venue splits"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            fontsize=9,
            title="Feature group",
            title_fontsize=9,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()

        out = _REPORTS_DIR / "final_feature_importance.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality evaluation figures for the Edge project."
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help=(
            "Skip figures that require database access "
            "(final_calibration.png). "
            "Still generates final_metrics_comparison.png and "
            "final_feature_importance.png from local JSON/text files."
        ),
    )
    args = parser.parse_args()

    _REPORTS_DIR.mkdir(exist_ok=True)

    # ── Figures from local files (no DB required) ─────────────────────────────
    print("\n--- Feature importance figure ---")
    _plot_final_feature_importance()

    print("\n--- Metrics comparison figure ---")
    _plot_final_metrics_comparison()

    # ── Database-dependent figure ─────────────────────────────────────────────
    if args.no_db:
        print(
            "\nSkipping calibration figure (--no-db). "
            "Run without --no-db when the database is available."
        )
    else:
        print("\n--- Calibration figure ---")
        engine = _build_engine()
        if engine is not None:
            _plot_final_calibration(engine)
        else:
            print(
                "[INFO] Calibration figure skipped — "
                "use --no-db to suppress this message.",
                file=sys.stderr,
            )

    print("\nDone. Figures saved to reports/")


if __name__ == "__main__":
    main()
