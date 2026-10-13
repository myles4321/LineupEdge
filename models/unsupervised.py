"""
Phase 7d — Unsupervised Analysis.

Two experiments on the Tier 1 feature matrix (all 3 seasons, 1,140 matches):

  1. K-Means Team Clustering — group EPL teams by playing-style features
     (ELO, home/away win rates, PPG form); best k chosen by silhouette score;
     visualised in PCA 2D space annotated with team IDs.

  2. Hierarchical Match-Context Clustering — Ward linkage on 4 match-level
     features (ELO diff, points diff, home/away form); dendrogram + 4-cluster
     cut; outcome distribution (H/D/A %) per cluster.

Usage:
    python models/unsupervised.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # headless — no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_REPORTS_DIR  = pathlib.Path("reports")
_FEATURES_CSV = pathlib.Path("data/processed/tier1_features.csv")

_KMEANS_CANDIDATE_K = [3, 4, 5, 6]
_HIER_N_CLUSTERS    = 4
_HIER_FEATURES      = ["elo_diff", "points_diff", "home_form5_win_rate", "away_form5_win_rate"]
_KMEANS_FEATURES    = ["mean_elo", "home_win_rate", "away_win_rate", "mean_home_ppg", "mean_away_ppg"]


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_full_dataset() -> pd.DataFrame:
    """Load all 1,140 rows from tier1_features.csv."""
    df = pd.read_csv(_FEATURES_CSV)
    logger.info("Loaded %d matches × %d columns from %s", len(df), len(df.columns), _FEATURES_CSV)
    return df


# ── Experiment 1: K-Means Team Clustering ─────────────────────────────────────


def _build_team_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-team features across all matches (home and away appearances).

    Returns a DataFrame with one row per unique team and these columns:
        mean_elo, home_win_rate, away_win_rate, mean_home_ppg, mean_away_ppg
    Index is the team_id (integer API-Football ID).
    """
    # Home appearances: each row where team is the home side
    home = df[["home_team_id", "home_elo", "home_ppg_last5", "result"]].copy()
    home.columns = ["team_id", "elo", "ppg_last5", "result"]
    home["is_home"] = True

    # Away appearances: each row where team is the away side
    away = df[["away_team_id", "away_elo", "away_ppg_last5", "result"]].copy()
    away.columns = ["team_id", "elo", "ppg_last5", "result"]
    away["is_home"] = False

    appearances = pd.concat([home, away], ignore_index=True)

    records = []
    for team_id, grp in appearances.groupby("team_id"):
        home_games = grp[grp["is_home"]]
        away_games = grp[~grp["is_home"]]

        mean_elo = grp["elo"].mean()

        # Actual win rates from results
        home_win_rate = (
            (home_games["result"] == "H").sum() / len(home_games)
            if len(home_games) > 0 else np.nan
        )
        away_win_rate = (
            (away_games["result"] == "A").sum() / len(away_games)
            if len(away_games) > 0 else np.nan
        )

        mean_home_ppg = home_games["ppg_last5"].mean()
        mean_away_ppg = away_games["ppg_last5"].mean()

        records.append({
            "team_id":      team_id,
            "mean_elo":     mean_elo,
            "home_win_rate": home_win_rate,
            "away_win_rate": away_win_rate,
            "mean_home_ppg": mean_home_ppg,
            "mean_away_ppg": mean_away_ppg,
        })

    team_df = pd.DataFrame(records).set_index("team_id")
    team_df = team_df.dropna()  # drop any team missing all features (shouldn't occur)
    logger.info("Team feature matrix: %d teams × %d features", len(team_df), len(team_df.columns))
    return team_df


def run_kmeans_experiment(df: pd.DataFrame) -> dict:
    """
    K-Means team clustering with silhouette-based k selection and PCA 2D visualisation.
    """
    logger.info("=== Experiment 1: K-Means Team Clustering ===")

    team_df = _build_team_feature_matrix(df)
    n_teams = len(team_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(team_df.values)

    # ── Select best k by silhouette score ─────────────────────────────────────
    silhouette_scores: dict[int, float] = {}
    for k in _KMEANS_CANDIDATE_K:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        score  = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = round(float(score), 4)
        logger.info("  k=%d  silhouette=%.4f", k, score)

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    logger.info("Best k=%d (silhouette=%.4f)", best_k, silhouette_scores[best_k])

    # ── Final KMeans with best k ───────────────────────────────────────────────
    km_final   = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    cluster_labels = km_final.fit_predict(X_scaled)
    team_df["cluster"] = cluster_labels

    # ── PCA 2D projection for visualisation ───────────────────────────────────
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_scaled)
    team_df["pc1"] = X_pca[:, 0]
    team_df["pc2"] = X_pca[:, 1]

    # ── Scatter plot ───────────────────────────────────────────────────────────
    cluster_palette = ["steelblue", "darkorange", "crimson", "seagreen", "purple", "goldenrod"]
    fig, ax = plt.subplots(figsize=(10, 7))

    for c in range(best_k):
        mask = team_df["cluster"] == c
        ax.scatter(
            team_df.loc[mask, "pc1"],
            team_df.loc[mask, "pc2"],
            color=cluster_palette[c % len(cluster_palette)],
            s=120, zorder=3, label=f"Cluster {c}",
        )

    # Annotate each point with team ID
    for team_id, row in team_df.iterrows():
        ax.annotate(
            str(team_id),
            xy=(row["pc1"], row["pc2"]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, color="dimgray",
        )

    ax.set_xlabel(
        f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11,
    )
    ax.set_ylabel(
        f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11,
    )
    ax.set_title(
        f"K-Means Team Clusters (k={best_k}) — EPL 2022–2024\n"
        "PCA 2D projection of 5 playing-style features",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    kmeans_path = _REPORTS_DIR / "kmeans_clusters.png"
    fig.savefig(kmeans_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("K-Means cluster plot saved to %s", kmeans_path)

    # ── Cluster profiles ───────────────────────────────────────────────────────
    profile_cols = ["mean_elo", "home_win_rate", "away_win_rate", "mean_home_ppg", "mean_away_ppg"]
    clusters_json: dict[str, dict] = {}

    cluster_profiles = (
        team_df.groupby("cluster")[profile_cols]
        .mean()
        .sort_values("mean_elo", ascending=False)
    )
    for c_id, row in cluster_profiles.iterrows():
        teams_in_cluster = team_df[team_df["cluster"] == c_id].index.tolist()
        clusters_json[str(c_id)] = {
            "size":          int((team_df["cluster"] == c_id).sum()),
            "mean_elo":      round(float(row["mean_elo"]), 1),
            "home_win_rate": round(float(row["home_win_rate"]), 4),
            "away_win_rate": round(float(row["away_win_rate"]), 4),
            "mean_home_ppg": round(float(row["mean_home_ppg"]), 4),
            "mean_away_ppg": round(float(row["mean_away_ppg"]), 4),
            "teams":         [int(t) for t in sorted(teams_in_cluster)],
        }

    return {
        "n_teams":       n_teams,
        "features_used": _KMEANS_FEATURES,
        "silhouette_scores": {str(k): v for k, v in silhouette_scores.items()},
        "best_k":        best_k,
        "clusters":      clusters_json,
    }


# ── Experiment 2: Hierarchical Match-Context Clustering ───────────────────────


def _assign_cluster_label(
    mean_elo_diff: float,
    mean_pts_diff: float,
    home_pct: float,
    away_pct: float,
) -> str:
    """
    Assign an interpretive label to a match-context cluster based on its
    mean feature values and observed outcome distribution.
    """
    if mean_elo_diff > 150 and home_pct > 0.55:
        return "Strong Home"
    if mean_elo_diff > 50 and home_pct > 0.50:
        return "Home Favoured"
    if mean_elo_diff < -50 and away_pct > 0.35:
        return "Away Favoured"
    if abs(mean_elo_diff) < 30 and abs(mean_pts_diff) < 3:
        return "Balanced Contest"
    return "Moderate Home Edge"


def run_hierarchical_experiment(df: pd.DataFrame) -> dict:
    """
    Ward-linkage hierarchical clustering of match contexts; dendrogram + 4-cluster cut.
    """
    logger.info("=== Experiment 2: Hierarchical Match-Context Clustering ===")

    # ── Prepare match-level features ──────────────────────────────────────────
    match_df = df[_HIER_FEATURES + ["result"]].dropna(subset=_HIER_FEATURES).copy()
    match_df = match_df.reset_index(drop=True)
    n_matches = len(match_df)
    logger.info("Match context matrix: %d matches (dropped %d NaN rows)", n_matches, len(df) - n_matches)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(match_df[_HIER_FEATURES].values)

    # ── Ward linkage ──────────────────────────────────────────────────────────
    Z = linkage(X_scaled, method="ward")

    # ── Dendrogram (truncated to last 30 merges) ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
        ax=ax,
    )
    ax.set_title(
        "Ward Hierarchical Clustering — Match Contexts\n"
        f"(truncated to last 30 merges, {n_matches} total matches)",
        fontsize=11,
    )
    ax.set_xlabel("Match Cluster / (count)", fontsize=10)
    ax.set_ylabel("Ward Linkage Distance", fontsize=10)

    # Mark the cut at 4 clusters with a horizontal line at the 4th-from-top merge
    cut_distance = (Z[-3, 2] + Z[-4, 2]) / 2
    ax.axhline(cut_distance, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Cut → {_HIER_N_CLUSTERS} clusters")
    ax.legend(fontsize=9)
    plt.tight_layout()
    dendro_path = _REPORTS_DIR / "dendrogram.png"
    fig.savefig(dendro_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Dendrogram saved to %s", dendro_path)

    # ── Cut tree to 4 clusters ────────────────────────────────────────────────
    cluster_labels = fcluster(Z, t=_HIER_N_CLUSTERS, criterion="maxclust")
    match_df["cluster"] = cluster_labels

    # ── Profile each cluster ───────────────────────────────────────────────────
    clusters_json: dict[str, dict] = {}
    for c_id in sorted(match_df["cluster"].unique()):
        grp = match_df[match_df["cluster"] == c_id]

        home_pct  = round(float((grp["result"] == "H").mean() * 100), 1)
        draw_pct  = round(float((grp["result"] == "D").mean() * 100), 1)
        away_pct  = round(float((grp["result"] == "A").mean() * 100), 1)

        mean_elo_diff  = float(grp["elo_diff"].mean())
        mean_pts_diff  = float(grp["points_diff"].mean())
        h_form         = float(grp["home_form5_win_rate"].mean())
        a_form         = float(grp["away_form5_win_rate"].mean())

        label = _assign_cluster_label(mean_elo_diff, mean_pts_diff, home_pct / 100, away_pct / 100)

        clusters_json[str(c_id)] = {
            "size":             int(len(grp)),
            "mean_elo_diff":    round(mean_elo_diff, 1),
            "mean_points_diff": round(mean_pts_diff, 1),
            "home_win_rate":     round(h_form, 4),
            "away_win_rate":     round(a_form, 4),
            "outcome_pct":      {"H": home_pct, "D": draw_pct, "A": away_pct},
            "label":            label,
        }

    return {
        "n_matches":     n_matches,
        "features_used": _HIER_FEATURES,
        "n_clusters":    _HIER_N_CLUSTERS,
        "clusters":      clusters_json,
    }


# ── Summary printing ───────────────────────────────────────────────────────────


def _print_summary(kmeans_results: dict, hier_results: dict) -> None:
    sep  = "=" * 66
    dash = "-" * 66

    print(f"\n{sep}")
    print(f"  PHASE 7d UNSUPERVISED ANALYSIS — EPL 2022-2024")
    print(f"{sep}")

    # K-Means summary
    print("\nK-MEANS TEAM CLUSTERING")
    sil = kmeans_results["silhouette_scores"]
    sil_str = "  ".join(f"k={k}: {sil[str(k)]:.3f}" for k in _KMEANS_CANDIDATE_K)
    print(f"  Silhouette scores: {sil_str}")
    print(f"  Best k: {kmeans_results['best_k']} (silhouette={sil[str(kmeans_results['best_k'])]:.3f})")
    print()
    print(f"  {'Cluster':>7}  {'Size':>4}  {'Mean ELO':>8}  {'Home W%':>7}  {'Away W%':>7}  {'Home PPG':>8}  {'Away PPG':>8}")
    print(f"  {dash}")
    for c_id, info in sorted(
        kmeans_results["clusters"].items(),
        key=lambda x: -x[1]["mean_elo"],
    ):
        print(
            f"  {c_id:>7}  {info['size']:>4}  {info['mean_elo']:>8.0f}"
            f"  {info['home_win_rate']*100:>6.1f}%  {info['away_win_rate']*100:>6.1f}%"
            f"  {info['mean_home_ppg']:>8.3f}  {info['mean_away_ppg']:>8.3f}"
        )

    # Hierarchical summary
    n = hier_results["n_matches"]
    print(f"\nHIERARCHICAL MATCH-CONTEXT CLUSTERING (n={n} matches, {_HIER_N_CLUSTERS} clusters)")
    print(
        f"  {'Cluster':>7}  {'Size':>5}  {'ELO diff':>8}  {'Pts diff':>8}"
        f"  {'H-form':>6}  {'A-form':>6}  {'H%':>5}  {'D%':>5}  {'A%':>5}  Label"
    )
    print(f"  {dash}")
    for c_id, info in sorted(hier_results["clusters"].items(), key=lambda x: -x[1]["mean_elo_diff"]):
        op = info["outcome_pct"]
        print(
            f"  {c_id:>7}  {info['size']:>5}  {info['mean_elo_diff']:>8.1f}"
            f"  {info['mean_points_diff']:>8.1f}"
            f"  {info['home_win_rate']:>6.3f}  {info['away_win_rate']:>6.3f}"
            f"  {op['H']:>5.1f}  {op['D']:>5.1f}  {op['A']:>5.1f}"
            f"  {info['label']}"
        )

    print(f"\n{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    _REPORTS_DIR.mkdir(exist_ok=True)

    df = _load_full_dataset()

    kmeans_results = run_kmeans_experiment(df)
    hier_results   = run_hierarchical_experiment(df)

    _print_summary(kmeans_results, hier_results)

    results = {
        "kmeans":       kmeans_results,
        "hierarchical": hier_results,
    }
    json_path = _REPORTS_DIR / "unsupervised_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to %s", json_path)
    logger.info("Phase 7d complete.")


if __name__ == "__main__":
    main()
