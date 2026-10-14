"""Explainability & Clustering page — SHAP analysis and unsupervised findings."""

import json
import pathlib
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Explainability · LineupEdge",
    page_icon="🔎",
    layout="wide",
)
st.title("🔎 Explainability & Clustering")
st.caption(
    "Phase 7e: TreeExplainer SHAP values for XGBoost on the EPL 2024 test set. "
    "Phase 7d: K-Means team clustering and Ward hierarchical match-context clustering."
)

REPORTS = pathlib.Path(__file__).parent.parent.parent / "reports"


# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data
def load_shap() -> dict:
    path = REPORTS / "shap_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_unsupervised() -> dict:
    path = REPORTS / "unsupervised_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


shap_data    = load_shap()
unsup_data   = load_unsupervised()


# ── Section 1: SHAP Feature Importance ───────────────────────────────────────

st.header("SHAP Feature Importance")

if not shap_data:
    st.warning(
        "**shap_results.json not found.** "
        "Run `python models/explain.py` to generate it."
    )
else:
    top15 = shap_data.get("shap_top15", [])
    if top15:
        shap_df = pd.DataFrame(top15)[
            ["rank", "feature", "mean_abs_shap", "gain_rank", "delta_rank"]
        ].rename(columns={
            "rank":          "SHAP Rank",
            "feature":       "Feature",
            "mean_abs_shap": "Mean |SHAP|",
            "gain_rank":     "Gain Rank",
            "delta_rank":    "Δ Rank",
        })

        col_tbl, col_notes = st.columns([2, 1])
        with col_tbl:
            st.subheader("Top 15 features by mean |SHAP| (all classes)")
            st.dataframe(
                shap_df.style.format(
                    {"Mean |SHAP|": "{:.6f}"},
                    na_rep="—",
                ).highlight_max(subset=["Mean |SHAP|"], color="#dcfce7"),
                use_container_width=True,
                height=420,
            )
        with col_notes:
            top1 = top15[0]
            st.metric("Top SHAP feature",  top1["feature"])
            st.metric("Mean |SHAP|",       f"{top1['mean_abs_shap']:.4f}")
            st.metric("Gain rank matches?", "✅ Yes" if top1["delta_rank"] == 0 else "⚠ No")
            st.info(
                "**Convergent validity:** `elo_diff` is rank 1 by both SHAP "
                "and XGBoost gain — two independent importance metrics agree "
                "on the primary driver."
            )

    st.divider()
    col_bee, col_bar = st.columns(2)
    with col_bee:
        bee_path = REPORTS / "shap_beeswarm.png"
        if bee_path.exists():
            st.image(str(bee_path), caption="Beeswarm — Home Win class, EPL 2024 test set")
    with col_bar:
        bar_path = REPORTS / "shap_bar.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Mean |SHAP| across all classes")

    draw_path = REPORTS / "shap_draw_analysis.png"
    if draw_path.exists():
        st.image(str(draw_path), caption="Draw SHAP analysis — features pushing toward/away from draw")


# ── Section 2: Illustrative Match Predictions ─────────────────────────────────

st.divider()
st.header("Illustrative Match Predictions")

if shap_data:
    ill = shap_data.get("illustrative_matches", {})

    _LABELS = {
        "high_conf_correct": ("✅ High-Confidence Correct", "shap_waterfall_correct.png"),
        "high_conf_wrong":   ("❌ High-Confidence Wrong",   "shap_waterfall_wrong.png"),
        "near_draw":         ("⚖️ Most Uncertain",          "shap_waterfall_draw.png"),
    }

    cols = st.columns(3)
    for col, (key, (title, png_name)) in zip(cols, _LABELS.items()):
        match = ill.get(key, {})
        with col:
            st.subheader(title)
            if match:
                predicted = match.get("predicted", "—")
                actual    = match.get("actual", "—")
                max_prob  = match.get("max_prob", 0.0)
                correct   = predicted == actual

                st.metric("Predicted", predicted)
                col_act, col_p = st.columns(2)
                col_act.metric("Actual",   actual)
                col_p.metric("Max P",      f"{max_prob:.3f}")

                if correct:
                    st.success("Prediction correct")
                else:
                    st.error("Prediction wrong")

                wf_path = REPORTS / png_name
                if wf_path.exists():
                    st.image(str(wf_path), use_container_width=True)


# ── Section 3: Team Clustering ────────────────────────────────────────────────

st.divider()
st.header("Team Clustering (K-Means)")

if not unsup_data:
    st.warning(
        "**unsupervised_results.json not found.** "
        "Run `python models/unsupervised.py` to generate it."
    )
else:
    kmeans = unsup_data.get("kmeans", {})

    col_img, col_tbl = st.columns([3, 2])
    with col_img:
        km_path = REPORTS / "kmeans_clusters.png"
        if km_path.exists():
            st.image(str(km_path), caption="PCA 2D projection — K-Means team clusters")

    with col_tbl:
        best_k  = kmeans.get("best_k", "—")
        sil     = kmeans.get("silhouette_scores", {})
        best_sil = sil.get(str(best_k), "—")

        st.metric("Best k", best_k)
        st.metric("Silhouette score", f"{best_sil:.3f}" if isinstance(best_sil, float) else best_sil)

        clusters_raw = kmeans.get("clusters", {})
        if clusters_raw:
            cluster_rows = []
            for c_id, info in sorted(
                clusters_raw.items(), key=lambda x: -x[1]["mean_elo"]
            ):
                cluster_rows.append({
                    "Cluster": c_id,
                    "Size":    info["size"],
                    "ELO":     round(info["mean_elo"], 0),
                    "Home W%": f"{info['home_win_rate']*100:.1f}%",
                    "Away W%": f"{info['away_win_rate']*100:.1f}%",
                })
            st.dataframe(
                pd.DataFrame(cluster_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Cluster ELO bands: ~1635 (elite top-6), ~1542 (upper-mid), "
            "~1477 (mid-table), ~1430 (relegation zone)."
        )


# ── Section 4: Match Context Clustering ──────────────────────────────────────

st.divider()
st.header("Match Context Clustering (Ward Hierarchical)")

if unsup_data:
    hier = unsup_data.get("hierarchical", {})

    col_dend, col_hier = st.columns([3, 2])
    with col_dend:
        dend_path = REPORTS / "dendrogram.png"
        if dend_path.exists():
            st.image(str(dend_path), caption="Ward linkage dendrogram (truncated to last 30 merges)")

    with col_hier:
        st.metric("Total matches", hier.get("n_matches", "—"))
        st.metric("Clusters", hier.get("n_clusters", 4))

        hier_clusters = hier.get("clusters", {})
        if hier_clusters:
            hier_rows = []
            for c_id, info in sorted(
                hier_clusters.items(),
                key=lambda x: -x[1]["mean_elo_diff"],
            ):
                op = info.get("outcome_pct", {})
                hier_rows.append({
                    "Cluster": c_id,
                    "Label":   info.get("label", "—"),
                    "Size":    info["size"],
                    "ELO Δ":   round(info["mean_elo_diff"], 1),
                    "H%":      op.get("H", "—"),
                    "D%":      op.get("D", "—"),
                    "A%":      op.get("A", "—"),
                })
            st.dataframe(
                pd.DataFrame(hier_rows),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Away Favoured (n=543) is the largest cluster — "
            "most EPL matches feature a stronger visiting side."
        )
