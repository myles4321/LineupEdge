"""Model Tournament page — Phase 7a ranked comparison across 10 classifiers."""

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Model Tournament · LineupEdge",
    page_icon="🏆",
    layout="wide",
)
st.title("🏆 Model Tournament")
st.caption(
    "Phase 7a: 10 classifiers evaluated on the EPL 2024 test season (380 matches). "
    "All models trained on 2022+2023, tuned on 2023 validation season."
)

REPORTS = pathlib.Path(__file__).parent.parent.parent / "reports"

# ── Display name mapping ──────────────────────────────────────────────────────

_MODEL_NAMES = {
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

_FAILED_MODELS = {"Gaussian NB", "QDA"}   # catastrophic log loss (>2.0)


# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data
def load_tournament() -> dict:
    path = REPORTS / "tournament_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


@st.cache_data
def load_stack() -> dict:
    path = REPORTS / "stack_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


tournament = load_tournament()
stack      = load_stack()

if not tournament:
    st.warning(
        "**tournament_results.json not found.** "
        "Run `python models/tournament.py` to generate it."
    )
    st.stop()


# ── Build rankings DataFrame ──────────────────────────────────────────────────


def _build_rankings(results: list[dict], bookmaker: dict, stack_data: dict) -> pd.DataFrame:
    rows = []
    for r in sorted(results, key=lambda x: x["log_loss"]):
        display_name = _MODEL_NAMES.get(r["model_version"], r["model_version"])
        rows.append({
            "Model":      display_name,
            "Log Loss":   round(r["log_loss"], 4),
            "Brier":      round(r["brier_score"], 4),
            "Accuracy %": round(r["accuracy"] * 100, 2),
            "AUC":        round(r["roc_auc_macro"], 4),
            "CV LL":      round(r["cv_mean_log_loss"], 4),
            "CV σ":       round(r["cv_std_log_loss"], 4),
            "_is_failed": display_name in _FAILED_MODELS,
        })

    if stack_data:
        rows.append({
            "Model":      "Super Learner (Stack)",
            "Log Loss":   stack_data["stack"]["log_loss"],
            "Brier":      stack_data["stack"].get("brier_score", float("nan")),
            "Accuracy %": round(stack_data["stack"].get("accuracy", float("nan")) * 100, 2),
            "AUC":        float("nan"),
            "CV LL":      float("nan"),
            "CV σ":       float("nan"),
            "_is_failed": False,
        })

    bm = bookmaker
    rows.append({
        "Model":      "Bookmaker B365 ★",
        "Log Loss":   round(bm["log_loss"], 4),
        "Brier":      round(bm["brier_score"], 4),
        "Accuracy %": round(bm["accuracy"] * 100, 2),
        "AUC":        round(bm["roc_auc_macro"], 4),
        "CV LL":      float("nan"),
        "CV σ":       float("nan"),
        "_is_failed": False,
    })

    return pd.DataFrame(rows)


rankings_df = _build_rankings(
    tournament["results"],
    tournament["bookmaker"],
    stack,
)

tab1, tab2, tab3 = st.tabs(["📊 Rankings", "📈 Charts", "🔁 CV Results"])


# ── Tab 1: Rankings table ─────────────────────────────────────────────────────

with tab1:
    best_model  = rankings_df[~rankings_df["Model"].str.contains("B365|Stack")].iloc[0]
    bm_ll       = rankings_df[rankings_df["Model"].str.contains("B365")].iloc[0]["Log Loss"]
    gap_to_bm   = round(float(best_model["Log Loss"]) - bm_ll, 4)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model",      best_model["Model"])
    c2.metric("Best log loss",   f"{best_model['Log Loss']:.4f}")
    c3.metric("Bookmaker B365",  f"{bm_ll:.4f}")
    c4.metric("Gap to bookmaker", f"{gap_to_bm:+.4f}", delta_color="inverse")

    st.divider()
    st.subheader("Full rankings (sorted by log loss)")

    display_df = rankings_df.drop(columns=["_is_failed"]).copy()
    display_df.insert(0, "Rank", range(1, len(display_df) + 1))

    def _style_rows(row):
        if "B365" in str(row["Model"]):
            return ["background-color: #fef9c3"] * len(row)   # gold — bookmaker benchmark
        if row["Rank"] == 1:
            return ["background-color: #dcfce7"] * len(row)   # green — best model
        if str(row["Model"]) in _FAILED_MODELS:
            return ["background-color: #fee2e2"] * len(row)   # red — failed models
        return [""] * len(row)

    fmt = {
        "Log Loss":   "{:.4f}",
        "Brier":      "{:.4f}",
        "Accuracy %": "{:.2f}",
        "AUC":        "{:.4f}",
        "CV LL":      "{:.4f}",
        "CV σ":       "{:.4f}",
    }
    st.dataframe(
        display_df.style.apply(_style_rows, axis=1).format(fmt, na_rep="—"),
        use_container_width=True,
        height=420,
    )

    with st.expander("Academic notes"):
        st.markdown(
            f"""
            - **Best model:** {best_model['Model']} (LL={best_model['Log Loss']:.4f})
              is **{abs(gap_to_bm):.4f} nats worse** than Bookmaker B365 (LL={bm_ll:.4f}).
              This gap is expected for team-level features — lineup data (Tier 2) is
              the anticipated source of the remaining signal.
            - **QDA and Gaussian NB fail catastrophically** (LL>2.7) due to collinearity
              among features. QDA's covariance matrix inversion is numerically unstable;
              GNB's feature-independence assumption is severely violated.
            - **Super Learner (stack) underperforms LASSO** (+0.0115 LL) because all 8
              base models share identical features, producing correlated errors that
              stacking cannot diversify away.
            - ★ Bookmaker B365 shown as benchmark only — not a trained model.
            """
        )


# ── Tab 2: Charts ─────────────────────────────────────────────────────────────

with tab2:
    # Exclude catastrophically failed models from the main chart for readability
    chart_df = rankings_df[~rankings_df["_is_failed"]].copy()
    chart_df = chart_df[~chart_df["Model"].str.contains("B365")].sort_values("Log Loss")
    bm_ll_val = float(rankings_df[rankings_df["Model"].str.contains("B365")]["Log Loss"].iloc[0])

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Log Loss by model")
        st.caption("(QDA and Gaussian NB excluded — LL > 2.0)")

        fig, ax = plt.subplots(figsize=(7, 5))
        colours = ["#2563eb" if "Stack" not in m else "#9333ea"
                   for m in chart_df["Model"]]
        bars = ax.barh(chart_df["Model"], chart_df["Log Loss"],
                       color=colours, alpha=0.85, edgecolor="white")
        ax.axvline(bm_ll_val, color="#dc2626", linestyle="--", linewidth=1.5,
                   label=f"B365 benchmark ({bm_ll_val:.4f})")
        ax.set_xlabel("Log Loss (lower = better)", fontsize=10)
        ax.set_title("Log Loss — EPL 2024 Test Set", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        for bar, val in zip(bars, chart_df["Log Loss"]):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_right:
        st.subheader("Log Loss vs Brier Score")

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        models_chart = chart_df["Model"].tolist()
        x = np.arange(len(models_chart))
        width = 0.35
        ax2.bar(x - width / 2, chart_df["Log Loss"], width,
                label="Log Loss", color="#2563eb", alpha=0.85, edgecolor="white")
        ax2.bar(x + width / 2, chart_df["Brier"], width,
                label="Brier Score", color="#16a34a", alpha=0.85, edgecolor="white")
        ax2.axhline(bm_ll_val, color="#dc2626", linestyle="--",
                    linewidth=1.2, label=f"B365 LL ({bm_ll_val:.4f})")
        ax2.set_xticks(x)
        ax2.set_xticklabels(models_chart, rotation=25, ha="right", fontsize=8)
        ax2.set_title("Log Loss & Brier Score — EPL 2024", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)


# ── Tab 3: CV Results ─────────────────────────────────────────────────────────

with tab3:
    cv_df = rankings_df[
        rankings_df["CV LL"].notna() & ~rankings_df["_is_failed"]
    ][["Model", "Log Loss", "CV LL", "CV σ"]].copy()

    st.subheader("5-fold stratified CV on train+val (2022–2023)")
    st.caption(
        "CV was run on the combined train+val set (760 matches). "
        "Test log loss shown for comparison."
    )
    st.dataframe(
        cv_df.style.format({"Log Loss": "{:.4f}", "CV LL": "{:.4f}", "CV σ": "{:.4f}"}),
        use_container_width=True,
    )

    # CV bar chart with error bars
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    x = np.arange(len(cv_df))
    ax3.bar(x, cv_df["CV LL"], yerr=cv_df["CV σ"],
            color="#6366f1", alpha=0.85, edgecolor="white",
            capsize=4, error_kw={"linewidth": 1.2})
    ax3.axhline(bm_ll_val, color="#dc2626", linestyle="--",
                linewidth=1.2, label=f"B365 LL ({bm_ll_val:.4f})")
    ax3.set_xticks(x)
    ax3.set_xticklabels(cv_df["Model"], rotation=25, ha="right", fontsize=9)
    ax3.set_ylabel("CV Log Loss")
    ax3.set_title("5-Fold CV Log Loss ± 1σ — Train+Val (2022–2023)", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)
