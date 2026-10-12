"""Feature importance page — top-20 features with group colour coding."""

import pathlib
import re
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Feature Importance · LineupEdge", page_icon="🔍", layout="wide"
)
st.title("🔍 Feature Importance")
st.caption("XGBoost feature importances from the Tier 1 model (gain metric).")

REPORTS = pathlib.Path(__file__).parent.parent.parent / "reports"

# ---------------------------------------------------------------------------
# Parse feature_analysis.txt
# ---------------------------------------------------------------------------

_ROW_RE = re.compile(
    r"^\s*\d+\.\s+(\S+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)"
)

_GROUP_MAP = {
    "elo":       ("ELO Rating",       "#2563eb"),
    "form":      ("Form",             "#16a34a"),
    "xg":        ("Expected Goals",   "#9333ea"),
    "h2h":       ("Head-to-Head",     "#ea580c"),
    "rest":      ("Rest / Fatigue",   "#0891b2"),
    "home":      ("Venue",            "#65a30d"),
    "away":      ("Venue",            "#65a30d"),
    "venue":     ("Venue",            "#65a30d"),
    "streak":    ("Form",             "#16a34a"),
    "goals":     ("Goals",            "#db2777"),
    "pts":       ("Points",           "#d97706"),
    "lineup":    ("Lineup (T2)",      "#f43f5e"),
    "formation": ("Lineup (T2)",      "#f43f5e"),
    "strength":  ("Lineup (T2)",      "#f43f5e"),
    "fatigue":   ("Lineup (T2)",      "#f43f5e"),
}

_DEFAULT_GROUP = ("Other", "#64748b")


def _group(name: str) -> tuple[str, str]:
    n = name.lower()
    for key, val in _GROUP_MAP.items():
        if key in n:
            return val
    return _DEFAULT_GROUP


def load_features() -> pd.DataFrame | None:
    p = REPORTS / "feature_analysis.txt"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        m = _ROW_RE.match(line)
        if not m:
            continue
        name, xgb_imp, uni_score = m.group(1), float(m.group(2)), float(m.group(3))
        if xgb_imp == 0.0:
            continue
        label, colour = _group(name)
        rows.append({
            "Feature":    name,
            "Importance": xgb_imp,
            "Univariate": uni_score,
            "Group":      label,
            "Colour":     colour,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


df = load_features()

if df is None:
    st.warning(
        "**feature_analysis.txt not found.** "
        "Run `python models/analyse_features.py` to generate it."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

top_n = st.slider("Features to display", min_value=5, max_value=len(df), value=min(20, len(df)), step=5)
view  = df.head(top_n)

# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

fig = px.bar(
    view.iloc[::-1],          # flip so top feature is at top of horizontal bar
    x="Importance",
    y="Feature",
    color="Group",
    color_discrete_map={g: c for _, (g, c) in _GROUP_MAP.items()} | {"Other": "#64748b"},
    orientation="h",
    title=f"Top {top_n} features by XGBoost gain",
    labels={"Importance": "Gain (XGBoost)", "Feature": ""},
    hover_data={"Univariate": ":.4f", "Group": True},
)

fig.update_layout(
    height=max(400, top_n * 26),
    legend_title_text="Feature group",
    plot_bgcolor="#f8fafc",
    xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
    yaxis=dict(showgrid=False),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

st.subheader("Feature ranking table")

tbl = view[["Feature", "Importance", "Univariate", "Group"]].copy()
tbl.index = tbl.index + 1   # 1-based rank
tbl = tbl.rename(columns={
    "Importance": "XGB Gain",
    "Univariate": "Univariate Score",
})

st.dataframe(
    tbl.style.format({"XGB Gain": "{:.4f}", "Univariate Score": "{:.4f}"}),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Academic framing
# ---------------------------------------------------------------------------

with st.expander("Academic context"):
    st.markdown(
        """
        **Feature importance (XGBoost gain)** measures the average reduction in
        training loss attributed to splits on a given feature — a higher value
        indicates the feature is more useful for discrimination.

        Key findings from the Tier 1 analysis:

        - **ELO rating gap** is consistently the strongest predictor, consistent
          with the broader sports analytics literature (Hvattum & Arntzen, 2010).
        - **Rolling form features** (recent points, goal difference) provide
          incremental lift over ELO alone.
        - **Head-to-head record** has moderate importance but high sparsity (~50%
          null for teams without prior EPL meetings).
        - **xG features** are 100% null until match-stats ingestion completes and
          will only contribute when the `--step stats` pipeline has run.
        - **Tier 2 lineup features** (formation, strength, fatigue) will appear
          here in orange once lineup ingestion and Tier 2 training are complete.
        """
    )
