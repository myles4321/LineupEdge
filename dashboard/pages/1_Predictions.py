"""Predictions browser — filter and explore match-level predictions."""

import pathlib
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from dashboard.db import query

st.set_page_config(page_title="Predictions · LineupEdge", page_icon="📊", layout="wide")
st.title("📊 Predictions")
st.caption("Browse model predictions for the 2024 test season (380 EPL matches).")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    return query(
        """
        SELECT
            m.date,
            m.season,
            m.matchday,
            ht.name  AS home_team,
            at.name  AS away_team,
            m.home_goals,
            m.away_goals,
            m.result AS actual,
            p.model_version,
            p.tier,
            ROUND(p.prob_home::numeric, 3) AS prob_home,
            ROUND(p.prob_draw::numeric, 3) AS prob_draw,
            ROUND(p.prob_away::numeric, 3) AS prob_away
        FROM predictions p
        JOIN matches m  ON m.match_id  = p.match_id
        JOIN teams   ht ON ht.team_id  = m.home_team_id
        JOIN teams   at ON at.team_id  = m.away_team_id
        WHERE m.status = 'FT'
        ORDER BY m.date, m.match_id
        """
    )


@st.cache_data(ttl=300)
def load_bookmaker() -> pd.DataFrame:
    return query(
        """
        SELECT
            o.match_id,
            ROUND(o.implied_home::numeric, 3) AS bm_home,
            ROUND(o.implied_draw::numeric, 3) AS bm_draw,
            ROUND(o.implied_away::numeric, 3) AS bm_away
        FROM odds o
        WHERE o.bookmaker = 'B365'
        """
    )


try:
    df = load_predictions()
    bm = load_bookmaker()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if df.empty:
    st.warning("No predictions found. Run `python models/train_tier1.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("Filters")

seasons = sorted(df["season"].unique())
sel_season = st.sidebar.selectbox("Season", seasons, index=len(seasons) - 1)

models = sorted(df["model_version"].unique())
sel_model = st.sidebar.selectbox("Model", models)

all_teams = sorted(
    set(df["home_team"].unique()) | set(df["away_team"].unique())
)
sel_team = st.sidebar.selectbox("Team (optional)", ["All"] + all_teams)

# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

view = df[
    (df["season"] == sel_season) &
    (df["model_version"] == sel_model)
].copy()

if sel_team != "All":
    view = view[
        (view["home_team"] == sel_team) | (view["away_team"] == sel_team)
    ]

# Derive predicted outcome (argmax of probabilities)
view["predicted"] = view[["prob_home", "prob_draw", "prob_away"]].idxmax(axis=1).map(
    {"prob_home": "H", "prob_draw": "D", "prob_away": "A"}
)
view["correct"] = view["predicted"] == view["actual"]

# Confidence = max probability
view["confidence"] = view[["prob_home", "prob_draw", "prob_away"]].max(axis=1)

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

n_total   = len(view)
n_correct = view["correct"].sum()
accuracy  = n_correct / n_total if n_total else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Matches",  n_total)
c2.metric("Correct",  n_correct)
c3.metric("Accuracy", f"{accuracy:.1%}")
c4.metric(
    "Avg confidence",
    f"{view['confidence'].mean():.1%}" if n_total else "—",
)

st.divider()

# ---------------------------------------------------------------------------
# Accuracy by outcome class
# ---------------------------------------------------------------------------

col_chart, col_conf = st.columns(2)

with col_chart:
    class_acc = (
        view.groupby("actual")["correct"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "correct", "count": "total"})
        .assign(accuracy=lambda d: d["correct"] / d["total"])
        .reset_index()
    )
    class_acc["actual"] = class_acc["actual"].map(
        {"H": "Home win", "D": "Draw", "A": "Away win"}
    )
    fig = px.bar(
        class_acc,
        x="actual",
        y="accuracy",
        color="actual",
        color_discrete_map={
            "Home win": "#2563eb",
            "Draw":     "#64748b",
            "Away win": "#dc2626",
        },
        text=class_acc["accuracy"].map("{:.1%}".format),
        title="Accuracy by outcome class",
        labels={"actual": "", "accuracy": "Accuracy"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

with col_conf:
    fig2 = px.histogram(
        view,
        x="confidence",
        color="correct",
        color_discrete_map={True: "#16a34a", False: "#dc2626"},
        nbins=20,
        title="Confidence distribution (green = correct)",
        labels={"confidence": "Max predicted probability", "correct": "Correct?"},
        barmode="overlay",
        opacity=0.75,
    )
    fig2.update_layout(legend_title_text="Correct")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Predictions table
# ---------------------------------------------------------------------------

st.subheader(f"Match predictions — {sel_model} — {sel_season} season")

display = view[[
    "date", "matchday", "home_team", "away_team",
    "home_goals", "away_goals", "actual", "predicted",
    "prob_home", "prob_draw", "prob_away", "confidence", "correct",
]].copy()

display["date"] = pd.to_datetime(display["date"]).dt.strftime("%d %b %Y")
display["score"] = display["home_goals"].astype(str) + "–" + display["away_goals"].astype(str)
display = display.drop(columns=["home_goals", "away_goals"])
display = display.rename(columns={
    "date":       "Date",
    "matchday":   "GW",
    "home_team":  "Home",
    "away_team":  "Away",
    "score":      "Score",
    "actual":     "Result",
    "predicted":  "Predicted",
    "prob_home":  "P(H)",
    "prob_draw":  "P(D)",
    "prob_away":  "P(A)",
    "confidence": "Confidence",
    "correct":    "✓",
})

def _row_style(row):
    bg = "#dcfce7" if row["✓"] else "#fee2e2"
    return [f"background-color: {bg}"] * len(row)

st.dataframe(
    display.style.apply(_row_style, axis=1).format({
        "P(H)": "{:.3f}", "P(D)": "{:.3f}", "P(A)": "{:.3f}",
        "Confidence": "{:.1%}",
    }),
    use_container_width=True,
    height=500,
)
