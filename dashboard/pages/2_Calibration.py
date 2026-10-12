"""Calibration page — reliability diagrams for each outcome class."""

import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from dashboard.db import query

st.set_page_config(page_title="Calibration · LineupEdge", page_icon="📈", layout="wide")
st.title("📈 Calibration")
st.caption(
    "Reliability diagrams: how well do predicted probabilities match observed frequencies? "
    "A perfectly calibrated model's curve lies on the diagonal."
)

_LABEL_TO_IDX = {"H": 0, "D": 1, "A": 2}
_CLASS_LABELS  = ["H", "D", "A"]
_CLASS_NAMES   = {"H": "Home win", "D": "Draw", "A": "Away win"}
_N_BINS        = 10

# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    return query(
        """
        SELECT
            m.result,
            p.model_version,
            p.tier,
            p.prob_home::float AS prob_home,
            p.prob_draw::float AS prob_draw,
            p.prob_away::float AS prob_away
        FROM predictions p
        JOIN matches m ON m.match_id = p.match_id
        WHERE m.season = 2024 AND m.status = 'FT'
        ORDER BY p.model_version
        """
    )


@st.cache_data(ttl=300)
def load_bookmaker() -> pd.DataFrame:
    return query(
        """
        SELECT
            m.result,
            o.implied_home::float AS prob_home,
            o.implied_draw::float AS prob_draw,
            o.implied_away::float AS prob_away
        FROM odds o
        JOIN matches m ON m.match_id = o.match_id
        WHERE o.bookmaker = 'B365' AND m.season = 2024 AND m.status = 'FT'
        """
    )


try:
    df  = load_data()
    bm  = load_bookmaker()
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

if df.empty:
    st.warning("No predictions in the database.")
    st.stop()

# ---------------------------------------------------------------------------
# Calibration curve helper
# ---------------------------------------------------------------------------

def calibration_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10):
    """Return (mean_pred, fraction_pos) for a reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mean_pred, frac_pos = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        mean_pred.append(probs[mask].mean())
        frac_pos.append(labels[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

models_in_db = df["model_version"].unique().tolist()

# Colour map
_COLOURS = {
    "tier1_logreg_v1": "#3b82f6",
    "tier1_xgb_v1":    "#8b5cf6",
    "tier2_logreg_v1": "#f59e0b",
    "tier2_xgb_v1":    "#10b981",
    "bookmaker":       "#ef4444",
}

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[_CLASS_NAMES[c] for c in _CLASS_LABELS],
    shared_yaxes=True,
)

for col_idx, cls in enumerate(_CLASS_LABELS, start=1):
    prob_col = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}[cls]

    # Perfect calibration diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#94a3b8", width=1),
            name="Perfect calibration" if col_idx == 1 else None,
            showlegend=(col_idx == 1),
            legendgroup="diagonal",
        ),
        row=1, col=col_idx,
    )

    # Model curves
    for mv in models_in_db:
        sub = df[df["model_version"] == mv]
        probs  = sub[prob_col].values
        actuals = (sub["result"] == cls).astype(int).values
        mp, fp = calibration_curve(probs, actuals, _N_BINS)
        colour = _COLOURS.get(mv, "#64748b")
        fig.add_trace(
            go.Scatter(
                x=mp, y=fp,
                mode="lines+markers",
                name=mv if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                legendgroup=mv,
                line=dict(color=colour, width=2),
                marker=dict(size=6),
            ),
            row=1, col=col_idx,
        )

    # Bookmaker curve
    if not bm.empty:
        bp  = bm[prob_col].values
        ba  = (bm["result"] == cls).astype(int).values
        mp, fp = calibration_curve(bp, ba, _N_BINS)
        fig.add_trace(
            go.Scatter(
                x=mp, y=fp,
                mode="lines+markers",
                name="Bookmaker B365" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
                legendgroup="bookmaker",
                line=dict(color=_COLOURS["bookmaker"], width=2, dash="dot"),
                marker=dict(size=6, symbol="diamond"),
            ),
            row=1, col=col_idx,
        )

fig.update_layout(
    height=420,
    title_text="Reliability Diagrams — 2024 Test Season",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    plot_bgcolor="#f8fafc",
)
for i in range(1, 4):
    fig.update_xaxes(title_text="Mean predicted probability", range=[0, 1], row=1, col=i)
fig.update_yaxes(title_text="Fraction of positives", range=[0, 1], row=1, col=1)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

with st.expander("How to read this chart"):
    st.markdown(
        """
        Each subplot shows one outcome class (Home win, Draw, Away win).

        - **X-axis**: the average predicted probability within a bin
        - **Y-axis**: the actual fraction of that outcome in that bin
        - **Dashed diagonal**: perfect calibration (predicted = actual)
        - **Above the diagonal**: model is under-confident (actual rate > predicted)
        - **Below the diagonal**: model is over-confident (actual rate < predicted)

        Well-calibrated models are useful for betting simulations — a model that
        predicts 60% should win roughly 60% of the time.
        """
    )

# ---------------------------------------------------------------------------
# Per-class log loss table
# ---------------------------------------------------------------------------

st.subheader("Per-class log loss (one-vs-rest)")

from sklearn.metrics import log_loss  # noqa: E402 — lazy import to keep startup fast

rows = []
for mv in models_in_db:
    sub = df[df["model_version"] == mv]
    for cls in _CLASS_LABELS:
        prob_col = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}[cls]
        y_true = (sub["result"] == cls).astype(int).values
        y_prob = sub[prob_col].values
        ll = log_loss(y_true, y_prob)
        rows.append({"Model": mv, "Class": _CLASS_NAMES[cls], "Log Loss": round(ll, 4)})

if not bm.empty:
    for cls in _CLASS_LABELS:
        prob_col = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}[cls]
        y_true = (bm["result"] == cls).astype(int).values
        y_prob = bm[prob_col].values
        ll = log_loss(y_true, y_prob)
        rows.append({"Model": "Bookmaker B365", "Class": _CLASS_NAMES[cls], "Log Loss": round(ll, 4)})

tbl = (
    pd.DataFrame(rows)
    .pivot(index="Model", columns="Class", values="Log Loss")
    .reset_index()
)

st.dataframe(tbl, use_container_width=True, hide_index=True)
