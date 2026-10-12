"""LineupEdge Dashboard — Home / Overview."""

import json
import pathlib
import sys

import streamlit as st

# Allow sibling imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

st.set_page_config(
    page_title="LineupEdge",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

REPORTS = pathlib.Path(__file__).parent.parent / "reports"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tier1() -> dict | None:
    p = REPORTS / "tier1_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_comparison() -> dict | None:
    p = REPORTS / "tier_comparison.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _find_model(results: list[dict], key: str) -> dict | None:
    for r in results:
        if key in r.get("model", ""):
            return r
    return None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("⚽ LineupEdge")
st.caption("EPL match outcome prediction — does pre-match lineup data improve results?")
st.divider()

tier1 = _load_tier1()
comparison = _load_comparison()

if tier1 is None:
    st.error(
        "**tier1_results.json not found.** "
        "Run `python models/train_tier1.py` first."
    )
    st.stop()

results = tier1["results"]
logreg = _find_model(results, "logreg")
xgb    = _find_model(results, "xgb")
bm     = _find_model(results, "bookmaker") or _find_model(results, "B365")

split = tier1.get("split", {})
n_features = tier1.get("n_features", "—")

# ---------------------------------------------------------------------------
# Research question
# ---------------------------------------------------------------------------

with st.expander("Research question", expanded=False):
    st.markdown(
        """
        Standard match prediction models rely on team-level statistics — ELO ratings,
        rolling form, head-to-head records, and venue effects. This project asks:

        > **Does adding pre-match lineup information (formation, player availability,
        > squad strength, fatigue) produce statistically significant improvements in
        > probabilistic match outcome predictions over team-level baselines?**

        **Tier 1** models use 38 team-level features.
        **Tier 2** models add 14 lineup features (formation win-rate, lineup strength,
        deviation from typical squad, fatigue index, key player availability).
        The bookmaker (Bet365) implied probabilities serve as an industry benchmark.
        """
    )

# ---------------------------------------------------------------------------
# Experiment setup cards
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Train season",    str(split.get("train", "2022")))
c2.metric("Val season",      str(split.get("val",   "2023")))
c3.metric("Test season",     str(split.get("test",  "2024")))
c4.metric("Tier 1 features", str(n_features))

st.divider()

# ---------------------------------------------------------------------------
# Tier 1 metric cards
# ---------------------------------------------------------------------------

st.subheader("Tier 1 — Test set results (2024 season, 380 matches)")

col_lr, col_xgb, col_bm = st.columns(3)

if logreg:
    col_lr.metric(
        "Logistic Regression",
        f"{logreg['log_loss']:.4f}",
        help="Log loss (lower = better)",
    )
    col_lr.caption(f"Brier: {logreg['brier_score']:.4f}")

if xgb:
    delta_vs_bm = (
        f"{xgb['log_loss'] - bm['log_loss']:+.4f} vs bookmaker"
        if bm else None
    )
    col_xgb.metric(
        "XGBoost",
        f"{xgb['log_loss']:.4f}",
        delta=delta_vs_bm,
        delta_color="inverse",
        help="Log loss (lower = better)",
    )
    col_xgb.caption(f"Brier: {xgb['brier_score']:.4f}")

if bm:
    col_bm.metric(
        "Bookmaker B365 (benchmark)",
        f"{bm['log_loss']:.4f}",
        help="Log loss implied from Bet365 odds",
    )
    col_bm.caption(f"Brier: {bm['brier_score']:.4f}")

# ---------------------------------------------------------------------------
# Tier 2 status / metrics
# ---------------------------------------------------------------------------

st.divider()

if comparison:
    st.subheader("Tier 2 — Results (lineup features)")

    t1x = comparison.get("tier1_xgb", {})
    t2x = comparison.get("tier2_xgb", {})
    bm2 = comparison.get("bookmaker", {})

    ca, cb, cc, cd = st.columns(4)
    ca.metric("T1 XGB log loss",  f"{t1x.get('log_loss', 0):.4f}")
    cb.metric(
        "T2 XGB log loss",
        f"{t2x.get('log_loss', 0):.4f}",
        delta=f"{t2x.get('log_loss', 0) - t1x.get('log_loss', 0):+.4f} vs T1",
        delta_color="inverse",
    )
    cc.metric("Bookmaker log loss", f"{bm2.get('log_loss', 0):.4f}")

    bootstrap = comparison.get("bootstrap", {})
    if bootstrap:
        p_val = bootstrap.get("p_value_log_loss")
        sig = "Yes ✓" if p_val is not None and p_val < 0.05 else "No"
        cd.metric("Significant improvement?", sig, help=f"Bootstrap p-value: {p_val:.3f}" if p_val else "")

else:
    st.info(
        "**Tier 2 results not yet available** — lineup ingestion is still running. "
        "Once `python models/train_tier2.py` and `python models/compare_tiers.py` "
        "have been run, full T1 vs T2 comparison will appear here.",
        icon="⏳",
    )

# ---------------------------------------------------------------------------
# Sidebar nav hint
# ---------------------------------------------------------------------------

st.sidebar.title("Navigation")
st.sidebar.markdown(
    """
    - **Home** — overview & metrics
    - **Predictions** — browse match predictions
    - **Calibration** — reliability diagrams
    - **Feature Importance** — top-20 features
    """
)
st.sidebar.divider()
st.sidebar.caption("LineupEdge · Graduate Seminar Project")
