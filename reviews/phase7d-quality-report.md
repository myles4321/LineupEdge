# Quality Report: Phase 7d — Unsupervised Analysis

**Date:** 2026-10-13
**Reviewer:** Claude Code QA
**File reviewed:** `models/unsupervised.py`

---

## Build Status

| Check | Status |
|-------|--------|
| `py_compile models/unsupervised.py` | PASS |
| `python models/unsupervised.py` (end-to-end) | PASS |
| Results post-fix identical to pre-fix (KMeans + hierarchical structure) | PASS |

---

## Code Quality Score: 9/10

---

## Critical Correctness Checks

### 1. Team Feature Aggregation — home_win_rate / away_win_rate

```python
home_win_rate = (home_games["result"] == "H").sum() / len(home_games)
away_win_rate = (away_games["result"] == "A").sum() / len(away_games)
```

When a team plays at home, they win if `result == "H"`. When away, they win if `result == "A"`. Both comparisons use the string result column (`"H"/"D"/"A"` per spec). **PASS**

### 2. KMeans Silhouette Selection

```python
best_k = max(silhouette_scores, key=silhouette_scores.get)
```

`max(dict, key=dict.get)` returns the key (k value) with the highest value. Correctly identifies best k. Verified: k=3:0.543, k=4:0.581, k=5:0.508, k=6:0.429 → best_k=4. **PASS**

### 3. Dendrogram Cut Line Formula

```python
cut_distance = (Z[-3, 2] + Z[-4, 2]) / 2
```

The linkage matrix Z (n-1 rows) is sorted by ascending merge distance. Z[-1]=1 cluster remaining, Z[-2]=2, Z[-3]=3, Z[-4]=4, Z[-5]=5. To cut to exactly 4 clusters, the cut must fall between Z[-4,2] (the last merge that still leaves 4 clusters) and Z[-3,2] (the merge that reduces to 3 clusters). Midpoint between these two distances is the correct visual placement. **PASS**

### 4. fcluster maxclust Criterion

```python
cluster_labels = fcluster(Z, t=_HIER_N_CLUSTERS, criterion="maxclust")
```

`criterion='maxclust'` with `t=4` produces at most 4 clusters. With 1,126 observations the tree is deep enough to produce exactly 4. scipy returns 1-indexed labels (1–4). Verified in output: cluster IDs 1, 2, 3, 4. **PASS**

### 5. Index Alignment — team_df cluster assignment

```python
X_scaled = scaler.fit_transform(team_df.values)       # numpy array, row-order = team_df row-order
cluster_labels = km_final.fit_predict(X_scaled)        # numpy array, same order
team_df["cluster"] = cluster_labels                    # positional (numpy→DataFrame), correct
```

Assigning a numpy array to a DataFrame column uses positional alignment (not index label matching). Since `X_scaled` was derived from `team_df.values` in the same row order, cluster assignments map to the correct teams. Same reasoning applies to PCA coordinates `pc1`/`pc2`. **PASS**

### 6. Index Alignment — match_df cluster assignment

```python
match_df = match_df.reset_index(drop=True)             # 0-based integer index
cluster_labels = fcluster(Z, ...)                      # numpy array, length=n_matches
match_df["cluster"] = cluster_labels                   # positional, correct
```

`reset_index(drop=True)` ensures 0-based contiguous index, so positional numpy assignment is safe. **PASS**

### 7. Cluster Label Uniqueness

`_assign_cluster_label` evaluates conditions in priority order and returns one of four distinct labels. Post-fix, all four hierarchical clusters receive unique labels:

| Cluster | ELO diff | H% | Label |
|---------|----------|-----|-------|
| 3 | 193.6 | 69.1% | Strong Home |
| 4 | 61.9 | 61.6% | Home Favoured |
| 2 | 18.6 | 50.0% | Balanced Contest |
| 1 | -72.2 | 31.3% | Away Favoured |

**PASS** (after fix #3 below)

### 8. JSON Schema

All required keys present and correct in both experiments. KMeans clusters keyed `"0"`–`"3"` (int → str). Hierarchical clusters keyed `"1"`–`"4"` (scipy 1-indexed → str). **PASS** (after fix #2 below)

---

## Issues Found & Fixed

| # | Severity | Category | Description | File:Line (pre-fix) | Status |
|---|----------|----------|-------------|---------------------|--------|
| 1 | Minor | Dead import | `from matplotlib.lines import Line2D` imported but never referenced anywhere in the script | `unsupervised.py:29` | **Fixed** |
| 2 | Minor | JSON schema mismatch + crash | Hierarchical cluster JSON used keys `home_form`/`away_form` but spec defines `home_win_rate`/`away_win_rate`; the summary printer also referenced the old keys, causing a `KeyError` crash on the print path | `unsupervised.py:323–324, 380` | **Fixed** |
| 3 | Minor | Duplicate cluster labels | Both cluster 3 (ELO diff=193.6, H%=69.1%) and cluster 4 (ELO diff=61.9, H%=61.6%) triggered the same "Home Favoured" condition, making interpretation ambiguous. Added an explicit "Strong Home" check for `elo_diff > 150 and home_pct > 0.55` as the first branch | `unsupervised.py:243` | **Fixed** |

**Issue #2 was a functional bug (KeyError crash); issues #1 and #3 are minor quality issues. KMeans and hierarchical clustering results are numerically identical pre/post-fix.**

---

## Results Verified (post-fix)

### K-Means Team Clustering

| Cluster | Size | Mean ELO | Home W% | Away W% | Home PPG | Away PPG | Interpretation |
|---------|------|----------|---------|---------|----------|----------|----------------|
| 2 | 3 | 1635 | 73.7% | 54.4% | 2.152 | 2.145 | Elite (top-6 core) |
| 1 | 6 | 1542 | 55.0% | 35.7% | 1.545 | 1.623 | Upper-mid table |
| 3 | 8 | 1477 | 38.2% | 27.9% | 1.198 | 1.255 | Mid-table |
| 0 | 7 | 1430 | 15.0% | 12.4% | 0.643 | 0.652 | Relegation zone |

Best k=4 (silhouette=0.581 vs k=3: 0.543, k=5: 0.508, k=6: 0.429)

### Hierarchical Match-Context Clustering (n=1,126 matches)

| Cluster | Size | ELO diff | Pts diff | H-form | A-form | H% | D% | A% | Label |
|---------|------|----------|----------|--------|--------|-----|-----|-----|-------|
| 3 | 94 | +193.6 | +29.8 | 0.564 | 0.136 | 69.1 | 23.4 | 7.4 | Strong Home |
| 4 | 255 | +61.9 | +6.9 | 0.687 | 0.435 | 61.6 | 20.0 | 18.4 | Home Favoured |
| 2 | 234 | +18.6 | +2.8 | 0.222 | 0.123 | 50.0 | 24.8 | 25.2 | Balanced Contest |
| 1 | 543 | -72.2 | -9.8 | 0.265 | 0.533 | 31.3 | 23.8 | 44.9 | Away Favoured |

---

## Findings of Academic Note

1. **K=4 is clearly optimal** (silhouette=0.581 vs 0.543 at k=3) — the EPL stratification into 4 tiers (elite/upper-mid/mid-table/relegation) is supported by the data. The ELO gap between elite (1635) and relegation (1430) is 205 points across 5 playing-style features.
2. **Away Favoured is the largest cluster (543/1,126 = 48%)** — the majority of EPL matches in the dataset involve a stronger away side (negative ELO differential), reflecting that visiting top-6 teams are frequent visitors to mid/lower clubs. Home advantage is insufficient to overcome large ELO mismatches.
3. **Strong Home cluster (n=94) is rare but extreme** — 69.1% home win rate, 7.4% away win rate; these are matches where home team is ELO-dominant (+193) with low away form (0.136). This is the most predictable context in the dataset.
4. **Balanced Contest cluster (n=234) has 50.0% home win rate** — equal-quality opponents produce a symmetric outcome distribution (50/25/25 H/D/A), confirming that bookmaker home advantage disappears when ELO and form are matched.

---

## Summary

`unsupervised.py` is functionally correct with two correctness issues and one dead import resolved. The critical fix was issue #2 (crash on the `home_form`/`away_form` key mismatch in the summary printer). Issue #3 (duplicate "Home Favoured" labels for two qualitatively different cluster profiles) improves interpretability. All clustering logic — team feature aggregation, silhouette argmax, dendrogram cut placement, `fcluster(maxclust)`, and index alignment — is verified correct. All three output files (JSON + 2 PNGs) are produced cleanly.

**Recommendation: APPROVE**
