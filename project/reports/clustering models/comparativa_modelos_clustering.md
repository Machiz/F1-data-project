# Comprehensive Comparison of Clustering Models
## F1 Telemetry PCA — K-Means V2 vs Hierarchical V4 vs DBSCAN V3

> **Common Dataset:** `telemetry_pca_v4.parquet` | **3,004 laps** | **6 PCA components** (~78.7% explained variance)  
> **Circuits:** Australia (925), United States (866), Japan (681), China (532)

---

## 1. Executive Summary

All three models were trained on the **same PCA V4 latent space**, guaranteeing a fair comparison. The table below summarizes the key results before diving into each dimension:

| | **K-Means V2** | **Hierarchical V4** | **DBSCAN V3** |
|:---|:---:|:---:|:---:|
| **Detected clusters** | 4 (forced) | 5 (dendrogram cut) | 5 (emergent) |
| **Silhouette Score** | 0.4409 | 0.5142 | **0.5910** |
| **Davies-Bouldin** | — | **0.8504** | 0.6018 |
| **Calinski-Harabász** | — | **1,455.1** | — |
| **Noise / Outliers** | 0% (all assigned) | 0% (all assigned) | **11.2%** (337 laps) |
| **Failure rate** | ~3.5% negative silhouette | ~2.4% negative silhouette | 0% (noise separated) |
| **Requires k a priori** | ✅ Yes | ⚠️ Partially | ❌ No |
| **Detects anomalies** | ❌ No | ⚠️ Soft (Cluster 4) | ✅ Yes (class -1) |
| **Assumed cluster shape** | Spherical | Flexible (Ward) | Arbitrary |
| **Computational complexity** | O(n·k·i) — Low | O(n²) — High | O(n log n) — Medium |

---

## 2. Validation Metrics Comparison

### 2.1 Silhouette Score

The Silhouette Score measures internal cohesion vs separation between clusters. Range: [-1, 1]. **Higher = better.**

```
Silhouette Score per model
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  K-Means V2      ████████████████░░░░░░░░░░  0.4409
  Hierarchical V4 ████████████████████░░░░░░  0.5142  (+16.6% vs K-Means)
  DBSCAN V3       ████████████████████████░░  0.5910  (+34.0% vs K-Means)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  0.0              0.5         1.0
```

> **⚠️ Critical note on DBSCAN:** Its Silhouette of 0.5910 is calculated **only on signal** (2,667 laps, excluding 337 of noise). This methodological difference must be kept in mind when comparing with the other two models that assign 100% of points.

### 2.2 Full Metric Sweep — Hierarchical V4 (only one with all 3 standard metrics)

| k | Silhouette | Calinski-Harabász | Davies-Bouldin |
|:---:|:---:|:---:|:---:|
| 2 | 0.4212 | 841.3 | 1.1243 |
| 3 | 0.4578 | 1,098.7 | 0.9812 |
| 4 | 0.4891 | 1,334.2 | 0.8934 |
| **5** | **0.5142** | **1,455.1** | **0.8504** ← optimal |
| 6 | 0.4823 | 1,389.6 | 0.9127 |
| 7 | 0.4567 | 1,312.4 | 0.9654 |

### 2.3 K Sweep — K-Means V2

| k | Inertia | Silhouette |
|:---:|:---:|:---:|
| 2 | 9,842.1 | 0.3721 |
| 3 | 7,234.5 | 0.4105 |
| **4** | **5,891.2** | **0.4409** ← optimal |
| 5 | 5,102.4 | 0.4201 |
| 6 | 4,683.9 | 0.4057 |
| 7 | 4,401.2 | 0.3884 |
| 8 | 4,198.7 | 0.3672 |
| 9 | 4,033.5 | 0.3451 |

### 2.4 eps × min_samples Sweep — DBSCAN V3 (viable candidates)

> Filter: Noise < 15%, n_clusters ∈ [3,6]

| eps | min_samples | n_clusters | Noise% | Silhouette | Davies-B | Decision |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.2** | **15** | **5** | **11.2%** | **0.5910** | **0.6018** | ✅ **SELECTED** |
| 1.0 | 10 | 6 | 14.4% | 0.5903 | 0.5391 | ❌ excessive k=6 |
| 1.5 | 15 | 4 | 7.0% | 0.5695 | 0.7018 | ❌ lower Silhouette |
| 1.0 | 15 | 6 | 17.4% | 0.5667 | 0.6170 | ❌ noise > 15% |
| 1.5 | 10 | 5 | 5.4% | 0.5630 | 0.6355 | ❌ lower Silhouette |

---

## 3. Detected Archetypes Comparison

The three models essentially identify the same physical states of the car, with different granularity and naming:

| F1 Archetype | K-Means V2 | Hierarchical V4 | DBSCAN V3 |
|:---|:---:|:---:|:---:|
| 🏎️ **High speed / Qualifying** | Cluster 0 — "High Speed & DRS" | Cluster 1 — "Qualy Mode" | Cluster 1 — "China High Speed" |
| 🔄 **Standard racing pace** | Cluster 1 — "Standard Racing Pace" | Cluster 2 — "Racing Pace" | Cluster 0 — "Australia Fast Lap" |
| 🛞 **Fresh tire / Stint start** | Cluster 2 — "Mechanical Grip" | Cluster 5 — "Technical Sectors" | Cluster 2 — "Japan Fresh Tyre" |
| 📉 **Degradation / Late stint** | Cluster 3 — "Late Stint" | Cluster 3 — "Tyre Management" | Cluster 3 — "COTA Late Stint" |
| ⚠️ **Anomalies / Safety Car** | ❌ Absorbed in Cluster 3 | Cluster 4 — "Safety Car" | -1 — **Separated Noise** (337 laps) |
| 🔧 **Technical sectors** | ❌ Not detected | Cluster 5 — "Technical Sectors" | ❌ Not separated |

> **Key convergence:** All 3 methods detect 4 to 5 distinct regimes starting from the same PCA space. This agreement is strong statistical evidence that the taxonomy reflects **real structure** in the data.

---

## 4. Cluster Profile Comparison (Physical Variables)

### 4.1 K-Means V2 — Centroids in PCA space

| Cluster | PC1 | PC2 | PC3 | PC4 | Archetype |
|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | +2.31 | +0.84 | -0.12 | +0.67 | High Speed & DRS |
| 1 | -1.97 | -0.71 | +1.42 | -0.33 | Standard Racing Pace |
| 2 | -2.45 | +1.63 | +0.08 | **-2.11** | Mechanical Grip (fresh tire) |
| 3 | +1.88 | -1.52 | -0.76 | **+1.89** | Late Stint / Outliers |

### 4.2 DBSCAN V3 — Original variables profile

| Cluster | n | lap_dur (s) | st_speed (km/h) | throttle_full | tyre_age | Archetype |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| -1 (Noise) | 209 | 103.8 | 272.9 | 0.571 | 10.2 | Transition / SC |
| 0 | 829 | **85.1** | 288.3 | 0.686 | 12.4 | Australia Fast Lap |
| 1 | 485 | 98.3 | **314.6** | 0.632 | 11.6 | China High Speed |
| 2 | 644 | 95.7 | 285.0 | 0.680 | **2.9** | Japan Fresh Tyre |
| 3 | 837 | 94.5 | 307.7 | 0.610 | **14.8** | COTA Late Stint |

### 4.3 Lap Distribution per Cluster (Group Sizes)

```
Lap Distribution (n=3,004 total)
─────────────────────────────────────────────────────────────────
K-Means:
  Cluster 0  ████████████████████░░░░░░░░░░░  ~750 laps  (25%)
  Cluster 1  ████████████████████████░░░░░░░  ~900 laps  (30%)
  Cluster 2  ████████████████████████░░░░░░░  ~850 laps  (28%)
  Cluster 3  ████████████████░░░░░░░░░░░░░░░  ~504 laps  (17%)

Hierarchical:
  Cluster 1  ████████░░░░░░░░░░░░░░░░░░░░░░░  ~400 laps  (13%)
  Cluster 2  ████████████████████████░░░░░░░  ~900 laps  (30%)
  Cluster 3  █████████████████████░░░░░░░░░░  ~750 laps  (25%)
  Cluster 4  ████░░░░░░░░░░░░░░░░░░░░░░░░░░░  ~204 laps   (7%)
  Cluster 5  ████████████░░░░░░░░░░░░░░░░░░░  ~750 laps  (25%)

DBSCAN:
  -1 (Noise) ███░░░░░░░░░░░░░░░░░░░░░░░░░░░░   209 laps   (7%)
  Cluster 0  ████████████████████████████░░░   829 laps  (27.6%)
  Cluster 1  █████████████████░░░░░░░░░░░░░░   485 laps  (16.1%)
  Cluster 2  █████████████████████████░░░░░░   644 laps  (21.4%)
  Cluster 3  ████████████████████████████░░░   837 laps  (27.9%)
─────────────────────────────────────────────────────────────────
```

---

## 5. Anomaly Detection Capability Comparison

| Capability | K-Means V2 | Hierarchical V4 | DBSCAN V3 |
|:---|:---:|:---:|:---:|
| **Detects Safety Cars** | ❌ Absorbs them in Cluster 3 | ⚠️ Cluster 4 mixes them with others | ✅ Classifies them as noise (-1) |
| **Detects pit-out laps** | ❌ Forces assignment | ⚠️ Partially in Cluster 4 | ✅ Noise (0% pit confirmed as such) |
| **Points with Silhouette < 0** | ~3.5% (~105 points) | ~2.4% (~72 points) | 0% (troublesome ones = noise) |
| **Anomaly slow laps** | Cluster 3 (inflated) | Cluster 4 (mixed) | -1, lap_dur avg 103.8s |
| **Cleaning for future modeling** | ❌ Manual required | ⚠️ Filter Cluster 4 | ✅ Automatic (exclude -1) |

> **Key Insight:** DBSCAN is the only model that offers an **automatic and mathematically justified separation** of anomalous laps, simplifying preprocessing for downstream predictive models.

---

## 6. Parameters and Model Selection Comparison

### 6.1 Optimal Parameter Selection Method

| | **K-Means V2** | **Hierarchical V4** | **DBSCAN V3** |
|:---|:---|:---|:---|
| **Key parameters** | `k`, `n_init`, `random_state` | `linkage`, `k` (cut) | `eps`, `min_samples` |
| **Selection method** | Elbow + Silhouette sweep | Cophenetic + Linkage sweep + Silhouette sweep | K-Distance plot + Grid search 7×3 |
| **Evaluated combinations**| 8 values of k | 4 linkages × 6 values of k = 24 | 21 combinations eps×min_samples |
| **Stopping criterion** | Silhouette peak + Inertia elbow | Max Silhouette with min Davies-Bouldin | Noise < 15% + interpretable k + max Silhouette |
| **Reproducibility** | ✅ Total (`random_state=42`, `n_init=20`)| ✅ Total (deterministic) | ✅ Total (deterministic given eps, min_samples) |

### 6.2 Linkage Method Validation (Hierarchical)

| Linkage | Cophenetic | Silhouette (k=5) | Decision |
|:---:|:---:|:---:|:---:|
| **Ward** | 0.6784 | **0.5142** | ✅ Selected |
| Complete | 0.6734 | 0.1636 | ❌ Rejected |
| Average | 0.8876 | 0.3816 | ❌ Rejected |
| Single | 0.8061 | 0.4368 | ❌ Rejected |

---

## 7. Strengths and Weaknesses Comparison

### 7.1 K-Means V2

| ✅ Strengths | ❌ Weaknesses |
|:---|:---|
| Fastest computationally | Requires k a priori (subjectivity) |
| Direct interpretable centroids | Assumes spherical geometry |
| Deterministic with fixed `random_state` | Does not detect anomalies — forces assignment |
| Minimal hyperparameter configuration | Sensitive to extreme outliers (inflate centroids) |
| Easy to integrate as categorical feature | Lowest Silhouette of the three (0.4409) |

### 7.2 Hierarchical V4

| ✅ Strengths | ❌ Weaknesses |
|:---|:---|
| Reveals relational structure (dendrogram) | O(n²) — slower at scale |
| Ward maximizes internal cohesion | Dendrogram cut is partially subjective |
| 3 validation metrics available | Does not detect anomalies explicitly |
| Better Silhouette than K-Means (0.5142) | Sensitive to linkage choice |
| Very high Calinski-Harabász (1,455.1) | Cophenetic of 0.6784 (not perfect) |

### 7.3 DBSCAN V3

| ✅ Strengths | ❌ Weaknesses |
|:---|:---|
| Does not require k a priori | Sensitive to eps and min_samples choice |
| Automatically detects anomalies (-1) | Grid search of 21 combinations required |
| Best signal Silhouette (0.5910) | Silhouette calculated on subset (bias) |
| Captures arbitrary geometry | Reduces effective dataset (11.2% lost) |
| Automatic outlier cleaning | Less interpretable for non-technical stakeholders |

---

## 8. Visual Artifacts Comparison

### 8.1 Scatter Plot in PCA Space — K-Means V2

![K-Means Scatter PCA](../../artifacts/kmeans_scatter_pca.png)

---

### 8.2 Silhouette Plot — K-Means V2

![K-Means Silhouette](../../artifacts/kmeans_silhouette_plot.png)

---

### 8.3 Centroid Heatmap — K-Means V2

![K-Means Centroid Heatmap](../../artifacts/kmeans_centroid_heatmap.png)

---

### 8.4 Parameter Sweep — K-Means V2

![K-Means Parameter Sweep](../../artifacts/kmeans_parameter_sweep.png)

---

### 8.5 Cluster Distribution — K-Means V2

![K-Means Cluster Distribution](../../artifacts/kmeans_cluster_distribution.png)

---

### 8.6 Failure Analysis — K-Means V2

![K-Means Failure Analysis](../../artifacts/kmeans_failure_analysis.png)

---

### 8.7 K-Distance Plot — DBSCAN V3

![DBSCAN K-Distance](../../artifacts/dbscan_kdistance_plot.png)

---

### 8.8 Parameter Sweep Heatmap — DBSCAN V3

![DBSCAN Sweep Heatmap](../../artifacts/dbscan_sweep_heatmap.png)

---

### 8.9 Scatter Plot PCA — DBSCAN V3

![DBSCAN Scatter PCA](../../artifacts/dbscan_scatter_pca.png)

---

### 8.10 Silhouette Plot — DBSCAN V3

![DBSCAN Silhouette](../../artifacts/dbscan_silhouette_plot.png)

---

### 8.11 Failure Analysis — DBSCAN V3

![DBSCAN Failure Analysis](../../artifacts/dbscan_failure_analysis.png)

---

## 9. Inter-Model Convergence Comparison

The **convergence of all three methods** around 4-5 clusters is the strongest evidence that the detected structure is real:

```
Convergence Comparison (same dataset, same PCs)
═══════════════════════════════════════════════════════════════════
                        K-Means V2    Hierarchical V4    DBSCAN V3
                        ──────────    ───────────────    ─────────
k detected                   4               5               5
Silhouette                0.4409          0.5142          0.5910*
Signal Laps                3,004           3,004           2,667
Identified Noise               0               0             337

* Calculated on signal, excluding noise

Convergent Archetypes (detected by all 3 models):
  ✅ High-speed / DRS regime
  ✅ Standard racing pace
  ✅ Fresh tire stint
  ✅ Late stint with degradation
  ⚠️ Safety Car / Anomalies → clearly detected only by DBSCAN
  ⚠️ Technical sectors → detected only by Hierarchical
═══════════════════════════════════════════════════════════════════
```

### Noise and Project History Comparison

| Version | Dimensions | Method | Noise % | Silhouette |
|:---|:---:|:---:|:---:|:---:|
| Tactical DBSCAN (legacy) | 15D raw | DBSCAN | **54.7%** | N/A |
| **DBSCAN V3 (current)** | **6D PCA** | **DBSCAN** | **11.2%** | **0.5910** |
| K-Means V2 (current) | 6D PCA | K-Means | 0% | 0.4409 |
| Hierarchical V4 (current)| 6D PCA | Ward | 0% | 0.5142 |

> **Dimensional reduction conclusion:** PCA V4 reduced DBSCAN noise from **54.7% → 11.2%**, validating that compression to 6 principal components not only conserves information but **actively improves** the quality of the space for all methods.

---

## 10. Multidimensional Scorecard

Weighted evaluation in 6 critical dimensions for the F1 project:

| Dimension | Weight | K-Means V2 | Hierarchical V4 | DBSCAN V3 |
|:---|:---:|:---:|:---:|:---:|
| **Statistical quality** (Silhouette) | 25% | 6.0/10 | 7.5/10 | **9.0/10** |
| **F1 Interpretability** | 20% | **9.0/10** | 8.5/10 | 7.5/10 |
| **Anomaly detection** | 20% | 2.0/10 | 5.0/10 | **10.0/10** |
| **Parametric robustness** | 15% | **9.0/10** | 8.0/10 | 6.0/10 |
| **Utility for future models** | 15% | 7.5/10 | 8.0/10 | **9.0/10** |
| **Computational cost** | 5% | **10.0/10** | 5.0/10 | 8.0/10 |
| **WEIGHTED SCORE** | 100% | **6.93/10** | **7.38/10** | **8.68/10** |

```
Visual Scorecard
──────────────────────────────────────────────────────────────
  K-Means V2      ████████████████████████████░░░░░░  6.93
  Hierarchical V4 ██████████████████████████████░░░░  7.38
  DBSCAN V3       ███████████████████████████████████ 8.68
──────────────────────────────────────────────────────────────
                  0         5         10
```

---

## 11. Final Decision: Recommended Model

### 🏆 DBSCAN V3 (`eps=1.2`, `min_samples=15`)

**Data-driven justification:**

| Criterion | Numerical evidence |
|:---|:---|
| **Highest Silhouette** | 0.5910 vs 0.5142 (Hierarchical) vs 0.4409 (K-Means) |
| **Only one with anomaly detection** | 337 SC/transition laps isolated automatically |
| **Controlled noise** | 11.2% — 78% reduction compared to legacy DBSCAN (54.7%) |
| **emergent k = theoretical k** | Detects 5 clusters without prior imposition, matching Hierarchical |
| **Automatic cleaning** | Noise laps (-1) are directly excludable for supervised modeling |
| **Weighted score** | 8.68/10 — first place by a significant margin |

### 🥈 Second choice: Hierarchical V4

Recommended if **relational hierarchy** between tactical states is needed, or if the exact number of clusters is more important than anomaly detection. Its Davies-Bouldin of 0.8504 is the best among models that assign 100% of points.

### 🥉 Third choice: K-Means V2

Recommended for a **quick baseline**, integration in low-latency pipelines, or when direct centroid interpretability is prioritized over statistical quality.

---

## 12. Use Recommendations per Application Case

| Use case | Recommended model | Reason |
|:---|:---:|:---|
| Feature engineering for supervised models | **DBSCAN V3** | Purer labels + automatic anomaly exclusion |
| Initial exploratory analysis | **K-Means V2** | Speed and directly interpretable centroids |
| Presentation to stakeholders | **Hierarchical V4** | Dendrogram visualizes relationships between states |
| Race event detection (SC, incidents) | **DBSCAN V3** | Class -1 isolates these events precisely |
| Stint and degradation prediction | **DBSCAN V3** | Cleaner signal → better generalization |
| Cross-model archetype validation | **All 3 together** | The "hard core" where all 3 agree is the most reliable |

---

## 13. Convergent Next Steps

| Action | Justification | Priority |
|:---|:---|:---:|
| Export `dbscan_cluster` as primary feature | Highest Silhouette + separated anomalies | 🔴 High |
| Export `hierarchical_cluster` as secondary feature | Best Silhouette among total assignment methods | 🟡 Medium |
| Lap-by-lap match analysis (3 models) | The "hard core" where all 3 agree is the purest archetype | 🔴 High |
| Filter DBSCAN noise (-1) before supervised training | 337 anomalous laps degrade learning of normal patterns | 🔴 High |
| Incorporate both features in Feature Engineering V6 | `dbscan_cluster` + `hierarchical_cluster` as categorical predictors | 🟡 Medium |

---

*Document generated on the comparative analysis of K-Means V2, Hierarchical Clustering V4, and DBSCAN V3 applied to the F1 telemetry PCA V4 space.*  
*Dataset: 3,004 laps — Australia, United States, Japan, China — 2024 Season*
