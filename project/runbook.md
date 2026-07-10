# Execution and Reproducibility Guide (Runbook) — F1 Strategic Decision Engine

This document provides step-by-step instructions to configure the development environment, download the data via the OpenF1 API, run the data processing pipeline, and reproduce the model training and graph analysis experiments with consistent and deterministic results.

---

## 🛠️ 1. Development Environment Setup

To guarantee scientific reproducibility and avoid library conflicts, all executions must be performed under the same Python virtual environment.

### Prerequisites:
*   **Python:** Version `3.10` or higher (3.10 is recommended).
*   **Working Directory:** All commands detailed in this Runbook must be executed from the `project/` directory of the repository:
    ```bash
    cd project
    ```

### Installation Steps:

1.  **Create the virtual environment (venv):**
    ```bash
    python -m venv venv
    ```

2.  **Activate the virtual environment:**
    *   **On Windows (PowerShell):**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **On macOS/Linux (Bash/zsh):**
        ```bash
        source venv/bin/activate
        ```

3.  **Install required dependencies:**
    The `requirements.txt` file includes stable versions of the main libraries (`pandas`, `polars`, `xgboost`, `scikit-learn`, `networkx`, `requests`, `pyarrow`, `joblib`, `jupyter`, and `streamlit` for the interactive demo). Install them by running:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

---

## ♻️ Important Note: Reconstruction After Cloning the Repository

The repository **does not version the trained models (`models/*.pkl`, `*.joblib`) or the processed parquet datasets (`data/processed/recommendation/*.parquet`)**, as they exceed GitHub's size limits and are generated deterministically from the raw data.

Consequently, after a `git clone`, the `models/` folder and recommendation parquets **do not exist yet**. Before running the audit or the demo, it is mandatory to rebuild them by running the complete chain of Steps 1 to 5 (Sections 2 to 6). A newly cloned repository will not have trained models until this chain is executed.

If you only want to rebuild the recommendation subsystem (assuming the telemetry features already exist), it is sufficient to execute Section 6 in its strict order.

---

## 🏎️ 2. Step 1: Raw Data Ingestion (E-L)

The first step downloads telemetry, lap times, tire compounds, and gaps from the official OpenF1 API. The script includes exponential retries and request throttling to avoid server blocks (HTTP 429 Error).

Run the extraction script indicating the 4 races of the 2026 season:
```bash
python src/data_extraction/extract_f1_data.py --year 2026 --races Australia China Japan "United States"
```

*   **Input:** REST calls to the API `https://api.openf1.org/v1`.
*   **Output:** Unified CSV files in `data/raw/` structured by race:
    *   `data/raw/australia_2026/laps.csv`, `pit.csv`, `stints.csv`, `car_data.csv`, `weather.csv`, `drivers.csv`
    *   `data/raw/china_2026/...`
    *   `data/raw/japan_2026/...`
    *   `data/raw/united_states_2026/...`

---

## 🧹 3. Step 2: Preprocessing and Event Extraction

This pipeline unifies the raw telemetry and sensor CSVs, corrects positioning inconsistencies using cumulative race elapsed times, and generates a consolidated file of tactical interactions (Overtakes and Pit Stops).

Run the events pipeline:
```bash
python src/features/f1_events_pipeline.py
```

*   **Inputs:** CSV files from `data/raw/`.
*   **Outputs:**
    *   **Master Parquets:** `data/processed/master/{race}_master.parquet` (1 row = 1 lap of 1 driver).
    *   **Events Parquets:** `data/processed/events/{race}_events.parquet` (1 row = 1 tactical event/interaction).

---

## 📈 4. Step 3: Feature Engineering and Dimensionality Reduction

Since the following steps are performed using Jupyter Notebooks to allow visual analysis, open Jupyter Notebook and start the server:
```bash
jupyter notebook
```

Execute the following notebooks sequentially:

### A. Feature Engineering
Open and run all cells in `notebooks/feature engineering/Feature_engineering_v5.ipynb`.

*   **Objective:** Splits the data space into Layer A (Telemetry) and Layer B (Tactics).
*   **Output:** `data/processed/features/telemetry_features_v4.parquet` and `tactical_features_v4.parquet`.

### B. PCA (Linear Dimensionality Reduction)
Open and run all cells in `notebooks/dimensionality reduction/PCA_v4.ipynb`.

*   **Objective:** Reduces the 24 numerical telemetry variables to 6 orthogonal principal components.
*   **Output:** `data/processed/features/telemetry_pca_v4.parquet`.

### C. t-SNE (Manifold Learning Embeddings)
Open and run all cells in `notebooks/dimensionality reduction/tSNE_Embeddings_Manifold_Learning.ipynb`.

*   **Objective:** Projects high-dimensional tactical events into 2D and 3D spaces.
*   **Output:** `data/processed/features/tactical_embeddings.parquet`.

---

## 🔬 5. Step 4: Clustering Analysis

To replicate and validate the unsupervised segmentation of the car's physical performance states, run the three comparative notebooks in the `notebooks/clustering models/` folder:

1.  **K-Means V2:** Run `K_Means_Clustering_V2_Telemetry_PCA.ipynb`.
2.  **Hierarchical Clustering:** Run `Hierarchical_Clustering_Telemetry_PCA.ipynb`.
3.  **DBSCAN V3:** Run `DBSCAN_V3_Telemetry_PCA.ipynb`.

*   **Input:** `data/processed/features/telemetry_pca_v4.parquet`.
*   **Output:** Cohesion and separation evaluations of the clusters.

---

## 🎯 6. Step 5: Recommender Pipeline and Ranking System

The core of the decision engine is composed of a hybrid architecture of two decoupled layers (physical degradation regression in Layer 1 and pointwise ranking in Layer 2).

### Target Formulation: Seven Actions with NO_PIT

Each decision group (race, driver, lap) generates **seven candidates**:

*   `wait_laps = 0 … 5`: pit stop after waiting *w* laps.
*   `wait_laps = 6` → **NO_PIT / STAY_OUT**: do not pit in the next 5-lap window.

The `NO_PIT` action is explicit and receives the winning label (`0.0`) in laps where there was no real pit stop within the window; offsets `0 … 5` receive `-2.0` except the one that matches a real stop, which receives its `success_score`. This formulation corrects a previous bias where `wait_laps = 0` received the best label by default in non-pit laps, forcing the ranker to learn the trivial rule "pit immediately". With `NO_PIT` as its own class, "staying out" is learned as a deliberate decision rather than an labeling artifact.

### Strict Execution Order

Dependency constraints: `update_candidates_cost.py` requires that Layer 1 is already trained; Layer 2 requires the candidates with the cost already injected. Candidate generation does not depend on Layer 1 (initializes the cost to `0.0`), so it can be executed before or after training it, but always before the cost bridge.

#### 6.1. Generate Recommender Candidates (Layer C)
Expands telemetry by adding rolling time traffic gaps and generates the seven candidates per group with success targets (including the `NO_PIT` action):
```bash
python src/features/f1_recommender_pipeline.py
```
*   **Output:** `data/processed/recommendation/pit_decision_candidates_v1.parquet` (7 rows per decision group).

#### 6.2. Train the Physical Degradation Regression Model (Layer 1)
Trains the Stacking ensemble (XGBoost + Extra Trees → Ridge Regression) with GroupKFold cross-validation by race:
```bash
python src/models/train_regression_layer1.py
```
*   **Output:** `models/regression_layer1_model.pkl` and column alignment metadata `models/regression_features.joblib`.

#### 6.3. Calculate the Strategic Cost Bridge
Predicts future stay pace and generates the cumulative cost in seconds for each waiting window. The `NO_PIT` candidate (`wait_laps = 6`) is clipped with `clip(upper=5)` to represent staying out for the entire 5-lap window, not a literal 6-lap wait:
```bash
python src/models/update_candidates_cost.py
```
*   **Output:** Updates `pit_decision_candidates_v1.parquet` by injecting the `predicted_cost_of_staying` column.

#### 6.4. Train the Decision Ranker (Layer 2)
Trains the Point-wise Random Forest Regressor to prioritize the **seven actions** (6 stop offsets + `NO_PIT`) and saves it to production:
```bash
python src/models/train_ranking_layer2.py
```
*   **Output:** `models/ranking_layer2_model.pkl`. Bias and utility evaluation are performed in Step 5.5 (audit), which is the reference metric of the system.

#### 6.5. Audit the Ranker Bias (Verification, Not Training)
Evaluates performance by class, compares against the trivial baseline, and calculates accuracy on groups where a stop was optimal:
```bash
python src/models/audit_ranking_bias.py
```
*   **Output:** `reports/ranking_system/ranking_bias_audit.md`. See Section 9 of this runbook for interpretation of the figures.

#### 6.6. Launch the Interactive Demo (Optional)
Once the models are trained, run the box wall simulated real-time assistant:
```bash
streamlit run demo/realtime_demo/app_streamlit.py
```
*   The banner distinguishes the three actions: **BOX** (pit now), **STAY** (optimal window in *k* laps), and **STAY OUT / NO_PIT** (do not pit in the window).
*   Requires `streamlit` installed (included in `requirements.txt`).

---

## 🕸️ 7. Step 6: Graph Construction and Analysis

Generates PageRank, Betweenness, and Connected Components centralities for both the combat and DRS graphs:

```bash
# Wheel-to-wheel Overtake Graph
python src/graphs/graph_construction.py

# Physical Proximity and DRS Intervals Graph
python src/graphs/drs_graph_construction.py
```

*   **Input:** Parquet files from `data/processed/events/` and `data/raw/`.
*   **Output:** Dominance and grouping metrics printed to the console and integrated into the graph reports.

---

## 🔒 8. Verification of Reproducibility and Consistency

To ensure that results do not vary across independent runs or different machines, the following controls were implemented:

1.  **Fixed Random Seed (`random_state=42`):**
    All non-deterministic estimators or those based on stochastic partitions are configured with a fixed seed. This includes:
    *   `train_regression_layer1.py` (XGBRegressor and ExtraTreesRegressor with `random_state=42`).
    *   `train_ranking_layer2.py` (RandomForestRegressor with `random_state=42`).
    *   Clustering Notebooks (K-Means and DBSCAN initialized with constant seeds).
2.  **No Lookahead Bias:**
    Validation of Layer 1 and Layer 2 is performed on test sets grouped by circuit (`GroupKFold`), ensuring that production performance metrics simulate arriving at a completely new and unknown track. The demo replicates this condition by filtering telemetry only up to the queried lap (no future data).
3.  **Data Integrity in Inference:**
    The cost bridge `update_candidates_cost.py` and the real-time pipeline (`realtime_pipeline.py`) use `regression_features.joblib` to force the same column order and dummy variables as Layer 1. Candidate generation in both training and inference is mirrored (seven candidates in both cases, with the same `clip(upper=5)` on cost), ensuring the ranker sees the same scale of `predicted_cost_of_staying` in production as it learned in training.

---

## 📊 9. Known Limitations and Audit Results

This section documents with transparency the real performance of the recommender, measured by `audit_ranking_bias.py` on 3331 decision groups.

### Audit Results

| Metric | Value |
|---|---|
| Global Accuracy (exact action) | 0.9093 |
| Baseline "always NO_PIT" | 0.8898 |
| **Binary Decision Accuracy (pit vs stay out)** | **0.9147** |
| Groups with real optimal stop (optimal ≠ NO_PIT) | 367 |
| Binary accuracy in those groups (detects a stop is needed) | 0.3896 |
| Exact accuracy in those groups (correct offset) | 0.3406 |

### Honest Interpretation

The global accuracy (0.9093) beats the trivial baseline "always NO_PIT" (0.8898) by 1.95 percentage points.

The correct indicator is the **binary decision pit / stay out (0.9147), which beats the baseline (0.8898)**. The model has positive net signal. Decomposing it: when the model recommends staying out, it is correct about 96% of the times (high specificity, rarely invents a stop), but out of the 367 real optimal stop windows, it only detects 38.96% (sensitivity), and of those, it matches the exact offset in 34.06% of the cases.

In summary: the previous structural bias —97% of predictions collapsing into `wait_laps = 0`— **has been eliminated**. The model went from not beating the trivial baseline to beating it in the relevant decision. The remaining limitation is the low sensitivity, expected given the scarcity and noise of real stop windows and the proxy nature of the `success_score`.

### Improvement Roadmap (From Lower to Higher Effort)

1.  **Sample Weighting (`sample_weight`) in Layer 2 (Completed):** Implemented a dynamic weighting of ~6.24x for groups with real stops, successfully raising binary accuracy on stop groups from 36.78% to 38.96%, and exact offset accuracy from 22.07% to 34.06%.
2.  **Margin in Inference:** Recommending `NO_PIT` only if its score exceeds the best stop candidate by a threshold; shifts the operating point toward calling more pit stops without retraining.
3.  **Two-stage Model:** A binary classifier "¿pit in the window?" followed by the offset ranker only on groups predicted to stop. Attacks the imbalance at the root.
4.  **Better Labels / PPO:** The proxy `success_score` is the ceiling of the supervised system. The PPO line natively models the sequential pit stop decision (including staying out) by evaluating the reward of the simulated race, solving the counterfactual without depending on historical observations. It is the most rigorous and high-effort path; currently, only the hyperparameter sweep exists (`ppo_best_hyperparameters.joblib`), without a trained agent.