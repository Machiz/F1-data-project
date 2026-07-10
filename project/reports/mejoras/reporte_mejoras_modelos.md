# Pit Stop Recommender Models Improvement and Change Report

This report documents the technical changes made to the data pipeline, feature engineering, and training of the regression (Layer 1) and ranking (Layer 2) models to resolve the systematic pit stop bias and mitigate extreme class imbalance.

---

## 1. Bias Diagnosis and Root Cause

Before the modifications, the recommender presented a critical bias where it suggested entering boxes on almost any lap of the race, failing to predict the actual optimal lap.

### Cause 1: The Temporal Factor of the Stint
The model previously used the total race lap (`lap_number`) as a continuous linear variable. For a regressor or decision tree, the absolute lap number has no physical meaning of degradation unless it is coupled directly to the wear of the current tire. The model learned spurious biases by seeing pit stops on arbitrary laps, without understanding the life cycle of the tire.

### Cause 2: Extreme Class Imbalance (95/5 Rule)
In a typical F1 race, 95% of laps are staying out on track (`NO_PIT`) and only 5% contain real pit stops. When structuring this in a pointwise model with 7 candidates per group (waiting 0..5 laps vs. NO_PIT), the positive class represents barely 11% of the rows, while the rest are labeled with a constant penalty of `-2.0`. Without a balancing mechanism, classifiers optimize their loss by always predicting values close to `-2.0`, failing to learn the precise pit stop moments.

---

## 2. Specific Improvements in the Regression Model (Layer 1)

The regression model of **Layer 1** (a stacked ensemble of XGBRegressor + ExtraTreesRegressor with a Ridge meta-model) is tasked with predicting the future race pace (`predicted_future_pace`) if the driver continues on track. Three key improvements were implemented to optimize its physical accuracy:

### A. Non-linear Decoupling of Compounds (One-Hot Compounds)
* **Previous Problem:** The model relied on `compound_ord` (Soft = 1.0, Medium = 2.0, Hard = 3.0). Linear models (like the final Ridge regression) assumed that degradation scaled linearly with this order.
* **Solution:** By separating the compounds into binary variables (`compound_SOFT`, `compound_MEDIUM`, `compound_HARD`), the regression model now learns curves of degradation and base pace loss that are completely independent for each compound type. This is physically correct, as Soft tires degrade thermally much faster and differently than Hards.

### B. Tire Thermal Integral (`delta_time_loss`)
* **Previous Problem:** The model projected future pace using only instantaneous variables such as `tyre_age` and `lap_vs_best_stint`. This ignored the historical stress of the tire in the current stint (e.g., lockups or very slow laps that overheated the compound).
* **Solution:** The expansive mean `delta_time_loss` acts as an integral representation of accumulated wear in the stint. It allows the regression model to differentiate between a 15-lap tire that has run in clean air with constant degradation, and one that has suffered heavy traffic and thermal spikes.

### C. Future Traffic Projection (`pit_gap_ahead` / `pit_gap_behind`)
* **Previous Problem:** The regressor did not know what track conditions the driver would face at the target lap $L + w$.
* **Solution:** By including projected traffic window gaps, the regressor can adjust its pace prediction. If the window projects dense traffic (small gap ahead), the regressor predicts a slower pace due to loss of aerodynamic downforce (dirty air) and difficulty overtaking, aligning with F1 physics.

**Regression Results (Layer 1):**
The training $R^2$ score of the Stacking Regressor increased from **0.9928 to 0.9939**, demonstrating a greater capacity to capture the physical and traffic dynamics of the track.

---

## 3. Specific Improvements in the Ranking Model (Layer 2)

The **Layer 2** model (Point-wise Random Forest Regressor) takes the cost predictions from Layer 1 and decides which action (wait_laps 0..5 or staying out 6) has the highest probability of strategic success.

### A. Class Balancing via Weights (Sample Weights)
To mitigate the bias towards `NO_PIT` (caused by the 95/5 imbalance), a sample weight of **~6.24x** was dynamically calculated for instances of the positive class (actions representing the actual optimal pit stops). This prevents the Random Forest from choosing to stay out by default to minimize absolute classification error.

### B. Success Label Correction
Spurious default rewards for `wait_laps = 0` on normal laps were removed, penalizing all incorrect pit stop options uniformly with `-2.0` and assigning the neutral label `0.0` only to optimal stays (`NO_PIT`).

---

## 4. Impact on Evaluation Metrics

| Metric | Before | After | Impact of Improvements |
|---|:---:|:---:|:---:|
| **NDCG@1 (Model Comparison CV)** | 89.74% | **92.17%** | **+2.43%** |
| **Global Accuracy (Exact Action)** | 87.57% | **90.93%** | **+3.36%** (Beats the `always NO_PIT` baseline) |
| **Binary Decision Accuracy (Pit/Stay Out)** | 89.07% | **91.47%** | **+2.40%** |
| **Binary Accuracy on Pit Stop Groups** | 35.42% | **38.96%** | **+3.54%** |
| **Exact Accuracy on Pit Stop Groups** | 21.80% | **34.06%** | **+12.26%** (Critical hit of the exact lap) |
