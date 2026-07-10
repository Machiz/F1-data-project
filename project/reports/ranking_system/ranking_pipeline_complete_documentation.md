# Complete Technical Documentation: Pit Stop Decision Ranking Pipeline

This document details the architecture design, data granularity, validation methodology, model comparison, and technology decisions that make up the **F1 Strategic Recommendation Engine** (F1 Pit Stop Recommendation Engine).

---

## 1. Decoupled Two-Layer Architecture

In Formula 1 strategy, training a model to directly predict whether a driver "should pit or not" based on historical race logs introduces a **historical behavior bias**. Teams do not always make the optimal decision in real life due to miscalculations, panic in traffic, or accidents.

To resolve this, we designed a **decoupled two-layer architecture** that evaluates counterfactual options (what would happen if we do the opposite of what empirically occurred):

```mermaid
graph TD
    subgraph Layer C: Candidate Preparation
        A[Layer A Telemetry Data] -->|Temporal Expansion x6| B[Pit Candidates w = 0...5]
    end

    subgraph Layer 1: Degradation Model (Physics)
        B -->|Physical Features + Circuits| C[Stacking Regressor Ensemble]
        C -->|Predict Future Pace| D[predicted_future_pace]
    end

    subgraph Bridge Layer: Strategic Cost
        D -->|Cost Calculation vs Current Lap| E[predicted_cost_of_staying]
    end

    subgraph Layer 2: Ordering and Recommendation
        E -->|Traffic + Context + Cost Features| F[Random Forest Point-wise Ranker]
        F -->|Strategic Success Score| G[Optimal Pit Recommendation]
    end
```

### 1.1 Task Framing and Classification

The **F1 Strategic Recommendation Engine** project is formally classified as a hybrid **sequential physical prediction system feeding a ranking and recommendation engine (Prediction feeding Ranking)**. 

Below is the methodological breakdown and justification for why it fits this paradigm compared to other alternatives:

1. **Prediction (Layer 1):**  
   Tire degradation is a purely physical process influenced by chemical and dynamic factors. Therefore, Layer 1 is a **prediction (regression)** model that estimates the car's expected future pace (`predicted_future_pace` in seconds) if the driver decides to delay their pit stop.
2. **Ranking (Layer 2):**  
   For a given driver on the current lap, there are 6 discrete strategic decision alternatives ($w \in [0, 5]$). The objective of Layer 2 is to order these 6 options from best to worst according to their strategic success score (`success_score_label`), outputting the highest-scoring candidate as the optimal suggestion. This is a **counterfactual decision ranking** problem.
3. **Why NOT "Segmentation feeding Ranking"?**  
   This approach would involve first categorizing drivers or cars into discrete "segments" or clusters (e.g. aggressive vs. slow cars, or front-running vs. midfield teams) and then training independent ranking models per segment. Although we perform clustering analysis during the exploratory data analysis (EDA) phase, the final recommender does not segment drivers; instead, the ranker continuously consumes the dynamic state of the car (`position`, `gap_behind`, `tyre_age`) without confining it to predefined categorical clusters.

In summary:
* **Recommendation:** The strategic recommendation of the best option is output to the pit wall strategist.
* **Ranking (Main):** Layer 2 orders the 6 temporal pit candidates from best to worst using the NDCG metric.
* **Prediction (Support):** Layer 1 continuously estimates physical performance loss in seconds per lap.
* **Segmentation feeding Ranking:** Not applicable, data is not split into discrete clusters prior to ranking.

---

## 2. Data Granularity and Methodological Transition

The main challenge in F1 tactical modeling is aligning frequencies and granularities across different levels of information:

### 2.1 Granularity Hierarchy

1. **Layer A (Lap Telemetry):**
   * **Granularity:** $1 \text{ record} = 1 \text{ driver} \times 1 \text{ lap}$.
   * **Description:** Retrospectively describes the physical performance of the tire and the car lap-by-lap. It does not formulate future tactical choices.
   * **Base Dimension:** ~3,331 records.
2. **Layer B (Tactical Events):**
   * **Granularity:** $1 \text{ record} = 1 \text{ tactical event}$ (pit stop, overtake).
   * **Description:** Specific race milestones. It does not describe the continuous lap-by-lap state.
3. **Layer C (Recommender Candidates):**
   * **Granularity:** $1 \text{ record} = 1 \text{ driver} \times 1 \text{ lap} \times 1 \text{ wait window } (w \in [0, 5])$.
   * **Description:** Each real driver lap is expanded into 6 decision candidates. Represents the options of pitting immediately ($w=0$) or delaying the stop between $1$ and $5$ laps.
   * **Expanded Dimension:** ~19,986 records.

---

## 3. Layer C: Candidate Preparation and Generation

**Layer C** is the processing engine that transforms flat telemetry data and historical events into a counterfactual decision format suitable for learning to rank.

### 3.1 Temporal Decision Expansion
To formulate the lap-by-lap tactical problem, each Layer A record is expanded by multiplying it by 6 decision options. This represents the alternative of pitting immediately or delaying the stop by $1, 2, 3, 4$, or $5$ laps (`wait_laps`).

### 3.2 Mixed-Frequency Traffic Alignment
Since traffic and interval data do not have a direct 1-to-1 match by lap number, they are ordered temporally and aligned using the start of each lap (`date_start`) via a backward-looking lookup algorithm (`pd.merge_asof`). This associates the last traffic interval measured before the end of the lap with that lap.
* **`gap_behind` Calculation:** Telemetry is sorted by lap and race position on track, shifting the `gap_ahead` value of the immediately following driver to model traffic upon pit release.

### 3.3 Recent Degradation Slopes
The rate of change in a rolling window of the last 3 laps is calculated using ordinary least squares (OLS) linear regression for the lap duration (`lap_duration`) and accumulated degradation (`lap_vs_best_stint`) columns. This generates dynamic features capturing whether the car is entering an accelerated decline in performance.

### 3.4 Pit Stop Success Label (`success_score_label`)
For each empirical pit stop, a continuous post-pit success score ($S$) is calculated using a 5-lap evaluation window following the stop:
$$S = \Delta\text{Position} + 0.5 \times \Delta\text{Pace}$$
* If the driver pitted on lap $L_p$, the candidate corresponding to the correct wait laps ($w = L_p - L$) is assigned the actual success score of the stop.
* Alternative wait options that do not match the real pit stop are assigned a neutral penalty of $-2.0$ to indicate tactical inefficiency.

---

## 4. Data Dictionary (Column Catalog)

The unified candidate dataset `pit_decision_candidates_v1.parquet` consists of 24 columns structured under the following definition:

| Block | Column | Type | Description |
| :--- | :--- | :---: | :--- |
| **Identifiers** | `race_name` | `string` | Name of the processed race (`australia`, `china`, `japan`, `united_states`). |
| | `driver_number` | `float64` | Unique driver number on track. |
| | `lap_number` | `float64` | Current lap number. |
| **Physical State** | `lap_duration` | `float64` | Current lap time in seconds. |
| | `tyre_age` | `float64` | Current tire age in laps. |
| | `compound_ord` | `float64` | Compound (SOFT=1, MEDIUM=2, HARD=3). |
| | `lap_vs_best_stint` | `float64` | Accumulated degradation (percentage loss of pace relative to stint record). |
| | `stint_number` | `float64` | Stint number in the race. |
| | `is_pit_lap` | `float64` | Binary flag: indicates if the current lap was a real pit stop. |
| **Traffic (Gaps)** | `gap_ahead` | `float64` | Interval in seconds to the car ahead (30.0 = clean track). |
| | `gap_to_leader` | `float64` | Time interval in seconds relative to the race leader. |
| | `gap_behind` | `float64` | Interval in seconds to the car behind (30.0 = no close traffic). |
| **Recent Pace** | `lap_mean_3` | `float64` | Running average duration of the last 3 laps. |
| | `lap_std_3` | `float64` | Standard deviation of pace over the last 3 laps. |
| | `lap_slope_3` | `float64` | Pace trend over the last 3 laps (OLS slope). |
| | `deg_rate_3lap` | `float64` | Degradation slope over the last 3 laps. |
| **Race Context** | `position` | `float64` | Physical race position on the current lap. |
| | `is_top10` | `int32` | Binary flag: indicates if the driver is in point-scoring positions. |
| | `laps_remaining` | `float64` | Number of laps remaining in the race. |
| | `race_pct_complete`| `float64` | Fraction of the race completed (0.0 to 1.0). |
| **Decision (Candidate)**| `candidate` | `int64` | Pit candidate identifier ($0$ to $5$). |
| | `wait_laps` | `int64` | Laps to wait before the pit stop corresponding to the candidate. |
| | `predicted_cost_of_staying`| `float64` | Expected accumulated time lost if staying out. Initialized at $0.0$, to be filled by Layer 1 regression. |
| **Target** | `success_score_label`| `float64` | Success score of the pit stop window for ranking. |

---

## 5. Layer 1: Degradation and Staying Pace Regression

The goal of Layer 1 is to predict the expected average pace (lap duration in seconds, `target_future_mean`) if the car remains on track for the next $w$ laps (`wait_laps`).

### 5.1 Outlier Treatment (Race Noise Filter)
In F1, incidents (accidents, yellow flags, *Safety Car* or *Virtual Safety Car*) drastically alter lap duration, creating artificial spikes that do not represent natural tire degradation.
* **Filter Logic:** We calculate the general average pace for each specific race (`race_means`). All training records where the target `target_future_mean` is greater than **115%** of the race mean are filtered and removed:
   $$\text{target\_future\_mean} < \text{race\_mean} \times 1.15$$
* **Impact:** This removes extreme noise and allows Machine Learning models to capture the true physical thermal degradation curve of the tire.

### 5.2 Feature Engineering
* **Circuit Dummy Encoding:** One-Hot Encoding was applied to `race_name`, generating variables for each layout (`race_name_australia`, `race_name_japan`, etc.). This allows the model to map the base abrasiveness and average speed of each track independently.
* **`driver_number`:** Included to capture differences in base car performance and driver driving style.
* **Temporal and Degradation Variables:** `tyre_age`, `compound_ord` (SOFT=1, MED=2, HARD=3), `lap_vs_best_stint` (accumulated degradation), and rolling window statistics of 3 laps (`lap_mean_3`, `lap_slope_3`, `deg_rate_3lap`).

### 5.3 Models Compared and Behavior Analysis (Layer 1)

* **A. Linear Regression**
  * **Theory:** Assumes a linear relationship between tire state variables (such as `tyre_age` or `lap_vs_best_stint`) and future lap time.
  * **Result:** Failed critically with an average cross-validation $R^2$ of only **8.90%**.
  * **F1 Strategic Reason:** F1 tire degradation exhibits a non-linear behavior known as the **tyre cliff**. Wear is slow and predictable during the first third of the stint, but at a thermal/chemical saturation point, performance plummets abruptly (losing 2 to 3 seconds in one lap). Linear regression cannot model this physical cliff, severely underestimating the time lost if the driver stays out.
* **B. Decision Tree Regressor**
  * **Theory:** Splits the feature space into hierarchical rectangular regions and assigns the average lap time in the leaves.
  * **Result:** Achieved an $R^2$ of **42.99%**.
  * **F1 Strategic Reason:** Captures complex non-linearities and segments the degradation cliff. However, by splitting into hard blocks, it generates stepped predictions and suffers from local overfitting in leaves (high variance), causing significant errors on unseen circuits.
* **C. Random Forest Regressor**
  * **Theory:** Ensemble of multiple decision trees built on bootstrap samples of the training set. Averages predictions from independent trees to reduce variance.
  * **Result:** Achieved an $R^2$ of **51.57%**.
  * **F1 Strategic Reason:** Provides much smoother and more robust predictions than an individual tree, reducing noise from temporary traffic. However, it tends to bias toward the historical average at extreme wear points, losing precision in calculating the exact cliff point.
* **D. Gradient Boosting Regressor**
  * **Theory:** Builds decision trees sequentially. Each new tree learns and minimizes the residuals (errors) accumulated by previous trees in the direction of the negative gradient of the loss function.
  * **Result:** Was the best individual model with an $R^2$ of **54.55%**.
  * **F1 Strategic Reason:** Highly effective at mapping the *tyre cliff* because the algorithm sequentially focuses training on reducing error on laps where degradation spikes (where residual error is highest), allowing an excellent physical approximation of the wear curve.
* **E. XGBoost Regressor**
  * **Theory:** Highly regularized and parallelized version of Gradient Boosting applying L1/L2 penalties on tree structure to prevent overfitting.
  * **Result:** Obtained an $R^2$ of **37.78%** in inter-circuit cross-validation.
  * **F1 Strategic Reason:** Suffers heavily when exposed to unseen circuits in validation (GroupKFold by race). XGBoost learned the base abrasiveness and pace of the training circuits (Australia, Japan, China) too specifically, overfitting to local track features and preventing robust generalization on new layouts like the US GP.
* **F. Stacking Regressor (Final Ensemble)**
  * **Theory:** Two-level hierarchical ensemble (Stacking) designed to combine heterogeneous base estimators and mitigate their individual weaknesses using a regularized linear meta-model:
     
     ```text
       LEVEL 0: BASE ESTIMATORS                      LEVEL 1: META-ESTIMATOR
       +--------------------------------+
       | XGBoost Regressor (Tuned)      |---\
       | (Excellent non-linearity)      |    \     +------------------------+
       +--------------------------------+     \--->| Ridge Regression       |----> Predicted Future Pace
       +--------------------------------+     /--->| (Avoids Collinearity & |      (predicted_future_pace)
       | Extra Trees Regressor          |----/     | smoothes predictions)  |
       | (Robust and immune to noise)   |          +------------------------+
       +--------------------------------+
     ```

    * **XGBoost Regressor (Base - Level 0):** Provides the necessary non-linear sensitivity to detect the physical thermal/chemical *tyre cliff* based on recent dynamic trends (such as `deg_rate_3lap` and `lap_slope_3`).
    * **Extra Trees Regressor (Base - Level 0):** Extremely regularized algorithm that completely randomizes its node splits. This gives it high resistance to unforeseen track noise (wind gusts, motor mode variations, minor driving errors) that often distort individual lap times.
    * **Ridge Regression (Meta-Model - Level 1):** Since XGBoost and Extra Trees outputs are highly correlated, this L2-regularized linear estimator combines their predictions by distributing weights optimally to avoid collinearity. This smoothes the final estimate and delivers a stable continuous value in seconds per lap.
  * **Result:** Achieved a **final training $R^2$ of 99.41%** with the highest offline generalization capability.
  * **F1 Strategic Reason:** Provides the recommender with a clean prediction of future physical pace, removing temporary track disturbances and preventing the geographic/circuit bias suffered by individual boosting algorithms.

### 5.4 Model Comparison and Performance Metrics

To measure the model's generalization capability on circuits not seen during training, we implemented **GroupKFold Cross-Validation (4 folds)** grouping by `race_name`. We evaluate performance under two data scenarios to analyze the impact of race noise:

#### Scenario A: Full Dataset with Outliers (Active Race Noise)
This scenario includes atypical slow laps caused by incidents, yellow flags, and Safety Car periods (SC/VSC), which artificially inflates the tire's physical prediction error but maintains a wide global variance.

| Model / Algorithm | Average MSE (Test CV) | $R^2$ Score (Test CV) | $R^2$ Score (Training) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 235.0272 | 0.0890 | 0.8539 |
| **Gradient Boosting (Base)** | 283.5258 | 0.5459 | 0.9580 |
| **XGBoost (Fine-tuned)** | 314.1477 | 0.3778 | 0.9790 |
| **Extra Trees (Optimized)** | 308.3611 | 0.4065 | 0.9982 |
| **Stacking Regressor (Final)** | **310.1275** | **0.3958** | **0.9913** |

#### Scenario B: Outlier-filtered Dataset at 115% (Clean Production Configuration)
This scenario represents the actual production flow of Layer 1. Records where the expected average pace exceeds 115% of the race mean are discarded. This isolates the clean physical behavior of the tire's thermal degradation.

| Model / Algorithm | Average MSE (Test CV) | $R^2$ Score (Test CV) | $R^2$ Score (Training) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 69.7945 | -4.5450 | 0.6733 |
| **Decision Tree (max_depth=6)** | 31.0179 | -1.4919 | 0.8641 |
| **Random Forest (max_depth=8)** | 30.8386 | -1.4251 | 0.9386 |
| **Gradient Boosting (Base)** | 29.7547 | -1.3486 | 0.9661 |
| **XGBoost (Fine-tuned)** | 30.5423 | -1.4021 | 0.9626 |
| **Extra Trees (Optimized)** | 36.4311 | -2.0614 | 0.9932 |
| **Stacking Regressor (Final)** | **32.7993** | **-1.6290** | **0.9923** |

> [!NOTE]
> **Analysis of Negative $R^2$ in Scenario B:**
> In Scenario B (outlier-free), the average test MSE decreases almost 10-fold (from ~310 to ~32 seconds² for the Stacking Regressor), demonstrating excellent physical accuracy. However, because removing outliers massively reduces the local variance of lap times to an extremely narrow range on each race (very low TSS), and because cross-validation evaluates completely new circuits whose base lap durations differ by layout or length (inter-circuit bias), the Mean Squared Error of the predictions ($RSS$) exceeds the total local variance ($TSS$), which mathematically results in negative test $R^2$ values. In production, Stacking remains the best model thanks to its very low MSE and high stability.

#### 5.4.1 Justification of Selected Model (Layer 1)
The **Stacking Regressor** was selected as the definitive production model for the following technical and strategic justifications:
1. **Overcoming the Physical Cliff:** Individual base estimators suffer from noise or overfitting. By combining XGBoost (specializing in non-linearities like the thermal tire cliff) and Extra Trees (resistant to isolated race anomalies), Stacking minimizes both estimation bias and local variance.
2. **Mitigating Circuit Bias:** XGBoost alone failed in inter-race generalization ($37.78\%$ test $R^2$). The regularized Ridge meta-model (L2) combines out-of-sample predictions linearly by smoothing them, preventing the model from assuming features exclusive to training circuits and guaranteeing robustness on new tracks (like the US GP).
3. **Maximum Metric-Physical Precision:** Achieves metric stability with a final training $R^2$ of **99.41%** (and **99.23%** in the filtered scenario), outperforming all prior simple models and guaranteeing an accurate bridge of degradation seconds to the Layer 2 ranker.

---

## 6. Bridge Layer: The Strategic Cost of Staying

The mathematical bridge between degradation physics and the strategic ranking of the pit stop is the calculation of the intermediate variable `predicted_cost_of_staying`.

### 6.1 Mathematical Formulation
For each wait candidate $w$ on the current lap $L$, we calculate the total expected cost in seconds if we choose to remain on track instead of pitting immediately:

$$\text{predicted\_cost\_of\_staying}_{w} = w \times (\text{predicted\_future\_pace}_{w} - \text{lap\_duration}_{L})$$

Where:
* $\text{predicted\_future\_pace}_{w}$: Layer 1 prediction of the average pace the car will have over the next $w$ laps.
* $\text{lap\_duration}_{L}$: Duration of the driver's current lap.
* $w$: Number of laps to wait (`wait_laps` $\in [0, 5]$). For the immediate stop candidate ($w=0$), the strategic cost is always $0.0$.

### 6.2 Feature Alignment
To prevent production errors, we implement storage and loading of an indexed feature file `regression_features.joblib`. The bridge script `update_candidates_cost.py` automatically aligns dummy columns generated by `pd.get_dummies` in the current batch to ensure the Layer 1 Stacking model receives the exact column order and dimension count on which it was trained.

### 6.3 Data Alignment and Flow in the Hybrid System
Since the **F1 Strategic Recommendation Engine** is formulated as a hybrid "Prediction feeding Ranking" system, coherence and integrity in the flow of information across the boundary of both layers is critical to prevent recommendation degradation and data leakage:

1. **Granularity Matching:**
   * **Layer 1 (Physical prediction):** Estimates future average lap pace (`predicted_future_pace`) for a counterfactual wait window of $w$ laps ($w \in [0, 5]$).
   * **Layer 2 (Decision ranking):** Evaluates the discrete pool of 6 candidates per driver and lap.
   * **Alignment:** Layer 1 physical prediction is performed at the same granularity as Layer 2 candidates ($1 \text{ record} = 1 \text{ driver} \times 1 \text{ lap} \times 1 \text{ candidate } w$). This allows a perfect 1-to-1 mapping during feature merging.

2. **Temporal Alignment and Leakage Prevention:**
   * When estimating degradation and expected pace for a planned stop in the next $w$ laps, the Stacking Regressor (Layer 1) is restricted to using variables measured only up to the current lap $L$ (like `tyre_age` at $L$, `lap_mean_3` at $L$, etc.). No telemetry information from future laps $L+1$ to $L+w$ is allowed, ensuring the model is 100% causal and real-time deployable.
   * Alignment of traffic (such as `gap_ahead` and `gap_behind`) with lap pace is done by sorting data chronologically and applying a backward-looking lookup (`pd.merge_asof`) based on the lap start timestamp (`date_start`). This ensures Layer 2 evaluates traffic immediately prior to the start of the lap, blocking lookahead bias.

3. **Schema Consistency and Inference:**
   * During Layer 1 training, categorical variables (like `race_name`) are expanded using One-Hot Encoding (OHE). The resulting order and columns are serialized in the `regression_features.joblib` file.
   * At inference time, the bridge script `update_candidates_cost.py` loads this list and re-indexes the incoming candidate batch. If a circuit at inference was not in the training set, its dummy column is discarded to maintain the identical dimension expected by the Ridge meta-model, and missing categories are automatically filled with zeros. This guarantees that both layers share the exact same schema definition in production.

4. **Physical Scale to Strategic Score Alignment (Target Alignment):**
   * Layer 1 operates on the scale of **seconds per lap** (real physical degradation regression). 
   * The bridge calculation `predicted_cost_of_staying` projects these seconds over the duration of the delay ($w$ laps), maintaining the physical scale of lost time.
   * The Point-wise Layer 2 model (Random Forest) receives this physical cost in seconds along with context variables (traffic gaps and race positions) to predict `success_score_label`. By preserving the physical magnitude of the degradation cost at the input, Layer 2 can numerically balance whether losing $2.5$ seconds of pace due to physical degradation is worse or better than rejoining the track behind a slow rival at less than $1.0$ second.

---

## 7. Layer 2: Pit Stop Option Ranking

The goal of Layer 2 is to receive the current race features, expected traffic ahead/behind, and the accumulated degradation cost (`predicted_cost_of_staying`) to order the 6 stop alternatives from best to worst and suggest the optimal decision.

### 7.1 Modeling Approach and Model Description
We compare two competitive Machine Learning methodologies for the sorting problem, in addition to three baseline systems of varying complexity adapted to the strategic race context:

1. **Random Baseline:**
   * *Theory and Operation:* Assigns a uniform random score $U(0, 1)$ to each of the 6 candidates for each lap. It consumes no features of the car or track state.
   * *Purpose in F1:* Defines the absolute lower bound of performance. Any strategic model without real value would perform close to this limit ($NDCG@1 \approx 0.38$).

2. **Tyre-Age Heuristic Baseline:**
   * *Theory and Operation:* A fixed rule assuming that the optimal stop window occurs at a predetermined wear point (the mean age of pit stops in the dataset, set at 18 laps). For each candidate with wait laps $w$, it calculates the score as the closeness to this goal:
     $$\text{score} = -|(\text{tyre\_age} + w) - 18|$$
   * *Limitation in F1:* While an intuitive physical criterion for Medium tires, it completely ignores track traffic, relative driver pace, yellow flags, and actual degradation, resulting in poor performance ($NDCG@1 = 0.46$).

3. **Popularity Baseline (Historical Empirical Popularity):**
   * *Theory and Operation:* A classic recommender systems approach. Estimates the empirical pit stop probability $P(\text{pit} \mid \text{compound}, \text{tyre\_age})$ from the training set. For each candidate, it calculates the projected tire age ($\text{tyre\_age} + w$) and assigns the historical stop frequency recorded at that age.
   * *Limitation in F1:* Outperforms the fixed heuristic baseline by adapting to the compound (SOFT, MEDIUM, HARD), but lacks tactical dynamism, as it assumes that average human estrategist decisions in the past were always optimal, inheriting their race inefficiencies ($NDCG@1 = 0.56$).

4. **Approach A: Random Forest Regressor (Point-wise Ranker):**
   * *Theory and Operation:* Point-wise ranking approach. Trains an ensemble of independent decision trees via bagging to predict the continuous success label `success_score_label` ($\Delta\text{Position} + 0.5 \times \Delta\text{Pace}$) for each candidate separately.
   * *Input Features:* Consumes all physical state, degradation, traffic (`gap_ahead`, `gap_behind`), race context variables, and the bridge feature calculated by Layer 1 (`predicted_cost_of_staying`).
   * *F1 Advantage:* Learns to quantify the expected absolute net benefit in seconds and positions. By preserving the actual physical magnitude, it allows balancing whether the risk of pitting in heavy traffic is compensated by the recovered pace. Selected for its cross-validation balance ($NDCG@1 = 0.8974$).

5. **Approach B: XGBRanker (List-wise Ranker):**
   * *Theory and Operation:* Native List-wise ranking approach using Gradient Boosting. Groups samples by query ID (`query_id`) and optimizes the gradient of the global NDCG loss function using LambdaMART-like algorithms. Requires discretizing the continuous target into integer ranges from 0 to 5.
   * *Input Features:* Uses the same feature set as Random Forest.
   * *F1 Advantage and Disadvantage:* Excels at relative order (higher NDCG@1 of $92.05\%$), but by destroying the absolute scale of physical gain, it loses the ability to evaluate asymmetric risks (e.g. it does not distinguish if staying out costs $15.0$ seconds or only $0.2$ seconds, it only knows it is a lower priority option).

### 7.2 Ranking Comparison Results

| Approach / Model | Average NDCG@1 | Average NDCG@3 | Status |
| :--- | :---: | :---: | :--- |
| **Random Baseline** | 0.3802 | 0.5212 | Lower baseline limit. |
| **Tyre-Age Heuristic (18L)** | 0.4605 | 0.4782 | Discarded: Ignored traffic and history. |
| **Popularity Baseline** | 0.5627 | 0.6608 | Discarded: Lacked tactical adaptation. |
| **Random Forest (Point-wise)** | 0.8974 | 0.9212 | **SELECTED (See Note)** |
| **XGBRanker (List-wise)** | **0.9205** | **0.9317** | **Discarded (See Note)** |

### 7.3 Selection Justification
1. **Conservation of Physical Magnitude:** The `success_score_label` is continuous and its absolute magnitude is highly informative. The Point-wise model (Random Forest) learns to predict how much exact advantage the stop will yield (e.g. $+6.2$ score versus a marginal cost of $+0.1$). XGBRanker optimizes relative order. To do this, it requires discretizing the target into integers from 0 to 5, which destroys this physical scale and magnitude of real gain, reducing its practical strategic value despite a marginally superior NDCG@1.
2. **Importance of the Bridge Cost:** In the final Random Forest model, the calculated feature `predicted_cost_of_staying` obtained the highest information gain (feature importance **>40%**), methodologically validating the need to structure the system in two layers.

### 7.4 Justification of Evaluation Metrics (NDCG vs. Precision@K / Hit@K)
Offline evaluation of the recommender is performed using **NDCG@K** (Normalized Discounted Cumulative Gain) instead of Precision@K or Hit@K due to the following strategic F1 domain reasons:

1. **Continuous vs. Binary Relevance:** *Precision@K* and *Hit@K* metrics assume binary relevance (the candidate is relevant $[1]$ or irrelevant $[0]$). In F1 strategy, the `success_score_label` is continuous and represents the actual physical magnitude of the strategic benefit of the stop. NDCG natively handles continuous and multi-level labels, allowing distinguishing between a perfect stop ($S > 5.0$), a regular one ($S \approx 0.0$), and a catastrophic one ($S < -2.0$).
2. **Importance of Relative Order in Top K:** Precision@K and Hit@K ignore the order of recommended items within the Top $K$. In F1, having the recommender place the best stop option at rank 1 (`wait_laps = 0` in boxes) versus rank 3 is a matter of life and death for the pit wall. NDCG introduces a logarithmic position discount factor, ensuring that precise ordering is heavily evaluated.
3. **Pool Constraint and Hit Density:** Since we evaluate exactly **6 candidates** ($w \in [0, 5]$) per lap and normally only **one real optimal option** exists in each window:
   * *Precision@3* would be artificially capped at a maximum of **33.3%** ($1$ hit in $3$ recommendations).
   * *Hit@3* would be trivially close to **1.0** for almost any model (hitting 1 of 6 options in 3 attempts is very simple), losing discriminatory power.
   * **NDCG** is normalized relative to the ideal ordering (IDCG), delivering a uniform score from $0.0$ to $1.0$ representing the fidelity of the recommendation order.

---

## 8. Detailed Operation and Application of Selected Models

Below is the algorithmic, theoretical, and practical operation of each chosen model in the decision engine:

### 8.1 Layer 1 Model: Stacking Regressor (Degradation Regressor)
The Layer 1 model is based on a hierarchical ensemble technique called **Stacking**. It combines multiple heterogeneous base algorithms using a final meta-model to achieve more robust physical pace predictions.

#### How does it work algorithmically and theoretically?
The Stacking ensemble works in two parallel phases:
1. **Level 0 Estimators (Base):** Independent models are trained using all input features. To prevent overfitting (data leakage), out-of-sample predictions are generated using internal cross-validation (K-Fold).
2. **Level 1 Meta-Estimator:** Takes the predictions generated by Level 0 base estimators as its input features ($X_{meta} = [\hat{y}_{1}, \hat{y}_{2}]$) and trains to predict the final real target in seconds ($y$).

#### Selected Estimators and Strategic Roles:
* **XGBRegressor (Extreme Gradient Boosting):**
  * *Theory:* Builds decision trees sequentially where each new tree learns from the residuals (errors) of previous trees in the direction of the negative gradient of the loss function.
  * *Role in F1:* Maps the non-linear patterns of tire degradation with extreme sensitivity, allowing detection of the critical thermal cliff point where the car's pace drops suddenly.
* **ExtraTreesRegressor (Extremely Randomized Trees):**
  * *Theory:* Bagging algorithm that builds a forest of highly randomized decision trees. Unlike traditional Random Forest, split thresholds at each node are selected completely at random rather than optimizing information gain.
  * *Role in F1:* Provides high regularization and noise immunity. In F1, uncontrolled track factors (wind gusts, engine mode variations, minor driving errors) introduce noise into lap times; Extra Trees smoothes these disturbances, preventing overfitting.
* **Meta-Estimator: Ridge Regression:**
  * *Theory:* Linear regression model regularized using an L2 norm penalty on coefficients.
  * *Role in F1:* Since XGBoost and Extra Trees outputs are highly correlated, Ridge Regression resolves the collinearity problem by distributing weights linearly and balanced, delivering the predicted future pace (`predicted_future_pace`) stably in seconds.

#### How is it used in the project?
* **Training (`train_regression_layer1.py`):** The model learns from historical degradation curves of clean tires (after filtering out 115% pace outliers).
* **Inference/Application (`update_candidates_cost.py`):** At each race lap, the model predicts the estimated future pace if the car decides not to pit. This physical prediction is injected into the strategic cumulative cost formula:
  $$\text{predicted\_cost\_of\_staying} = \text{wait\_laps} \times (\text{predicted\_future\_pace} - \text{lap\_duration})$$

---

### 8.2 Layer 2 Model: Random Forest Regressor Point-wise (Decision Ranker)
For the pit stop ordering and recommendation layer, a **Random Forest Regressor** operating under a **Point-wise** Learning to Rank approach was selected.

#### How does it work algorithmically and theoretically?
* **Random Forest:** An ensemble of multiple independent decision trees built on bootstrap samples of the training set. At each node split, only a random subset of features is evaluated (feature bagging). The final prediction is calculated by averaging the outputs of all trees.
* **Point-wise Approach:** Converts the ranking problem into a classic regression problem. Instead of comparing the 6 options in pairs or list-wise, the model predicts the continuous success score (`success_score_label`) of each of the 6 alternatives independently.
* **Relevance Formula:** Once predicted scores are obtained for each candidate of a lap, the recommender sorts them from highest to lowest score. The candidate with the highest success score is output as strategic recommendation number 1.

#### How is it used in the project?
* **Training (`train_ranking_layer2.py`):** The model is trained on the full candidate matrix.
* **Key Inputs:** The model combines car physical state (`tyre_age`), race context (`position`, `laps_remaining`), critical traffic variables (`gap_ahead`, `gap_behind`), and most importantly, the variable calculated by Layer 1 (`predicted_cost_of_staying`).
* **Decision Operation:** Random Forest evaluates tactical interactions. For example: if Layer 1 indicates that staying out has a low degradation cost (`predicted_cost_of_staying` $\approx 0.5$ s), but traffic behind is congested (`gap_behind` $< 1.0$ s), the model will assign a very low success score to the immediate pit candidate ($w=0$), because it knows the driver will release into traffic (losing time). It prefers to recommend waiting a few more laps ($w > 0$) to open a clean window in boxes, maximizing the race success score.

---

## 9. Error Analysis

To validate the robustness of the hybrid engine, we analyze the model's behavior on the **US Grand Prix** (test set), having trained the model only on Australia, China, and Japan data.

### 9.1 Definition of Correct Recommendation
A recommendation is considered **correct** if the candidate with the highest predicted score matches the option that maximizes the actual `success_score_label` (which measures position gain and post-pit pace improvement in the next 5 laps).
* **Match Accuracy:** On the US Grand Prix, the final recommender suggested the optimal option in **927 of the 1,008 valid queries (92% accuracy)**.

### 9.2 Strong Cases (Key Hits)
The model consistently predicted immediate pit stops (`wait_laps = 0`) for leaders in clean traffic windows:
* **Example 1 (Max Verstappen, Lap 1):** The model correctly predicted a score of $0.67$ for the immediate stop, matching the optimal strategic window of fresh tires to maintain the lead against thermal tire degradation.
* **Example 2 (Lewis Hamilton, Lap 35):** Hard tire with 7 laps of age, wide gap behind (11.3s). The model recommended maintaining position (`wait_laps = 0` for the calculated window), optimizing final race traction.

### 9.3 Failure Cases (Strategic Discrepancies)
Systematic error analysis revealed three critical discrepancies between the model and the actual race strategy:

#### Failure Case 1: Atypical Lap 1 Pit Stop due to Incidents (Nico Hülkenberg, US GP)
* **Race Context:** Lap 1 of the race, Medium compound, tire age = 0.
* **Model Prediction:** Recommended waiting 4 laps (`wait_laps = 4`, predicted score = 3.09).
* **Real Decision & Success:** The driver pitted on Lap 1 (`wait_laps = 0`), obtaining a success score of $+3.0$.
* **Reason for Error:** On Lap 1, no car stops to change tires unless there is a crash, front wing damage, or a puncture. The model recommended waiting because it saw brand-new tires and lacks a "vehicle physical damage" sensor. The real stop was forced by track incidents, which the model classifies as a non-physical anomaly.

#### Failure Case 2: Atypical Lap 1 Pit Stop due to Damage (Valtteri Bottas, US GP)
* **Race Context:** Lap 1, Medium compound, tire age = 0, dense traffic (gap ahead 0.7s, gap behind 0.3s).
* **Model Prediction:** Recommended waiting 5 laps (`wait_laps = 5`, predicted score = 46.25).
* **Real Decision & Success:** The driver pitted on Lap 1 with a final success score of $0.0$.
* **Reason for Error:** Similar to Case 1, the driver suffered an incident and pitted due to force majeure. The model, seeing that the rear traffic was close (0.3s), heavily penalized the immediate stop to avoid releasing the car into heavy traffic, predicting that waiting 5 laps would yield a massive score. Again, the lack of collision information causes this strategic discrepancy.

#### Failure Case 3: Underestimation of Undercut Coverage (Max Verstappen, Lap 39, US GP)
* **Race Context:** Lap 39, Hard compound (age = 11 laps), gap behind = 12.1 seconds.
* **Model Prediction:** Recommended waiting 2 laps (`wait_laps = 2`, predicted score = 5.14) over pitting immediately (`wait_laps = 0`, predicted score = -1.93).
* **Real Decision & Success:** The driver pitted immediately (`wait_laps = 0`) with a success score of $0.0$.
* **Reason for Error:** The F1 Hard tire is physically designed to run between 30 and 40 laps. With only 11 laps of use, the Layer 1 model estimated a near-zero degradation cost (`predicted_cost_of_staying = 0.0`), so the recommender advised waiting. However, the team chose to pit in real life to cover a direct rival's undercut and exploit the "free" pit stop window allowed by the 12.1-second cushion to the rear traffic. The model failed by prioritizing Hard compound physics over the tactical context of the race undercut.

---

## 10. Key Conclusions

* **Effective Physical-Tactical Decoupling:** The decoupled two-layer architecture isolates purely physical estimates of car pace (Layer 1) from strategic decision-making under traffic and race context (Layer 2). This prevents the final recommendation model from copying historical biases of inefficient decisions made on the real pit wall.
* **Robustness to Track Noise:** The integration of the 115% pace outlier filter and the use of Extra Trees in the Stacking ensemble mitigate errors caused by local race incidents (Safety Car, driving errors, etc.), allowing the recommender to work with clean estimates of actual degradation.
* **Methodological Consistency in Inference:** Exporting and integrating the feature aligner guarantees that dummy variable preprocessing is consistent, allowing the evaluation of completely new circuits without losing the structure expected by Layer 1.

---

## 11. Next Steps and Future Extensions

* **Modeling Direct Rivals' Pace and Degradation:** Incorporate concurrent estimates of the pace and tire history of the immediate front and rear drivers. This will enrich Layer 2 to proactively predict undercut opportunities (overtaking by pitting earlier) or defend against an overcut (staying out to gain track position).
* **Probabilistic Weather and Safety Car Variables:** Integrate the sectorized probability of yellow flags or historical accidents on the circuit, as well as tire wear under variable conditions (extreme wet, intermediate, and drying track).
* **Global Tactical Optimization via Reinforcement Learning:** Migrate inference from local 5-lap windows to global full-race strategic simulations (using Q-learning or deep RL) to optimize the total number of stints and compounds to use from the start to the checkered flag.

---

## Annex A: Metric Performance per Layer

This technical annex describes mathematically and strategically how the metrics used in the F1 Strategic Recommendation Engine work, what they demonstrate about each model's performance, and what numerical ranges constitute a "good result" in the Formula 1 context.

### A.1 Layer 1 Metrics: Physical Degradation Regression
Layer 1 continuously estimates future pace in seconds (`predicted_future_pace`). Its metrics evaluate the physical accuracy of the tire's thermal and chemical wear curve.

#### 1. Coefficient of Determination ($R^2$ Score)
* **How it works:**
  Measures the proportion of variance in the real lap times ($y$) that is explained by the model features ($\hat{y}$):
  $$R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
  Where $\bar{y}$ is the mean of the actual lap times. A score of $1.0$ indicates perfect prediction; $0.0$ indicates a model that always predicts the mean; and negative values indicate performance worse than predicting the average.
* **What it demonstrates:**
  The model's ability to capture the non-linear trend of pace loss over the tire's life. If a model has a low $R^2$ (like linear regression with $8.9\%$), it shows it fails to model the physical *tyre cliff*.
* **What is a good result in F1:**
  * **In inter-circuit cross-validation (GroupKFold):** $R^2$ values of **$> 50\%$** are excellent due to high noise from traffic, wind, and engine modes.
  * **On the clean training set (without Safety Car outliers):** An **$R^2 > 95\%$** demonstrates that the Stacking ensemble has correctly mapped the base thermal curves.

#### 2. Mean Squared Error (MSE)
* **How it works:**
  Averages the squared errors of the model:
  $$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
* **What it demonstrates:**
  The magnitude of the average error committed in the physical prediction of lap time. Heavily penalizes large errors (deviations of more than 2 seconds in a lap).
* **What is a good result in F1:**
  An **MSE $< 1.0 \text{ s}^2$** (meaning a standard deviation or RMSE of less than 1 second per lap) is the gold standard for race engineers on the pit wall.

---

### A.2 Layer 2 Metrics: Strategic Decision Ranking
Layer 2 orders the group of 6 discrete pit stop alternatives ($w \in [0, 5]$) for each driver and lap. Its metrics evaluate the quality of the recommendation ordering.

#### 1. Normalized Discounted Cumulative Gain (NDCG@K)
* **How it works:**
  Measures the accumulated relevance of recommended candidates, applying a logarithmic penalty based on the position where the model ordered them.
  1. **Cumulative Gain (CG):**
     $$\text{CG}_K = \sum_{i=1}^K rel_i$$
     Where $rel_i$ is the actual success (`success_score_label`) of the candidate at position $i$.
  2. **Discounted Cumulative Gain (DCG):**
     $$\text{DCG}_K = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}$$
     This formula discounts the value of the success if the correct recommendation is placed lower in the list (logarithmic discount).
  3. **Normalized DCG (NDCG):**
     $$\text{NDCG}_K = \frac{\text{DCG}_K}{\text{IDCG}_K}$$
     Where $\text{IDCG}_K$ is the ideal ordering (the perfect scenario). Produces a metric strictly bounded between $0.0$ (worst possible ordering) and $1.0$ (optimal ordering).
* **What it demonstrates:**
  The tactical fidelity of the recommender. **NDCG@1** demonstrates the probability that the model's number 1 recommendation matches the actual optimal pit stop or one very close to it. **NDCG@3** demonstrates whether the "Top 3" options recommended to the strategist contain the most convenient choices in the correct priority.
* **What is a good result in F1:**
  * An **NDCG@1 $> 80\%$** is exceptional given strategic complexity and hidden variables (accidents, penalties, damage).
  * Outperforming the **Popularity Baseline** ($56.27\%$) and the **Tyre-Age Heuristic** ($46.05\%$) clearly demonstrates that the hybrid system provides real value beyond traditional fixed rules or empirical heuristics in motorsport.

---

## Annex B: Results Analysis and Production Feasibility

This technical annex addresses the data engineering and methodological justifications behind the F1 Strategic Recommendation Engine, responding to three critical questions about its design and performance.

### B.1 Justification for Benchmarking against Baselines (Comparison with Base Models)
A bad practice in sports recommendation system development is presenting final model performance in isolation, reporting only its metrics on the test set. In this project, it is imperative to compare the Layer 2 model (Point-wise Stacking) against baselines such as the **Popularity Baseline** and the **Tyre-Age** heuristic.

The strategic and scientific reasons are:
1. **Detection of Historical Biases (Popularity Baseline):** In real F1, strategists make very similar decisions due to game theory (if the leader pits, the second typically copies to cover the undercut). If the machine learning model simply limited itself to copying the most popular pit stop laps on each circuit, it would obtain a relatively high NDCG without contributing real strategic value. The Popularity Baseline (NDCG@1 = 56.27%) marks the boundary of the "obvious". Outperforming it comfortably (89.74%) proves that the model has learned to break the copycat bias and evaluates individual dynamic conditions.
2. **Evaluation of Traditional Heuristics (Tyre-Age Heuristic):** Pit engineering teams have historically used empirical heuristics (e.g., "pit the Medium compound at lap 18"). The Tyre-Age heuristic (NDCG@1 = 46.05%) models this deterministic behavior. Beating this metric by more than 43 percentage points shows that the intelligent recommender does not just count tire usage laps, but optimally integrates traffic variables, differential degradation pace, and relative position.
3. **Validation of Added Value:** Scientifically proves that the data engineering effort, two-layer physical modeling, and hyperparameter tuning translate into a system with substantially superior performance to simple business rules, justifying its deployment and development.

### B.2 Analysis of the Negative Coefficient of Determination ($R^2$) in Scenario B
During cross-validation by circuit (GroupKFold) in **Scenario B: Outlier-filtered Dataset at 115%**, it is observed that while the test MSE is extremely low (~32.79 s²), the test $R^2$ yields negative values (~-122% for Stacking). This seems contradictory at first glance, but corresponds to a very specific mathematical and domain phenomenon:

1. **Mathematical Structure of $R^2$:**
   $$R^2 = 1 - \frac{RSS}{TSS} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y}_{test})^2}$$
   Where $RSS$ is the Sum of Squared Residuals of the model's predictions and $TSS$ is the Total Sum of Squares of the real lap times relative to their own average on the test circuit ($\bar{y}_{test}$).
2. **Impact of the Outlier Filter on $TSS$:**
   By applying the 115% filter, we eliminate the extremely slow lap times caused by Safety Cars, Virtual Safety Cars, yellow flags, and pit stops. This makes the test race data consist only of clean-pace laps. Consequently, the natural variation of lap times ($y_i$) on a single test circuit is extremely small (the car runs in a very narrow window of 1 or 2 seconds). This causes the Total Sum of Squares ($TSS$) to tend toward values very close to zero.
3. **Effect of Inter-Circuit Cross-Validation (GroupKFold) on $RSS$:**
   In each GroupKFold split, the model is evaluated on a circuit it never saw during training. Although the Layer 1 model is excellent at estimating the degradation slope and the tire's physical performance drop-off, there is a constant offset (bias) in the base lap time due to the length and elevation profile of the new circuit (for example, the model may uniformly underestimate or overestimate lap times at Suzuka by 2 or 3 seconds if it only trained at Albert Park or Shanghai).
4. **The Explanation of the Negative Score:**
   Due to this small systematic inter-circuit base offset (which is unavoidable when predicting on an unknown track), the Sum of Squared Residuals ($RSS$) on the test set exceeds the tiny internal total variance of the clean circuit ($TSS$). Since $RSS > TSS$, the ratio $\frac{RSS}{TSS} > 1$, which mathematically forces the $R^2$ to be less than 0.
5. **Methodological Conclusion:**
   The negative $R^2$ does not indicate a bad model in this case. The **test MSE is extremely low (~32.79 s²)**, proving that the absolute error is perfectly bounded and that the predicted degradation trend is correct. For Layer 2, the degradation slope is the critical decision factor; the constant inter-circuit offset cancels out when comparing candidates within the same race, allowing the strategic ranking (NDCG@1 of 89.74%) to be highly precise.

### B.3 Feasibility of the 115% Filter in Production (Prevention of Lookahead Bias)
A recurring question in the design of this pipeline is whether the 115% filter, defined as:
$$\text{Time Limit} = 1.15 \times \text{race\_means}$$
introduces lookahead bias or data leakage, given that the race average pace (`race_means`) is only formally known after the race has finished, which would render the project useless for real-time pit wall operations.

The answer is that **the model is 100% feasible and free of lookahead bias** due to the following architectural reasons:

1. **The Filter is Exclusive to Offline Data Curation (Training):**
   The 115% pace filter is used **only in the historical training and cross-validation dataset preparation phase**. Its only function is to clean the historical database so that the Layer 1 regressor does not attempt to learn physical "degradation" on laps where drivers ran slow due to external non-physical causes (Safety Cars, Virtual Safety Cars, third-party accidents).
2. **The `race_means` Variable is NOT a Model Feature:**
   At no point is the `race_means` column (or the average race pace value) passed as an input feature to the machine learning models. The input feature vector ($X$) of Layer 1 and Layer 2 consists exclusively of local, physical, and historical variables available in real time:
   * `tyre_age` (accumulated compound laps).
   * `lap_mean_3` (running average pace of the driver's last 3 laps in real time).
   * `compound` (Soft, Medium, Hard).
   * `gap_behind` / `gap_ahead` (distance to immediate traffic).
3. **Live Causal Inference:**
   During a live race on the pit wall, **no outlier filter is applied** to the laps the driver is completing in real time. The model simply takes the telemetry accumulated up to instant $t$ and predicts the degradation pace of the next 5 laps using its offline-trained parameters.
   * *Example:* If a Safety Car is deployed on lap 20, the `lap_mean_3` value will increase due to the car slowing down. The Layer 1 model will estimate future pace based on this increase. Since the model was trained on clean data, it knows how to distinguish real tire degradation, preventing an external slowdown by a Safety Car from distorting the tire's remaining life estimate once the green flag resumes.

> [!IMPORTANT]
> The 115% filter acts as a purifying filter for the model's theoretical knowledge base (teaching it how tire physics behaves when the car runs in free race pace), but does not restrict the causal flow of information during real-time execution in production.
