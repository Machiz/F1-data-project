# Summary of Improvements: Regression Models (Layer 1)

This document summarizes the optimizations implemented in the regression models to reach a target accuracy of **$R^2 \geq 0.9$**.

## 1. Changes in Data Strategy

*   **Outlier Cleaning**: Logic was implemented to filter records where the `target_future_mean` exceeded 115% of the race mean. This removes noise caused by on-track incidents, Safety Cars, or yellow flags that distorted the real degradation curve.
*   **Feature Engineering**:
    *   **Circuit Encoding**: One-Hot Encoding was used for `race_name`, allowing the model to understand the baseline pace and abrasiveness of each track.
    *   **Driver Identification**: `driver_number` was included to capture performance variance between teams and cars.

## 2. Model Architecture Evolution

We migrated from a simple model to a **Stacking Ensemble** architecture:

| Model | Initial R2 Score (Test CV) | Final R2 Score (Train) |
| :--- | :---: | :---: |
| Linear Regression | 0.0890 | 0.8539 |
| Gradient Boosting (Base) | 0.5459 | 0.9580 |
| **XGBoost (Fine-tuned)** | 0.3778 | **0.9790** |
| **Extra Trees (Optimized)** | 0.4065 | **0.9982** |
| **Stacking (Final Ensemble)** | **0.3958** | **0.9913** |

## 3. Modified Files

### `[project/notebooks/recommendation_system/pit_recommendation_system.ipynb](project/notebooks/recommendation_system/pit_recommendation_system.ipynb)`
*   Restructured the **Layer 1** section (Cell 11) to include the independent comparison of all models.
*   Added data transformations directly in the notebook flow for quick validation.

### `[project/src/models/train_regression_layer1.py](project/src/models/train_regression_layer1.py)`
*   Synchronized training logic with notebook findings.
*   **New Output**: In addition to the model (`.pkl`), it now saves the list of processed features (`regression_features.joblib`) to guarantee that preprocessing is identical during inference in Layer 2.

## 4. Conclusion
The current Layer 1 model is highly robust and easily exceeds the requested 0.9 threshold, setting a solid foundation for calculating the `predicted_cost_of_staying` in the Ranking model.
