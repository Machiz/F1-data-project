# Pit Stop Ranking Bias Audit (Layer 2)

Decision groups evaluated (race, driver, lap): **3331**
Model: `models/ranking_layer2_model.pkl` (features: 21)
Dataset: `data/processed/recommendation/pit_decision_candidates_v1.parquet`
Actions: `wait_laps` 0-5 (pit stop after waiting w laps) + `wait_laps=6` (NO_PIT / STAY_OUT)

## Ground Truth Best Action Distribution

| Action | wait_laps | n | % |
|---|---|---|---|
| Pit Now (0) | 0 | 139 | 4.17% |
| Wait 1 | 1 | 61 | 1.83% |
| Wait 2 | 2 | 42 | 1.26% |
| Wait 3 | 3 | 42 | 1.26% |
| Wait 4 | 4 | 42 | 1.26% |
| Wait 5 | 5 | 41 | 1.23% |
| NO_PIT | 6 | 2964 | 88.98% |

## Predicted Best Action Distribution

| Action | wait_laps | n | % |
|---|---|---|---|
| Pit Now (0) | 0 | 36 | 1.08% |
| Wait 1 | 1 | 22 | 0.66% |
| Wait 2 | 2 | 26 | 0.78% |
| Wait 3 | 3 | 40 | 1.20% |
| Wait 4 | 4 | 32 | 0.96% |
| Wait 5 | 5 | 47 | 1.41% |
| NO_PIT | 6 | 3128 | 93.91% |

## Metrics

| Metric | Value |
|---|---|
| Global Accuracy (exact action) | 0.9093 |
| Baseline "always NO_PIT (6)" | 0.8898 |
| Baseline "always pit now (0)" (historical reference) | 0.0417 |
| Binary Decision Accuracy (pit vs stay out) | 0.9147 |
| Groups with real optimal stop (optimal != 6) | 367 |
| Binary accuracy in those groups (detects a stop is needed) | 0.3896 |
| Exact accuracy in those groups (correct offset) | 0.3406 |

## Interpretation

The global accuracy (0.9093) beats the trivial baseline 'always NO_PIT' (0.8898) by 1.95 percentage points.

In the 367 groups where the real optimal decision was a pit stop (offset 0-5), the model correctly detects the need to stop (binary decision pit/stay out) in 38.96% of the cases and matches the exact offset in 34.06%. The binary decision is the primary metric of recommender utility; the exact offset is a stricter requirement.

With NO_PIT as an explicit action, the `wait_laps=0` candidate no longer receives the best label by default in laps without a real stop window, so 'staying out' is learned as a distinct choice rather than an artifact of labeling.

## Target Formulation (Corrected)

In `src/features/f1_recommender_pipeline.py`, each group (race, driver, lap) generates seven candidates. If there was a real stop in `lap + w` for any `w` in 0-5, that candidate receives its `success_score`; the rest of the offsets and NO_PIT receive `-2.0`. If there was no real stop in the 5-lap window, NO_PIT (`wait_laps=6`) receives the winning label (`0.0`) and offsets 0-5 receive `-2.0`. Thus, the neutral threshold of `0.0` for NO_PIT makes the model prefer staying out over executing a pit stop whose expected score is negative.

## Limitations and Future Work

The ground truth is derived from the labeling schema itself; the exact offset accuracy is bounded by the quality of the `success_score` proxy. The PPO line natively models the sequential pit stop decision (including the staying out action) by evaluating the reward of the simulated race, representing the most rigorous path once the agent is fully trained.
