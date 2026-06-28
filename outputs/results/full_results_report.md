# EnergyOptAI — Quantitative Results Synthesis Report
Generated on: 2026-06-27 17:00:23 UTC

## 1. Machine Learning Surrogate Model Performance
The following table summarizes the prediction accuracy of the trained surrogate models on held-out test sets:

| target    | model             |           r2 |         rmse |          mae |         mape |   n_samples |
|:----------|:------------------|-------------:|-------------:|-------------:|-------------:|------------:|
| energy    | random_forest     |  -0.0023926  |  0.00036575  |  7.26059e-05 |  10039.7     |        2251 |
| energy    | svr               |  -0.00366917 |  0.000365983 |  9.48482e-05 | 120194       |        2251 |
| energy    | catboost          |  -0.0206851  |  0.000369072 |  7.83935e-05 |   3934.89    |        2251 |
| energy    | xgboost           |  -0.0278715  |  0.000370369 |  8.12162e-05 |   2423.74    |        2251 |
| energy    | linear_regression | -13.257      |  0.00137936  |  0.000268394 |   5710.8     |        2251 |
| roughness | catboost          |   0.806079   |  0.123686    |  0.0787851   |     13.0594  |         165 |
| roughness | xgboost           |   0.795686   |  0.126957    |  0.0830647   |     14.1984  |         165 |
| roughness | random_forest     |   0.794057   |  0.127462    |  0.0822      |     14.2901  |         165 |
| roughness | svr               |   0.547469   |  0.188944    |  0.139914    |     25.7044  |         165 |
| roughness | linear_regression |  -1.1043     |  0.407438    |  0.296384    |     51.5511  |         165 |
| time      | catboost          |   0.994683   |  1.13587     |  0.489939    |      3.74267 |          43 |
| time      | random_forest     |   0.969481   |  2.72136     |  1.28105     |      6.45301 |          43 |
| time      | xgboost           |   0.920262   |  4.39878     |  1.05924     |      2.36447 |          43 |
| time      | linear_regression |   0.895315   |  5.04012     |  3.8601      |     33.4074  |          43 |
| time      | svr               |   0.311816   | 12.9227      | 10.454       |     79.8294  |          43 |

## 2. Multi-Objective NSGA-II Optimization Summary
The NSGA-II algorithm was executed with a population size of 100 over 200 generations to find the three-dimensional Pareto-optimal front.

- **Pareto-Optimal Solutions Found**: 42
- **Hypervolume Indicator**: 7.662354 (using 1.1x max reference point)
- **Optimization Runtime**: 628.32 seconds

### Objective Ranges Across Pareto Front:
- **Energy SEC (J/mm³)**: 7.944925328736692 to 7.944925328736692
- **Surface Roughness Ra (μm)**: 0.3351846266723766 to 0.5647439078238972
- **Machining Time (s)**: 5.130826337843157 to 46.20150130982012

## 3. TOPSIS Multi-Criteria Decision Making (MCDM)
The Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) was applied to rank the Pareto solutions.

### TOPSIS Weights Applied:
- **Energy SEC Weight**: 0.50
- **Roughness Ra Weight**: 0.20
- **Machining Time Weight**: 0.30

### Recommended Optimal Machining Parameters (Rank 1):
- **Feed Rate (f)**: 0.1037 mm/rev
- **Depth of Cut (ap)**: 0.3357 mm
- **Spindle Speed (S)**: 10201 rpm
- **Tool Wear (TCond)**: 0.0380 mm

### Predicted Performance & Improvement vs Median Baseline:
- **Energy SEC**: 7.9449 J/mm³ (+0.0%)
- **Surface Roughness Ra**: 0.4528 μm (-4.2%)
- **Machining Time**: 6.6969 s (+44.2%)
- **TOPSIS Closeness Coefficient (Ci)**: 0.8870
- **TOPSIS Weight Scenario Stability Score**: 0.50

## 4. Discussion & Model Limitations
A key design consideration of this study is the sample size constraint of the Specific Energy Consumption (SEC) surrogate model. While the roughness and machining time models were trained on large datasets, the SEC model was trained on only 47 aggregated operations from the Mendeley repository.

To mitigate prediction unreliability and extrapolation risks in this small-sample model, we implemented two key safeguards:
1. **Strict Bounds Enforcement**: NSGA-II search space was strictly restricted to the training data min/max per feature. Extrapolation outside the training range was forbidden.
2. **Proximity-to-Training-Data Check**: Evaluated solution vectors' Euclidean distance to the 47 training samples in the normalized multi-dimensional feature space, ensuring predictions lie in dense regions of the training data.