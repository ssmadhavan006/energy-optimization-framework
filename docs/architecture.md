# EnergyOptAI — System Architecture

## Overview
Five-stage integrated pipeline transforming raw CNC machining
data into explainable, Pareto-optimal machining recommendations.

## Pipeline Architecture

  ┌─────────────────────────────────────────────────────────────┐
  │                     EnergyOptAI Pipeline                    │
  └─────────────────────────────────────────────────────────────┘

  STAGE 1: DATA LAYER
  ┌───────────────────────────────────────────────────────────┐
  │  Dataset A           Dataset B          Dataset C         │
  │  Mendeley            Kaggle             Bosch UCI         │
  │  (Energy + Time)     (Surface Ra)       (Validation)      │
  └─────────────┬──────────────┬───────────────┬─────────────┘
                └──────────────▼───────────────┘
                       src/data/loaders.py

  STAGE 2: PREPROCESSING
  ┌───────────────────────────────────────────────────────────┐
  │  Outlier Removal (IQR)                                    │
  │  Missing Value Imputation (median)                        │
  │  Categorical Encoding (label)                             │
  │  Feature Engineering (MRR, SEC, cutting speed)           │
  │  Train/Test Split 80:20                                   │
  │  MinMax Scaling (fit on train only)                       │
  └───────────────────────────────────────────────────────────┘
                       src/data/preprocessors.py
                       src/data/feature_engineering.py

  STAGE 3: ML PREDICTION LAYER
  ┌───────────────────────────────────────────────────────────┐
  │  Baseline Models         Ensemble Models                  │
  │  ├─ LinearRegression     ├─ XGBoost Regressor            │
  │  └─ SVR                  ├─ CatBoost Regressor           │
  │                          └─ Random Forest Regressor       │
  │                                                           │
  │  Three parallel model sets (one per target):             │
  │    Model Set A → Predict f1: Energy (kWh)                │
  │    Model Set B → Predict f2: Surface Roughness (Ra μm)   │
  │    Model Set C → Predict f3: Machining Time (s)          │
  │                                                           │
  │  Tuning: Optuna Bayesian Optimization (5-fold CV)        │
  └───────────────────────────────────────────────────────────┐
                       src/models/

  STAGE 4: EXPLAINABILITY LAYER
  ┌───────────────────────────────────────────────────────────┐
  │  SHAP TreeExplainer on best model per target             │
  │  ├─ Global: Feature importance bar chart                  │
  │  ├─ Global: Beeswarm summary plot                        │
  │  ├─ Global: Dependence plots (top 5 features)            │
  │  └─ Local: Waterfall plot per individual sample          │
  │                                                           │
  │  Output: Which parameters drive energy waste             │
  └───────────────────────────────────────────────────────────┘
                       src/explainability/shap_analysis.py

  STAGE 5: MULTI-OBJECTIVE OPTIMIZATION
  ┌───────────────────────────────────────────────────────────┐
  │  NSGA-II (pymoo)                                         │
  │  Decision variables: [spindle_speed, feed_rate,          │
  │                        depth_of_cut, coolant_on]         │
  │  Objective functions: surrogate = trained ML models      │
  │    Minimize f1: Energy prediction model output           │
  │    Minimize f2: Roughness prediction model output        │
  │    Minimize f3: Time prediction model output             │
  │  Population: 100  │  Generations: 200                    │
  │  Output: Pareto front (N non-dominated solutions)        │
  └───────────────────────────────────────────────────────────┘
                       src/optimization/nsga2_optimizer.py

  STAGE 6: MCDM RANKING
  ┌───────────────────────────────────────────────────────────┐
  │  TOPSIS on Pareto solutions                              │
  │  Criteria weights (configurable, default):               │
  │    w_energy = 0.50                                       │
  │    w_time   = 0.30                                       │
  │    w_rough  = 0.20                                       │
  │  Steps: normalize → weight → ideal best/worst →          │
  │         distance → closeness coefficient → rank          │
  │  Sensitivity analysis: 5 weight configurations          │
  └───────────────────────────────────────────────────────────┘
                       src/optimization/topsis.py

  STAGE 7: OUTPUT
  ┌───────────────────────────────────────────────────────────┐
  │  Recommended: Spindle Speed, Feed Rate,                  │
  │               Depth of Cut, Coolant Setting              │
  │  Predicted:   Energy (kWh), Time (s), Ra (μm)           │
  │  Comparison:  vs dataset average baseline                │
  │  Savings:     % energy reduction                         │
  └───────────────────────────────────────────────────────────┘

## File-to-Stage Mapping
  Stage 1-2: src/data/loaders.py, preprocessors.py, feature_engineering.py
  Stage 3:   src/models/*.py
  Stage 4:   src/explainability/shap_analysis.py
  Stage 5:   src/optimization/nsga2_optimizer.py
  Stage 6:   src/optimization/topsis.py
  Utils:     src/utils/logger.py, config.py, validators.py

## Evaluation Framework
  ML Metrics:     R², RMSE, MAE, MAPE per model per target
  Optimization:   Hypervolume indicator, Spread/Diversity
  MCDM:          Closeness coefficient, rank stability
  Baseline comp: All metrics vs LinearReg and SVR
