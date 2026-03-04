# Market Health Index — Pipeline Documentation (V5.0)

## Project Overview

This project implements a data-driven **Market Health Signal** for the Vietnamese Stock Market. The pipeline uses **Wasserstein Distance (W1)** to select the most predictive features and employs **Machine Learning (ML)** models to generate robust, noise-filtered regime signals (Bull/Bear).

**Main Entry Point:** `application/main.py` (Orchestrates Steps 1–5)

---

## 1. The 4 Market Health Pillars

The pipeline measures 4 foundational aspects of market health to serve as candidate features:

| Pillar                 | Measurement Logic                                                                                                       |
| :--------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| **Market Trend**       | Normalized price distance from the 50-day EMA of the Equal-Weighted (EW) Index.                                         |
| **Market Breadth**     | Percentage of VN100 symbols trading above their 200-day EMA (using dynamic membership).                                   |
| **Market Cycle**       | Mean KST-based market "seasons" (mapped to -1 to +1) averaged across the VN100 universe.                                |
| **Market Volatility**  | Annualized volatility of the EW Index returns, estimated via **EGARCH(1,1)-t** (or rolling std fallback).               |

---

## 2. The 5-Step ML-Enhanced Pipeline

The V5.0 pipeline moves from raw data to a predictive machine learning model in five discrete steps.

### Step 1 — Feature Engineering (`Step1_feature_measuring.py`)

Calculates the 4 core pillars and applies normalization to prepare them for analysis.
- **Processing**: Winsorization (1st/99th percentiles) and Z-score normalization (rolling 252-day).
- **Output**: `application/step1_features/step1_all_features.csv`

### Step 2 — Ground Truth & W1 Scoring (`Step2_w1_score.py`)

- **Regime Detection**: Uses **L1 Trend Filtering** ($\lambda = 5$) on VNINDEX returns to identify historical "Bull" and "Bear" regimes.
- **Wasserstein Distance**: Measures the statistical "distance" between each feature's full distribution and its behavior during Bull/Bear regimes.
- **Correlation Filter**: Drops features with a correlation $> 0.85$ with higher-ranked ones.
- **Output**: `application/step2_w1_score/step2_w1_scores_all_features.csv` and `step2_bull_labels.csv`

### Step 3 — Component Selection (`Step3_component_selection.py`)

- **Selection**: Finalizes the top-performing features based on the W1 scores and "Kept" status from Step 2.
- **Output**: `application/step3_component_selection/step3_selected_5_features.csv`

### Step 4 — Composite Health Index (`Step4_market_health_index.py`)

- **PCA Weighting**: A rolling 252-day **Principal Component Analysis** determines dynamic weights for the selected features.
- **Signal Generation**: Uses a **Hysteresis Buffer** to generate binary ON/OFF signals based on the composite health score.
- **Output**: `application/step4_market_health_index/step4_market_health_final.csv`

### Step 5 — Predictive Modelling (`Step5_Predictive_modelling.py`)

- **ML Models**: Trains **Logistic Regression**, **XGBoost**, and **LightGBM** models on the selected features to predict future market regimes.
- **Validation**: Uses **Walk-Forward Validation** (5-year rolling window, monthly refit) to simulate real-world performance.
- **Final Signal**: Aggregates model probabilities with a hysteresis filter (Buffer: 0.05) to determine the tradable regime.
- **Output**: `application/step5_predictive_modelling/step5_evaluation_metrics.csv` and `step5_market_health_prediction_chart.png`

---

## 3. Key Output Catalog

| Asset                                          | Description                                                                      |
| :--------------------------------------------- | :------------------------------------------------------------------------------- |
| `step2_bull_regime_chart.png`                | Visualization of the L1 Trend Filtering "Ground Truth" labels.                   |
| `step2_w1_scores_all_features.csv`             | Full list of W1 scores and Kept/Dropped status.                                  |
| `step3_selected_5_features.csv`                | Final features chosen for the composite index and ML models.                     |
| `step4_market_health_final.csv`                | Composite scores, dynamic weights, and PCA-based signals.                        |
| `step5_evaluation_metrics_comparison.csv`      | Accuracy, Precision, Recall, and F1 scores for all ML models.                    |
| `step5_market_health_prediction_chart.png`     | **Primary Output**: VNINDEX price chart with ML-predicted Bull/Bear shading.     |
| `step5_walk_forward_oos_data.csv`              | Out-of-sample prediction results for walk-forward validation.                    |

---

*Generated Documentation for V5.0 — ML-Enhanced W1 Pipeline — Updated: 2026-03-03*
