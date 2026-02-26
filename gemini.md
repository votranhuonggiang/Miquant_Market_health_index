# Market Health Index — Pipeline Documentation (V4.0)

## Project Overview

This project implements a data-driven **Market Health Signal** for the Vietnamese Stock Market. Unlike traditional technical indicators, this pipeline uses **Wasserstein Distance (W1)** to select the most predictive features from 8 major market pillars and aggregates them into a unified, noise-filtered regime signal.

**Main Script:** `application/market_health_w1_pipeline.py`

---

## 1. Data Ingestion & Index Foundation

The pipeline begins by fetching multi-source data from QuestDB and building the required market proxies.

- **QuestDB Sources**: Data is pulled from `raw_eod`, `raw_bond_yield`, `raw_money_flow`, and institutional trading tables.
- **Dynamic Membership**: Using `build_membership_map`, the pipeline tracks which stocks are in the VN30 or VN100 list for every month, ensuring index calculations (EW/CW) are free from survivorship bias.
- **Index Construction**:
  - **EW (Equal Weight)**: Daily mean returns of all active members.
  - **CW (Cap Weight)**: Returns weighted by `Price * Volume` from the previous day (T-1).

---

## 2. The 8 Major Pillars

Before the 4-step selection pipeline, the script measures 8 foundational aspects of market health:

| Pillar                 | Measurement Unit  | Measurement Logic                                                                                                       |
| :--------------------- | :---------------- | :---------------------------------------------------------------------------------------------------------------------- |
| **Breadth**      | % of Stocks       | Fraction of VN30/VN100 symbols trading above their 50-day EMA.                                                          |
| **Momentum**     | MACD Hist         | MACD Histogram (12, 26, 9) of the EW Index.                                                                             |
| **Sentiment**    | Z-Score           | Consensus of Sign-Breadth, High/Low Strength, and Stock-to-Bond Safe Haven returns smoothed via**Kalman Filter**. |
| **Cycle**        | Season (-1 to +1) | KST-based market "seasons" (Spring, Summer, Autumn, Winter) averaged across the universe.                               |
| **Money Flow**   | VND (Smoothed)    | 50-day rolling sum of net raw money flow.                                                                               |
| **Foreign Flow** | VND               | 50-day rolling sum of Foreign Net Buying/Selling.                                                                       |
| **Prop Flow**    | VND               | 50-day rolling sum of local Broker Proprietary Net Buying/Selling.                                                      |
| **Dispersion**   | Vol Spread        | Cap-weighted standard deviation of stock returns (internal market variation).                                           |

---

## 3. The 4-Step W1 Pipeline

The core of the V4.0 pipeline is a 4-step process that moves from raw signals to a final tradable regime.

### Step 1 — Feature Engineering (`compute_and_save_step1`)

Each of the 8 pillars is expanded into multiple **candidate sub-features** (slopes, stability, rolling means, and cross-products).

- **Result**: ~22 diverse mathematical representations of market behavior.
- **Storage**: `application/step1_features/`

### Step 2 — Ground Truth & W1 Scoring (`compute_and_save_step2`)

- **Regime Detection**: The script uses **L1 Trend Filtering** ($\lambda = 5$) on VNINDEX returns to identify historical "Bull" regimes objectively.
- **Wasserstein Distance**: It measures the "distance" between each feature's full history (**Prior**) and its behavior during **Bull** regimes.
- **Ranking**: Features that shift most significantly during bullish markets receive higher W1 scores.
- **Storage**: `application/step2_w1_scores/`

### Step 3 — Component Selection & Deduplication (`compute_and_save_step3`)

- **Pillar-Based Ranking**: The script groups the best features by their original Pillar and selects the Top 5 Pillars.
- **Collision Avoidance**: It selects the Top 2 non-redundant features per pillar.
- **Correlation Filter**: Greedy deduplication drops any feature with a correlation $> 0.85$ with a higher-ranked one.
- **Result**: Exactly **10 Optimized Features**.
- **Storage**: `application/step3_component_selection/`

### Step 4 — Composite Health & Signal Generation (`compute_and_save_step4`)

- **PCA Weighting**: A rolling 252-day **Principal Component Analysis** determines dynamic weights for the selected 10 features.
- **Vol-Adaptive Hysteresis**:
  - **GARCH(1,1)**: Estimates conditional volatility of the market.
  - **Hysteresis**: Sets a "buffer" to turn signals ON/OFF. High volatility increases the buffer (requiring more conviction).
- **Consensus Scaling**: If all 10 features agree, the buffer is tightened (more responsive).
- **Signal**: Returns **1 (Bullish/ON)** or **0 (Bearish/OFF)**.
- **Storage**: `application/step4_market_health_index/`

---

## 4. Key Output Catalog

| Asset                              | Description                                                                      |
| :--------------------------------- | :------------------------------------------------------------------------------- |
| `step2_bull_regime_chart.png`    | Visualization of the L1 Trend Filtering "Ground Truth" labels.                   |
| `step3_selected_10_features.csv` | List of the final 10 features chosen by the W1 algorithm.                        |
| `step4_market_health_final.csv`  | The master dataset containing composite scores, weights, GARCH vol, and signals. |
| `step4_market_health_chart.png`  | **Primary Output**: VNINDEX price chart with Bull/Bear background shading. |
| `step4_summary.txt`              | Executive summary including transition count and current regime status.          |

---

*Generated Documentation for V4.0 — Unified W1 Pipeline — Updated: 2026-02-26*
