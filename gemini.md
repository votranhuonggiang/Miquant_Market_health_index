# Market Health Index — Project Documentation

## Overview

This project generates a **Market Health Signal** for the Vietnamese stock market using a
multi-factor, data-driven consensus model. The pipeline aggregates 8 structural signals
(breadth, momentum, sentiment, cycle, money flow, foreign flow, prop flow, dispersion)
from multiple cap-size universes, selects the most discriminative features using the
**Wasserstein (Earth Mover) Distance**, and outputs a daily **Bull / Bear regime signal**.

**Main script:** `application/upgrade_synth_trend_indices.py` (V3.2)

---

## Pipeline — 16 Steps, End to End

### Step 1 — Data Fetching from QuestDB
All raw data is pulled from QuestDB via HTTP REST (`/exec`):

| Data | Query Table | Key Columns |
| :--- | :--- | :--- |
| VN100 universe membership | `ai_ranking_universe` | `symbol`, `timestamp` (WHERE universe='VN100', 2018–2025) |
| VN30 universe membership | `ai_ranking_universe` | same (WHERE universe='VN30') |
| VNMID membership | `raw_historical_list` | `is_vnmid = 1` |
| VNSMALL membership | `raw_historical_list` | `is_vnsml = 1` |
| Daily OHLCV for all universes | `raw_eod` | `timestamp`, `symbol`, `close`, `high`, `low`, `volume` |
| VNINDEX daily close | `raw_eod` | symbol = 'VNINDEX' |
| 16 structural indices | `raw_eod` | VN100, VNALL, HNXINDEX, VNINDEX, VN30, VNMID, … |
| Gov bond yield (VN10Y) | `raw_bond_yield` | `close` = yield in % |
| Net Money Flow | `raw_money_flow` | `symbol`, `net_money_flow`, `timestamp` |
| Foreign net trading | `raw_foreign_trading_detail` | `symbol`, `netvalue` |
| Proprietary net trading | `raw_proprietary_trading_detail` | `symbol`, `netvalue` |

---

### Step 2 — Monthly Membership Maps (`build_membership_map`)
For each universe (VN30, VN100, VNMID, VNSMALL), a dictionary is built:
```
{ Period('YYYY-MM', 'M'): set(symbols) }
```
- **Gap-filling**: If a month is missing, the previous month's membership is carried forward.
- **Forward-fill**: The last known membership is extended up to the latest data date.
- Result: Every trading day can be mapped to the exact set of eligible symbols for that month.

---

### Step 3 — Index Construction
Two index types are built for VN100 and VN30:

**Equal-Weight Index (`build_ew_index`)**
- Pivot prices to wide format → compute daily pct_change returns.
- Average returns only across that month's members (`aggregate_by_membership`).
- Cumulative product × 100 → index level starting at 100.

**Cap-Weighted Index (`build_cw_index`)**
- Weights = `close × volume` lagged by 1 day (T−1 market cap proxy).
- Weighted average return per day → cumulative product × 100.

Both are used as inputs to downstream components.

---

### Step 4 — 8-Factor Computation

#### Component 1: Market Breadth (`measure_market_breadth`)
- **Universe**: VN30 (membership-weighted).
- **Signal**: Fraction of stocks whose close > EMA(50).
- Per stock: binary flag (1 if above EMA, 0 otherwise).
- Monthly `aggregate_by_membership` → daily mean = breadth score ∈ [0, 1].

#### Component 2: Momentum (`measure_momentum`)
- **Universe**: VN30 EW Index close.
- **Signal**: MACD Histogram = (EMA12 − EMA26) − Signal(EMA9 of MACD).
- Positive histogram → upward momentum acceleration.

#### Component 3: Sentiment (`measure_sentiment`)
- **Universe**: VN100 stocks + VNINDEX + VN10Y bond.
- 3 sub-components:
  1. **Net Advancing**: sum of sign(daily return) across all VN100 stocks (breadth of daily moves).
  2. **New High − New Low**: count of stocks at 252-day highs minus those at 252-day lows.
  3. **Safe-Haven Demand**: 20-day return of VNINDEX minus 20-day return of bond PV proxy (`1000 / (1 + yield/100)^10`).
- Each sub-component is Z-scored (adaptive rolling window ≤ 252 days).
- Composite = sum of the 3 Z-scores → Z-scored again.
- Final smoothing via **Kalman Filter** (transition covariance = 1e-3) → removes short-term noise.

#### Component 4: Market Cycle (`measure_market_cycle`)
- **Universe**: VN100.
- **Signal**: KST (Know Sure Thing) indicator per stock → mapped to 4 seasons:
  - Summer (KST ≥ 0, rising): `+1.0`
  - Spring (KST < 0, rising): `+0.5`
  - Autumn (KST ≥ 0, falling): `−0.5`
  - Winter (KST < 0, falling): `−1.0`
- Cross-sectional mean across VN100 members → daily cycle score ∈ [−1, +1].

#### Component 5: Net Money Flow (`measure_money_flow`)
- **Universe**: VN30 (membership-filtered).
- Raw: daily `net_money_flow` per stock.
- 50-day rolling sum per stock to smooth noise.
- Sum across all VN30 members each month → total net capital flow (VND).

#### Component 6: Foreign Flow (`measure_institutional_flow`, label="foreign")
- **Universe**: Broad market (all symbols in `raw_foreign_trading_detail`).
- 50-day rolling sum of `netvalue` per stock → sum across all symbols.
- Positive = net foreign buying; negative = net foreign selling.

#### Component 7: Proprietary Flow (`measure_institutional_flow`, label="prop")
- Identical approach to Foreign Flow, using `raw_proprietary_trading_detail`.
- Tracks local broker/prop desk net activity.

#### Component 8: Dispersion Index (`measure_dispersion_index`)
- **Universe**: VN100 (membership-filtered, cap-weighted).
- Weights = `close × volume` (T−1 market cap proxy).
- Daily **cap-weighted standard deviation** of log-returns across VN100 members.
- High dispersion = stocks moving independently (healthy differentiation or bottoming).
- Low dispersion = herding / panic (fragile market).

---

### Step 5 — Alignment & Z-Score Normalization
- **Master timeline**: VNINDEX daily close (used as the date spine).
- All 8 components are aligned to VNINDEX dates via `ffill()` + `dropna()`.
- Each raw component is **rolling Z-scored** (`compute_z_score`):
  - Window = min(252, max(N/3, 15)), min_periods = window/4.
  - Z = (x − rolling_mean) / rolling_std.
  - **Winsorized** at ±3 σ to suppress outliers.
- Resulting columns: `breadth_z`, `momentum_z`, `sentiment_z`, `market_cycle_z`, `money_flow_z`, `foreign_flow_z`, `prop_flow_z`, `dispersion_z`.

---

### Step 6 — PCA Dynamic Weighting (`compute_pca_weights`)
- **Goal**: Assign data-driven weights to the 8 Z-scored components instead of equal weights.
- **Method**: Rolling PCA (window = 252 days, min_chunk = 126 rows):
  - At each date `i`, fit PCA on the last 252 rows.
  - Extract PC1 loadings (sign-corrected so their sum > 0).
  - These loadings become the weights for that day.
- Weights are forward-filled and default to `1/8` if too few data points.
- **Composite Health** = weighted sum: `Σ (weight_c × component_z_c)` for all 8 pillars.

---

### Step 7 — GARCH Volatility (`compute_garch_rolling`)
- **Input**: VN100 EW Index daily log-returns × 100.
- **Model**: Expanding-window EGARCH(1,1) with Student-t errors (via `arch` library).
- **Schedule**: Re-estimate parameters every 63 days (quarterly). Burn-in = 252 days.
- **Output**: `sigma_ann` = annualised conditional volatility (× √252).
- Fallback to simple rolling std if GARCH fails or data is too short.
- Used in Step 8 to scale the switching threshold dynamically.

---

### Step 8 — Hysteresis Regime Switching (`apply_vol_weighted_hysteresis`)
- **Input**: `composite_health` score + `sigma_ann` volatility + `agreement` (signal consensus).
- **Volatility multiplier**: `vol_mult = sigma_ann / rolling252_mean(sigma_ann)`. High vol → wider buffer.
- **Consensus scaling**: `agreement = |mean(sign(component_z))|` across all 8 pillars (0→1). Low agreement → wider buffer.
- **Effective buffer** = `BASE_BUFFER × vol_mult × (2 − agreement)`.
  - Default `BASE_BUFFER = 0.5`.
- **State rules**:
  - Score > buffer → state = 1 (Bullish).
  - Score < −buffer → state = 0 (Bearish).
- **Min-Hold**: Must stay in a state for at least **5 days** before flipping (prevents whipsaws).
- Output: `health_signal_v2` = 1 (Bullish) or 0 (Bearish).

---

### Step 9 — Output: V2 CSV + Chart
- **CSV**: `market_health_v2_final.csv` — all columns including Z-scores, weights, GARCH vol, signal.
- **Chart**: VNINDEX price with green (Bullish) / red (Bearish) background shading + Buy (▲ green) / Sell (▼ red) flip markers.

---

### Step 10 — Noise Diagnostics
For each of the 8 Z-scored components, prints:

| Metric | Formula | Interpretation |
| :--- | :--- | :--- |
| DailyΔ Std | `diff().std()` | How much the component jumps day-to-day |
| Sign-Flip% | `(sign().diff() ≠ 0).mean() × 100` | % of days where the sign reverses |
| AutoCorr(1) | `autocorr(lag=1)` | Persistence; low = noisy |
| PCA Weight | Average rolling PC1 loading | Contribution to composite |

Flag: ⚠ NOISY if `sign_flip > 10%` or `autocorr < 0.90`.

---

### Step 11 — Multi-Universe Profile
For each of VN30, VN100, VNMID, VNSMALL:
- Current breadth (% stocks above EMA-50).
- 1-month return and 3-month return of the EW Index.
- Regime label: **BULLISH** (breadth > 50%) or **CAUTIOUS** (breadth ≤ 50%).

---

### Step 12 — Wasserstein Feature Selection (`run_wasserstein_feature_selection`)
This step uses the **full historical sample** (no train/test split) to find which engineered
sub-signals best statistically separate Bull days from non-Bull days.

#### 12A — Ground Truth Labels (`define_bull_base_labels`)
- **Index Used**: VNINDEX only (filtered from `load_multiple_indices()`).
- **Method**: L1 Trend Filtering (`trend_filtering`) on VNINDEX daily returns, with **λ = 5**.
  - Solves: `min ‖r − β‖² + λ ‖D·β‖₁` (piecewise-linear smoothing on returns).
  - Solved via `cvxpy` (CLARABEL solver preferred).
- **Label**: day is **Bull Base (1)** if smoothed β > 1e-5, else **0**.
- Label covers the full available date range.

#### 12B — Feature Engineering (`extract_candidate_features`)
~22 sub-signals are derived from the 8 Z-scored pillars:

| Pillar | Sub-Signals |
| :--- | :--- |
| Breadth | `breadth_mean20`, `breadth_stability`, `breadth_pct_high` |
| Momentum | `momentum_mean20`, `momentum_signflip20`, `ema50_dist` |
| Sentiment | `sentiment_mean20`, `sentiment_mania_pct`, `sentiment_breadth_cross` |
| Market Cycle | `cycle_mean20`, `cycle_skew`, `cycle_flow_cross` |
| Money Flow | `mflow_mean50`, `mflow_slope20`, `flow_composite` |
| Foreign Flow | `fflow_mean50`, `fflow_neg_pct` |
| Prop Flow | `pflow_mean50`, `pflow_consec_neg` |
| Dispersion | `dispersion_mean20`, `dispersion_spike_pct`, `dispersion_vol_ratio` |

#### 12C — Winsorization
Each feature is clipped at the **1st and 99th percentile** of its full-sample distribution
to prevent extreme values from distorting W1 distances.

#### 12D — W1 Scoring (`compute_wasserstein_scores`)
For each feature `f`, using `scipy.stats.wasserstein_distance`:
```
score_f = w0 × W1(prior_f, class0_f) + w1 × W1(prior_f, class1_f)
```
- `w0`, `w1` = proportion of Bear/Bull days (class weights).
- `prior_f` = full distribution of `f`.
- `class0_f` / `class1_f` = distribution of `f` on Bear / Bull days.
- A **high W1 score** means the feature's distribution shifts significantly between regimes → high discriminative power.
- Score is also **normalized by feature std** for cross-feature comparison.
- Features with fewer than 10 samples in either class, or near-zero variance, are skipped.

#### 12E — Redundancy Deduplication (`deduplicate_by_correlation`)
- Rank features by W1 score (descending).
- **Greedy selection**: keep the top feature; drop any subsequent feature whose
  **|Pearson r| > 0.85** with any already-selected feature.
- Prevents including highly correlated redundant signals.

#### 12F — Validation (`validate_with_logistic`)
- Fits a `LogisticRegression` (class_weight='balanced', max_iter=1000) on the selected features vs. Bull labels (in-sample, full dataset).
- Reports: **Accuracy**, **ROC-AUC**, and **Bull Recall** as a sanity check.
- AUC < 0.60 triggers a warning that features or labels may be too noisy.

#### Outputs of Step 12
- **Console table**: ranked feature list with W1 score, normalized W1, and Kept/Dropped status.
- **Per-pillar narrative**: which pillar has the highest max-W1 among its sub-signals.
- **`feature_selection_w1.csv`**: full scored table.
- **`feature_selection_w1_chart.png`**: horizontal bar chart (green = Kept, grey = Dropped).
- **`feature_distribution_w1.png`**: distribution comparison plots for top 10 features (prior vs. conditional on Bull).

---

### Steps 13–16 — V3 Signal Construction

#### Step 13 — Select Top 5 Features
From the W1 ranking, the **Top 5 selected (non-redundant)** features are extracted:
```python
TOP_K_FEATS = w1_selected[:5]
```

#### Step 14 — PCA Weighting on V3 Features
- Each of the Top 5 features is Z-scored again (`compute_z_score`) → `{feat}_v3z` columns.
- Rolling PCA (252-day window) on these 5 Z-scores → `composite_health_v3`.

#### Step 15 — Final Hysteresis on V3 Composite
- Same vol-adaptive, consensus-scaled hysteresis as Step 8, applied to `composite_health_v3`.
- Output: `health_signal_final` (and synced to `health_signal_v2` for chart).

#### Step 16 — Final V3 Output
- **`market_health_v3_final.csv`**: all columns including V3 features, composite, and final signal.
- **Chart** (`market_health_v2_chart.png` overwritten): VNINDEX with V3 Bull/Bear shading (from 2022 onwards).
- **Console summary**: selected pillar names, total signal flips, current regime (BULLISH/BEARISH), last 10 days of V3 features and composite.

---

## Key Design Decisions

| Decision | Rationale |
| :--- | :--- |
| **VNINDEX-only labels (Step 12)** | Avoids noise from less liquid indices (UPCOM, HNX). Single-index labeling is cleaner and more interpretable. |
| **Whole-sample W1 scoring** | Maximises the number of observations available for statistical distance estimation, giving more robust feature rankings. |
| **λ = 5 for L1 filter** | More sensitive to short/medium-term trend shifts. Higher λ would over-smooth and miss genuine regime changes. |
| **No train/test split** | W1 is a descriptive statistic (distributional distance), not a predictive model. There is no leakage concern for ranking purposes. The logistic check is an in-sample sanity check only. |
| **Min-Hold = 5 days** | Prevents whipsawing on transient noise spikes while remaining responsive to genuine regime shifts. |
| **VN30 for Breadth/Momentum** | Blue-chip signals lead the broader market and are less noisy than small-cap breadth. |
| **Broad market for Foreign/Prop flow** | Institutional activity spans the whole market; restricting to VN30 would miss significant off-index flows. |

---

## Data Schema & Sources

| Source | Table | Key Columns |
| :--- | :--- | :--- |
| Universe membership | `ai_ranking_universe` | `symbol`, `timestamp`, `universe` |
| Mid/Small membership | `raw_historical_list` | `is_vnmid`, `is_vnsml` |
| Daily prices | `raw_eod` | `timestamp`, `symbol`, `close`, `high`, `low`, `volume` |
| Bond yields | `raw_bond_yield` | `symbol='VN10Y'`, `close` (yield in %) |
| Money flow | `raw_money_flow` | `symbol`, `net_money_flow`, `timestamp` |
| Foreign trading | `raw_foreign_trading_detail` | `symbol`, `netvalue`, `timestamp` |
| Proprietary trading | `raw_proprietary_trading_detail` | `symbol`, `netvalue`, `timestamp` |

---

## Output Files

| File | Description |
| :--- | :--- |
| `market_health_v2_final.csv` | V2 signal with all 8 components, Z-scores, PCA weights, GARCH vol |
| `market_health_v3_final.csv` | V3 final signal with W1-selected features and composite |
| `market_health_v2_chart.png` | VNINDEX chart with V3 Bull/Bear shading + signal markers |
| `feature_selection_w1.csv` | W1 scores, normalized scores, Kept/Dropped status for all ~22 features |
| `feature_selection_w1_chart.png` | Bar chart of W1 scores (green = Kept, grey = Dropped) |
| `feature_distribution_w1.png` | Prior vs. conditional (Bull) distributions for the top 10 features |

---

*Updated: 2026-02-25 — V3.2: Whole-sample W1 scoring with VNINDEX-only ground truth labeling. Removed train/test split.*
*Previous: 2026-02-25 — V3.2: VNINDEX-Anchored Trend Labeling.*
*Previous: 2026-02-24 — V3.1 — L1 Trend-Optimized Wasserstein Analysis.*
*Previous: 2026-02-24 — V3.0 — W1-DRIVEN Health Index.*

## Task: Market Health Index Pipeline (W1-Driven)

### Objective
Calculate `market_health_index` for the Vietnamese stock market via a 4-step pipeline.
Each step outputs CSV + Graph into its own subfolder under `output/stepN_<name>/`.

### Reference Files (mandatory)
- `upgrade.py` — master template for the entire pipeline: feature computation, W1 selection,
  noise filtering, PCA weighting, hysteresis signal
- `synth_trend.py` — bull/bear market detection via L1 Trend Filtering (cvxpy)

### Critical Substitution
The `define_bull_base_labels()` in `upgrade.py` uses Z-score conditions to label bull days.
**REPLACE THIS** with `define_bull_base_labels()` from `synth_trend.py`, which uses
L1 Trend Filtering (DAILY_LAMBDA=5) on VNINDEX daily close to produce
`market_trend_regime = 1` (bull) / `0` (bear).

### Output Folder Structure
```
output/
  step1_features/
  step2_w1_scores/
  step3_component_selection/
  step4_market_health_index/
```

### Pipeline Summary
- Step 1: Compute raw sub-features from 8 components
- Step 2: Compute W1 scores per feature using synth_trend bull labels
- Step 3: Select top 5 components → top 2 features per component → 10 final features
- Step 4: Compute market_health_index from the 10 features (PCA weights + vol hysteresis)
