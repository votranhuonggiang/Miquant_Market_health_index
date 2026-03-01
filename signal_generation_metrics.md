# Signal Generation Metrics: Technical Synthesis Framework for Systematic Trading

Below is a structured framework used by quantitative researchers to operationalize trading objectives into measurable, testable technical features. This framework maps factors to multiple metrics with clear interpretation and explicit measurement definitions suitable for coding and backtesting.

---

### Technical Synthesis Framework for Systematic Trading

| **Factor**                       | **Metrics**                        | **Meaning**                                       | **Measure (Quant Implementation)**                                              |
| -------------------------------- | ---------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| **1. Strength of Current Trend** | `f1_ma_slope` (MA Slope)          | Measures directional persistence                  | β from rolling OLS of log price on time over last 20 periods                   |
|                                  | `f1_r2_trend` (R² Trend)          | Measures goodness-of-fit of trend                 | R² from rolling OLS of log price on time over last 20 periods                  |
|                                  | `f1_ma_alignment` (Alignment)     | Confirms multi-horizon agreement                  | Boolean: EMA20 > EMA50 > EMA200                                                |
|                                  | `f1_adx` (ADX)                    | Measures trend strength independent of direction  | ADX(14) calculated on market-wide synthetic OHLC (VN100 mean)                   |
|                                  | `f1_dist_long_ma` (Long MA Dist)  | Quantifies extension relative to structural trend | (Price − EMA200) / EMA200                                                      |
|                                  | `f1_hh_hl` (HH/HL Structure)      | Confirms structural trend                         | Boolean: Price > 20d Max AND 5d Min > 20d Min                                  |
|                                  | `f1_vol_confirm` (Vol. Confirm)   | Confirms participation                            | Total market volume / 20d Moving Average of volume                             |
|                                  | `f1_macd_hist` (MACD Hist)        | Measures momentum strength                        | MACD(12,26) − Signal(9)                                                        |
|                                  | `f1_rel_strength` (Rel. Strength) | Measures leadership vs benchmark                  | 20d return of Asset − 20d return of VNINDEX                                    |
|                                  | `f1_vol_adj_ret` (Sharpe Proxy)   | Quality of trend                                  | 20d mean return / 20d return standard deviation                                |

---

| **Factor**                       | **Metrics**             | **Meaning**                             | **Measure (Quant Implementation)**                |
| -------------------------------- | ----------------------- | --------------------------------------- | ------------------------------------------------- |
| **2. Maturity / Stage of Trend** | `f2_rsi` (RSI Level)              | Identifies overbought/oversold maturity | RSI(14) of the VN100 EW Index                                     |
|                                  | `f2_dist_short_ma` (Short MA Dist)| Measures short-term overextension       | (Price − EMA20) / EMA20                                           |
|                                  | `f2_atr_expansion` (ATR Exp.)     | Late-stage volatility expansion         | ATR(14) / ATR(50)                                                 |
|                                  | `f2_mom_divergence` (Divergence)  | Detects weakening internal momentum     | Boolean: Price HH(20) while RSI < rolling 20d RSI max             |
|                                  | `f2_bb_position` (BB Pos)         | Measures expansion phase                | (Price − LowerBB) / (UpperBB − LowerBB) [20d, 2σ]                 |
|                                  | `f2_sar_dist` (SAR Dist)          | Late acceleration stage                 | (Price − 10d Min) / 10d std deviation                             |
|                                  | `f2_time_since_breakout` (Age)    | Aging of trend                          | Index of max price in rolling 50-day window                       |
|                                  | `f2_hurst_proxy` (Hurst Proxy)    | Trend persistence regime                | Lag-1 autocorrelation of 100-day returns                          |
|                                  | `f2_rolling_skew` (Skewness)      | Euphoria/Exhaustion signal              | Rolling 63-day skewness of returns                                |
|                                  | `f2_vol_climax` (Vol. Climax)     | Distribution phase signal               | Boolean: Volume > 50d Mean + 2σ                                   |

---

| **Factor**                                  | **Metrics**                       | **Meaning**                          | **Measure (Quant Implementation)**           |
| ------------------------------------------- | --------------------------------- | ------------------------------------ | -------------------------------------------- |
| **3. Reward-to-Risk Ratio**      | `f3_atr_stop_dist` (ATR Stop)     | Defines volatility-adjusted downside | ATR(14) / Price                                         |
|                                  | `f3_swing_low_dist` (Swing Low)   | Structural invalidation level        | (Price − 20d Min) / Price                               |
|                                  | `f3_expected_move_ratio` (E/M)    | Asymmetric payoff estimate           | (20d Max − Price) / (Price − 20d Min)                   |
|                                  | `f3_risk_adj_trend` (Risk-Adj)    | Expected return relative to vol      | `f1_ma_slope` / 20d return standard deviation           |
|                                  | `f3_hist_vol_rank` (Vol Rank)     | Market pricing of risk               | 252d percentile rank of 20d realized volatility         |
|                                  | `f3_mae` (MAE)                    | Historical drawdown profile          | (Price − 20d Max) / 20d Max                             |
|                                  | `f3_downside_vol` (Downside Vol)  | Asymmetric risk measure              | 20d standard deviation of negative returns              |
|                                  | `f3_breakout_proj` (Proj. Move)   | Measured move target                 | (20d Max − 20d Min) / Price                             |
|                                  | `f3_kelly_proxy` (Kelly Proxy)    | Optimal capital sizing               | 63d mean return / (63d return variance + 1e-6)          |
|                                  | `f3_var_5pct` (VaR)               | Statistical risk bound               | 5% quantile of rolling 63-day returns                   |

---

| **Factor**                                          | **Metrics**                    | **Meaning**                 | **Measure (Quant Implementation)**          |
| --------------------------------------------------- | ------------------------------ | --------------------------- | ------------------------------------------- |
| **4. Potential Entry Levels**    | `f4_breakout_level` (Breakout)    | Structural confirmation     | Boolean: Price >= rolling 50-day high           |
|                                  | `f4_pullback_ma` (MA Pullback)    | Mean-reversion within trend | 1 / (1 + abs(Price − EMA50) / EMA50)            |
|                                  | `f4_fib_retracement` (Fib 61.8)   | Support zone within trend   | (Price − 20d Min) / (20d Max − 20d Min)         |
|                                  | `f4_vol_confirmation` (Vol Conf)  | Institutional participation | Boolean: Vol > 1.5× 20d avg AND Price > Prev.   |
|                                  | `f4_vol_contraction` (Narrow)     | Energy build-up             | 1 / clip(ATR14/ATR50, min=0.1)                  |
|                                  | `f4_vwap_dist` (VWAP Dist)        | Fair value entry            | (Price − Monthly VWAP Proxy) / VWAP Proxy       |
|                                  | `f4_confluence` (MA Confluence)   | Multi-signal alignment      | Mean proximity score to EMA20, EMA50, EMA200    |
|                                  | `f4_rsi_reset` (RSI Reset)        | Momentum cooldown           | Boolean: RSI(14) returns from >70 to [50, 60]   |
|                                  | `f4_donchian_break` (Donchian)    | Systematic entry trigger    | Boolean: Price > Prev 20d Max                   |
|                                  | `f4_market_regime` (Regime)       | Macro alignment             | Boolean: VNINDEX > EMA200 of VNINDEX            |

---

### How a Quant Would Combine These

In practice, metrics are often transformed into normalized features:

* **Z-score** each metric over a rolling window.
* **Combine** via a weighted scoring model.
* **Machine Learning**: Use a classifier to estimate trend continuation probability.
* **Backtest** conditional forward returns given specific thresholds.

**Example Model Construction:**

**Trend Strength Score**
= 0.3 × ADX_norm
+ 0.3 × MA_slope_norm
+ 0.2 × RS_norm
+ 0.2 × R²_norm

**Entry Probability Model**
P(positive 20-day return | breakout + ADX > 25 + RSI < 70)
