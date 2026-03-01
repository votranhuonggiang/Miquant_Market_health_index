# Signal Generation Metrics: Technical Synthesis Framework for Systematic Trading

Below is a structured framework used by quantitative researchers to operationalize trading objectives into measurable, testable technical features. This framework maps factors to multiple metrics with clear interpretation and explicit measurement definitions suitable for coding and backtesting.

---

### Technical Synthesis Framework for Systematic Trading

| **Factor**                       | **Metrics**                        | **Meaning**                                       | **Measure (Quant Implementation)**                                              |
| -------------------------------- | ---------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------- |
| **1. Strength of Current Trend** | Moving Average Slope               | Measures directional persistence                  | β from OLS regression of log price on time over last N periods; or ΔMA(N)/MA(N) |
|                                  | Moving Average Alignment           | Confirms multi-horizon agreement                  | 20MA > 50MA > 200MA (bullish regime dummy = 1)                                  |
|                                  | ADX (Average Directional Index)    | Measures trend strength independent of direction  | ADX(14); strong trend if ADX > 25                                               |
|                                  | Price Distance from Long MA        | Quantifies extension relative to structural trend | (P − MA200) / MA200                                                             |
|                                  | R² of Trend Regression             | Measures goodness-of-fit of trend                 | R² from regression of log price over rolling window                             |
|                                  | Higher High / Higher Low Structure | Confirms structural trend                         | Rolling max/min comparison over last k swings                                   |
|                                  | Volume Trend Confirmation          | Confirms participation                            | OBV slope; Volume / Volume MA(20)                                               |
|                                  | MACD Histogram                     | Measures momentum strength                        | MACD(12,26) − Signal(9)                                                         |
|                                  | Relative Strength vs Benchmark     | Measures cross-sectional leadership               | RS = Asset return − Index return over N periods                                 |
|                                  | Volatility-Adjusted Return         | Quality of trend                                  | Sharpe-like: mean return / std dev over rolling window                          |

---

| **Factor**                       | **Metrics**             | **Meaning**                             | **Measure (Quant Implementation)**                |
| -------------------------------- | ----------------------- | --------------------------------------- | ------------------------------------------------- |
| **2. Maturity / Stage of Trend** | RSI Level               | Identifies overbought/oversold maturity | RSI(14); >70 late bull phase                      |
|                                  | Distance from Short MA  | Measures short-term overextension       | (P − MA20)/MA20                                   |
|                                  | ATR Expansion           | Late-stage volatility expansion         | ATR(14) / ATR(50)                                 |
|                                  | Momentum Divergence     | Detects weakening internal momentum     | Price makes HH while RSI makes LH                 |
|                                  | Bollinger Band Position | Measures expansion phase                | (P − LowerBB) / (UpperBB − LowerBB)               |
|                                  | Parabolic SAR Distance  | Late acceleration stage                 | Distance between price and SAR                    |
|                                  | Time Since Breakout     | Aging of trend                          | Bars since last 50-day high breakout              |
|                                  | Hurst Exponent          | Trend persistence regime                | H > 0.5 trending, declining H suggests exhaustion |
|                                  | Rolling Skewness        | Late euphoric stage often right-skewed  | Skewness of returns over N                        |
|                                  | Volume Climax Indicator | Distribution phase signal               | Volume spike > 2σ above mean                      |

---

| **Factor**                                  | **Metrics**                       | **Meaning**                          | **Measure (Quant Implementation)**           |
| ------------------------------------------- | --------------------------------- | ------------------------------------ | -------------------------------------------- |
| **3. Reward-to-Risk Ratio of New Position** | ATR-based Stop Distance           | Defines volatility-adjusted downside | Stop = Entry − k × ATR(14)                   |
|                                             | Recent Swing Low Distance         | Structural invalidation level        | Entry − recent support                       |
|                                             | Expected Move vs Stop             | Asymmetric payoff estimate           | Target distance / Stop distance              |
|                                             | Risk-Adjusted Trend Strength      | Expected return relative to vol      | Trend slope / realized volatility            |
|                                             | Implied Volatility (if available) | Market pricing of risk               | IV percentile rank                           |
|                                             | Maximum Adverse Excursion (MAE)   | Historical drawdown profile          | Average MAE of similar setups                |
|                                             | Downside Volatility               | Asymmetric risk measure              | Std dev of negative returns                  |
|                                             | Expected Breakout Projection      | Measured move target                 | Height of base projected upward              |
|                                             | Kelly Fraction Proxy              | Optimal capital sizing               | Edge / Variance estimate                     |
|                                             | Value-at-Risk (VaR)               | Statistical risk bound               | Quantile (5%) of rolling return distribution |

---

| **Factor**                                          | **Metrics**                    | **Meaning**                 | **Measure (Quant Implementation)**          |
| --------------------------------------------------- | ------------------------------ | --------------------------- | ------------------------------------------- |
| **4. Potential Entry Levels for New Long Position** | Breakout Level                 | Structural confirmation     | Close > rolling 50-day high                 |
|                                                     | Pullback to MA                 | Mean-reversion within trend | Price near MA20 or MA50                     |
|                                                     | Fibonacci Retracement          | Support zone within trend   | 38.2%–61.8% retracement level               |
|                                                     | Volume Confirmation            | Institutional participation | Breakout volume > 1.5× avg volume           |
|                                                     | Volatility Contraction Pattern | Energy build-up             | Declining ATR and narrowing Bollinger Bands |
|                                                     | VWAP Support                   | Fair value entry            | Price near anchored VWAP                    |
|                                                     | Support Confluence             | Multi-signal alignment      | Overlap of MA + horizontal support          |
|                                                     | RSI Reset                      | Momentum cooldown           | RSI returns from >70 to 50–60               |
|                                                     | Donchian Channel Break         | Systematic entry trigger    | Close > 20-day high                         |
|                                                     | Market Regime Filter           | Macro alignment             | Index above 200MA                           |

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
