import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from scipy.stats import linregress, skew, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

# ============================================================
# CONFIG & HELPERS (Borrowed from pipeline)
# ============================================================
LOOKBACK_PERIOD = 252

def query_questdb(sql_query):
    host = os.environ.get("QUEST_DB_URL", "http://localhost:9000")
    auth = (os.getenv("QUESTDB_USERNAME"), os.getenv("QUESTDB_PASSWORD"))
    try:
        response = requests.get(
            host + "/exec", params={"query": sql_query}, auth=auth, timeout=60
        ).json()
        if "dataset" not in response or "columns" not in response:
            print(f"Error or no data: {response}")
            return pd.DataFrame()
        df = pd.DataFrame(
            response["dataset"],
            columns=pd.DataFrame(response["columns"])["name"].values,
        )
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def load_vn100_universe():
    return query_questdb("""
        SELECT symbol, timestamp
        FROM ai_ranking_universe
        WHERE universe = 'VN100'
          AND timestamp >= '2016-01-01' AND timestamp <= '2025-12-31'
    """)

def load_universe_prices(universe_df):
    if universe_df.empty: return pd.DataFrame()
    symbol_list = universe_df["symbol"].unique().tolist()
    in_clause = "('" + "','".join(symbol_list) + "')"
    return query_questdb(f"""
        SELECT timestamp, symbol, close, high, low, volume
        FROM raw_eod
        WHERE symbol IN {in_clause}
          AND timestamp >= '2016-01-01'
    """)

def load_vnindex():
    return query_questdb(
        "SELECT * FROM raw_eod WHERE symbol = 'VNINDEX' AND timestamp >= '2016-01-01'"
    )

def build_membership_map(universe_df, end_date):
    df = universe_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["period"] = df["timestamp"].dt.to_period("M")
    base_map = {p: set(g["symbol"].astype(str)) for p, g in df.groupby("period")}
    if not base_map: return {}
    periods_sorted = sorted(base_map.keys())
    start_p, last_p = periods_sorted[0], periods_sorted[-1]
    filled = {}
    last_set = None
    for p in pd.period_range(start=start_p, end=pd.Period(pd.to_datetime(end_date), freq="M"), freq="M"):
        if p in base_map: last_set = base_map[p]
        if last_set is not None: filled[p] = last_set
    return filled

def aggregate_by_membership(panel, membership_map):
    if panel.empty or not membership_map: return pd.Series(dtype=float)
    panel = panel.copy()
    panel.index = pd.to_datetime(panel.index)
    out_parts = []
    for month, month_block in panel.groupby(panel.index.to_period("M")):
        members = membership_map.get(month, None)
        if not members: continue
        cols = month_block.columns.intersection(pd.Index(sorted(members)))
        if len(cols) > 0: out_parts.append(month_block[cols].mean(axis=1))
    return pd.concat(out_parts).sort_index() if out_parts else pd.Series(dtype=float)

def build_ew_index(vn_df, membership_map=None):
    prices = vn_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    daily_returns = prices.pct_change().dropna(how="all")
    if membership_map: ew_returns = aggregate_by_membership(daily_returns, membership_map)
    else: ew_returns = daily_returns.mean(axis=1)
    ew_level = (1 + ew_returns.dropna()).cumprod() * 100.0
    return pd.DataFrame({"close": ew_level}, index=ew_level.index)

# ============================================================
# TECHNICAL INDICATOR HELPERS
# ============================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()
    up_move, down_move = high.diff(), -(low.diff())
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    pos_di = 100 * (pd.Series(pos_dm, index=df.index).rolling(window=period).mean() / atr_val.replace(0, np.nan))
    neg_di = 100 * (pd.Series(neg_dm, index=df.index).rolling(window=period).mean() / atr_val.replace(0, np.nan))
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di).replace(0, np.nan)
    return dx.rolling(window=period).mean()

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line

def rolling_linreg(series, window=20):
    slopes = pd.Series(index=series.index, dtype=float)
    r2s = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series) + 1):
        y = series.iloc[i-window:i]
        x = np.arange(window)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        slopes.iloc[i-1] = slope
        r2s.iloc[i-1] = r_value**2
    return slopes, r2s

# ============================================================
# FEATURE ENGINEERING: 4 FACTORS / 40 METRICS
# ============================================================
def extract_signal_generation_features(ew_df, vnidx_df, univ_prices_df):
    """
    Implements 40 metrics from signal_generation_metrics.md
    Using the VN100 EW Index (ew_df) as the primary proxy.
    """
    # ── Normalise ew_df index to tz-naive datetime ──────────────────────────
    # build_ew_index uses pivot(index='timestamp') on raw QuestDB rows, so the
    # index can be raw strings ("2016-01-04T00:00:00.000000Z"). Convert once
    # here so every .reindex(ew_df.index) below finds the right dates.
    ew_df = ew_df.copy()
    _ew_idx = pd.to_datetime(ew_df.index)
    ew_df.index = (_ew_idx.tz_convert(None) if _ew_idx.tz is not None else _ew_idx).normalize()

    feat = pd.DataFrame(index=ew_df.index)
    p = ew_df['close']
    log_p = np.log(p)
    
    # Pre-calculate common indicators
    ema20 = p.ewm(span=20, adjust=False).mean()
    ema50 = p.ewm(span=50, adjust=False).mean()
    ema200 = p.ewm(span=200, adjust=False).mean()
    
    # Build synthetic market OHLC from VN100 universe prices.
    # VNINDEX in raw_eod has no real high/low (index prices, not traded) -> ADX/ATR = NaN.
    # Use mean daily high/low across VN100 stocks as the market-wide OHLC proxy.
    _ohlc = univ_prices_df[['timestamp', 'high', 'low', 'close']].copy()
    _ohlc_ts = pd.to_datetime(_ohlc['timestamp'])
    _ohlc['timestamp'] = (_ohlc_ts.dt.tz_convert(None) if _ohlc_ts.dt.tz is not None else _ohlc_ts).dt.normalize()
    vn_idx = pd.DataFrame({
        'high':  _ohlc.groupby('timestamp')['high'].mean(),
        'low':   _ohlc.groupby('timestamp')['low'].mean(),
        'close': _ohlc.groupby('timestamp')['close'].mean(),
    })
    vn_idx.index = pd.to_datetime(vn_idx.index)
    vn_idx = vn_idx.reindex(ew_df.index).ffill()
    vn_idx['close'] = p.values  # override with EW index close for consistency
    # Keep VNINDEX close separately as benchmark for f1_rel_strength
    _vn = vnidx_df.set_index('timestamp').copy()
    _vn_ts = pd.to_datetime(_vn.index)
    _vn.index = _vn_ts.tz_convert(None) if _vn_ts.tz is not None else _vn_ts
    vn_close_bench = _vn['close'].astype(float).reindex(ew_df.index).ffill()
    
    # ------------------------------------------------------------
    # 1. Strength of Current Trend
    # ------------------------------------------------------------
    feat['f1_ma_slope'], feat['f1_r2_trend'] = rolling_linreg(log_p, 20)
    feat['f1_ma_alignment'] = ((ema20 > ema50) & (ema50 > ema200)).astype(float)
    feat['f1_adx'] = calculate_adx(vn_idx, 14)  # Uses real market-wide high/low from VN100
    feat['f1_dist_long_ma'] = (p - ema200) / ema200
    feat['f1_hh_hl'] = ((p > p.rolling(20).max().shift(1)) & (p.rolling(5).min() > p.rolling(20).min().shift(1))).astype(float)
    
    # Volume Trend Confirmation: Total market volume
    # Parse → strip timezone → normalize to date only, so index aligns with ew_df.index
    _vol_df = univ_prices_df[['timestamp', 'volume']].copy()
    _ts = pd.to_datetime(_vol_df['timestamp'])
    _vol_df['timestamp'] = (_ts.dt.tz_convert(None) if _ts.dt.tz is not None else _ts).dt.normalize()
    total_vol = _vol_df.groupby('timestamp')['volume'].sum()
    total_vol.index = pd.to_datetime(total_vol.index)
    total_vol = total_vol.reindex(ew_df.index).ffill().fillna(0)
    feat['f1_vol_confirm'] = total_vol / total_vol.rolling(20).mean().replace(0, np.nan)
    
    feat['f1_macd_hist'] = calculate_macd(p)
    
    # Relative Strength vs VNINDEX (use dedicated VNINDEX close benchmark)
    ret_ew = p.pct_change(20)
    ret_vn = vn_close_bench.pct_change(20)
    feat['f1_rel_strength'] = ret_ew - ret_vn
    
    feat['f1_vol_adj_ret'] = p.pct_change().rolling(20).mean() / p.pct_change().rolling(20).std().replace(0, np.nan)

    # ------------------------------------------------------------
    # 2. Maturity / Stage of Trend
    # ------------------------------------------------------------
    feat['f2_rsi'] = calculate_rsi(p, 14)
    feat['f2_dist_short_ma'] = (p - ema20) / ema20
    
    atr14 = calculate_atr(vn_idx, 14)
    atr50 = calculate_atr(vn_idx, 50)
    feat['f2_atr_expansion'] = atr14 / atr50.replace(0, np.nan)
    
    # Momentum Divergence: Price HH vs RSI LH (simplified score)
    rsi14 = feat['f2_rsi']
    feat['f2_mom_divergence'] = ((p > p.rolling(20).max().shift(1)) & (rsi14 < rsi14.rolling(20).max().shift(1))).astype(float)
    
    rolling_std = p.rolling(20).std()
    upper_bb = ema20 + 2 * rolling_std
    lower_bb = ema20 - 2 * rolling_std
    feat['f2_bb_position'] = (p - lower_bb) / (upper_bb - lower_bb).replace(0, np.nan)
    
    feat['f2_sar_dist'] = (p - p.rolling(10).min()) / p.rolling(10).std().replace(0, np.nan) # SAR-like distance
    feat['f2_time_since_breakout'] = (p.rolling(50).apply(lambda x: np.argmax(x == np.max(x)))) # Days since 50d high (approx)
    
    # Hurst Exponent (simplified via autocorrelation lag-1)
    feat['f2_hurst_proxy'] = p.pct_change().rolling(100).apply(lambda x: x.autocorr(lag=1))
    
    feat['f2_rolling_skew'] = p.pct_change().rolling(63).skew()
    feat['f2_vol_climax'] = (total_vol > total_vol.rolling(50).mean() + 2 * total_vol.rolling(50).std()).astype(float)

    # ------------------------------------------------------------
    # 3. Reward-to-Risk Ratio
    # ------------------------------------------------------------
    feat['f3_atr_stop_dist'] = atr14 / p.replace(0, np.nan)
    feat['f3_swing_low_dist'] = (p - p.rolling(20).min()) / p.replace(0, np.nan)
    feat['f3_expected_move_ratio'] = (p.rolling(20).max() - p) / (p - p.rolling(20).min()).replace(0, np.nan)
    feat['f3_risk_adj_trend'] = feat['f1_ma_slope'] / p.pct_change().rolling(20).std().replace(0, np.nan)
    
    # Implied Vol Proxy: Percentile of Historical Vol
    hist_vol = p.pct_change().rolling(20).std()
    feat['f3_hist_vol_rank'] = hist_vol.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if not x.empty else np.nan)
    
    feat['f3_mae'] = (p - p.rolling(20).max()) / p.rolling(20).max().replace(0, np.nan)
    
    neg_rets = p.pct_change().where(p.pct_change() < 0, 0)
    feat['f3_downside_vol'] = neg_rets.rolling(20).std()
    
    feat['f3_breakout_proj'] = (p.rolling(20).max() - p.rolling(20).min()) / p.replace(0, np.nan)
    
    # Kelly Fraction Proxy: (Mean Return / Variance) - simplified
    feat['f3_kelly_proxy'] = p.pct_change().rolling(63).mean() / (p.pct_change().rolling(63).var() + 1e-6)
    
    feat['f3_var_5pct'] = p.pct_change().rolling(63).quantile(0.05)

    # ------------------------------------------------------------
    # 4. Potential Entry Levels
    # ------------------------------------------------------------
    feat['f4_breakout_level'] = (p >= p.rolling(50).max()).astype(float)
    feat['f4_pullback_ma'] = 1 / (1 + (p - ema50).abs() / ema50) # Closer to 1 means closer to MA50
    feat['f4_fib_retracement'] = (p - p.rolling(20).min()) / (p.rolling(20).max() - p.rolling(20).min()).replace(0, np.nan)
    feat['f4_vol_confirmation'] = ((total_vol > 1.5 * total_vol.rolling(20).mean()) & (p > p.shift(1))).astype(float)
    feat['f4_vol_contraction'] = 1 / (atr14 / atr50.replace(0, np.nan)).clip(lower=0.1)
    
    # VWAP Support (Anchored to current month)
    # Simple proxy: Cumulative Price * Volume / Cumulative Volume within the month
    vwap_proxy = (vn_idx['close'] * total_vol).cumsum() / total_vol.cumsum()
    feat['f4_vwap_dist'] = (p - vwap_proxy) / vwap_proxy.replace(0, np.nan)
    
    # Confluence Score: proximity to multiple MAs
    feat['f4_confluence'] = (1 / (1 + (p - ema20).abs()/p) + 1 / (1 + (p - ema50).abs()/p) + 1 / (1 + (p - ema200).abs()/p)) / 3.0
    
    feat['f4_rsi_reset'] = ((rsi14 > 50) & (rsi14 < 60)).astype(float)
    feat['f4_donchian_break'] = (p > p.rolling(20).max().shift(1)).astype(float)
    feat['f4_market_regime'] = (vn_close_bench > vn_close_bench.ewm(span=200, adjust=False).mean()).astype(float)

    return feat

def winsorize_features(feat_df, lower=1, upper=99):
    out = feat_df.copy()
    for col in out.columns:
        valid = out[col].dropna()
        if valid.empty: continue
        lo = np.nanpercentile(valid, lower); hi = np.nanpercentile(valid, upper)
        out[col] = out[col].clip(lo, hi)
    return out

def compute_z_score(df, window=252):
    mu = df.rolling(window=window, min_periods=20).mean()
    sigma = df.rolling(window=window, min_periods=20).std()
    return ((df - mu) / sigma.replace(0, np.nan)).clip(-3, 3)

# ============================================================
# MAIN STEP 1 REPLACEMENT
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    base_out = os.path.dirname(__file__)
    output_dir = os.path.join(base_out, "step1_features")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Fetching data for new 40-metric framework...")
    univ100 = load_vn100_universe()
    df100 = load_universe_prices(univ100)
    vnidx = load_vnindex()
    
    if df100.empty or vnidx.empty:
        print("Error: Could not fetch data.")
        exit()
        
    max_dt = vnidx["timestamp"].max()
    map100 = build_membership_map(univ100, max_dt)
    ew100 = build_ew_index(df100, map100)
    
    print("Computing 40 Signal Generation Metrics...")
    feat_df = extract_signal_generation_features(ew100, vnidx, df100)
    
    # Winsorize and Z-score (Normalization as suggested in 'How a Quant Would Combine These')
    feat_df = winsorize_features(feat_df)
    feat_df = compute_z_score(feat_df)
    feat_df = feat_df.ffill().fillna(0).dropna(how='all')
    
    # Save the output to step1_features/step1_all_features.csv
    output_file = os.path.join(output_dir, "step1_all_features.csv")
    feat_df.to_csv(output_file)
    print(f"Saved {len(feat_df.columns)} features to {output_file}")
    
    # Visualization of all features (like in pipeline.py)
    cols = feat_df.columns
    n = len(cols)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(20, 5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        if i < len(axes):
            axes[i].plot(feat_df.index, feat_df[col])
            axes[i].set_title(col, fontsize=10)
            axes[i].grid(True, alpha=0.3)
    for j in range(n, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step1_features_overview.png"))
    plt.close()
    
    print("Step 1 Complete: 10 pillars replaced by 40 signal generation metrics.")
