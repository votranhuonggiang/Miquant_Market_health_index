import pandas as pd
import numpy as np
import os
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cvxpy as cp
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
from arch import arch_model
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# IMPORTS & CONFIG
# ============================================================
DAILY_LAMBDA = 5
LOOKBACK_PERIOD = 252
GARCH_BURN_IN = 252
GARCH_REFIT_FREQ = 63
PCA_WINDOW = 252
BASE_BUFFER = 0.5
MIN_HOLD = 5

pillar_map = {
    "breadth":    ["breadth_mean20", "breadth_stability", "breadth_pct_high"],
    "momentum":   ["momentum_mean20", "momentum_signflip20", "ema50_dist"],
    "sentiment":  ["sentiment_mean20", "sentiment_mania_pct", "sentiment_breadth_cross"],
    "cycle":      ["cycle_mean20", "cycle_skew", "cycle_flow_cross"],
    "money_flow": ["mflow_mean50", "mflow_slope20", "flow_composite"],
    "foreign":    ["fflow_mean50", "fflow_neg_pct"],
    "prop":       ["pflow_mean50", "pflow_consec_neg"],
    "dispersion": ["dispersion_mean20", "dispersion_spike_pct", "dispersion_vol_ratio"],
}

# ============================================================
# DATA LOADING
# ============================================================

# Helper function to execute SQL queries on QuestDB and return data as a pandas DataFrame
def query_questdb(sql_query, label="Data"):
    """Execute SQL query against QuestDB and return DataFrame with tracking info."""
    host = os.environ.get("QUEST_DB_URL", "http://localhost:9000")
    auth = (os.getenv("QUESTDB_USERNAME"), os.getenv("QUESTDB_PASSWORD"))
    try:
        response = requests.get(
            host + "/exec", params={"query": sql_query}, auth=auth
        ).json()
        if "dataset" not in response or "columns" not in response:
            print(f"  [DB] {label:25} | Error or no data: {response}")
            return pd.DataFrame()
        
        df = pd.DataFrame(
            response["dataset"],
            columns=pd.DataFrame(response["columns"])["name"].values,
        )
        
        if not df.empty:
            count = len(df)
            date_info = ""
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"])
                min_ts = ts.min()
                max_ts = ts.max()
                date_info = f" | From: {min_ts.date() if not pd.isnull(min_ts) else 'N/A'} To: {max_ts.date() if not pd.isnull(max_ts) else 'N/A'}"
            print(f"  [DB] {label:25} | Rows: {count:7}{date_info}")
        else:
            print(f"  [DB] {label:25} | Rows:       0 (Empty)")
            
        return df
    except Exception as e:
        print(f"  [DB] {label:25} | Exception: {e}")
        return pd.DataFrame()

def load_vn100_universe():
    return query_questdb("""
        SELECT symbol, timestamp
        FROM ai_ranking_universe
        WHERE universe = 'VN100'
          AND timestamp >= '2016-01-01' AND timestamp <= '2025-12-31'
    """, label="VN100 Universe Map")

def load_universe_prices(universe_df, label="VN100"):
    if universe_df.empty:
        print(f"  WARNING: No symbols in {label} universe")
        return pd.DataFrame()
    symbol_list = universe_df["symbol"].unique().tolist()
    in_clause = "('" + "','".join(symbol_list) + "')"
    return query_questdb(f"""
        SELECT timestamp, symbol, close, high, low, volume
        FROM raw_eod
        WHERE symbol IN {in_clause}
          AND timestamp >= '2016-01-01'
    """, label=f"{label} Prices")

def load_vnindex():
    return query_questdb(
        "SELECT * FROM raw_eod WHERE symbol = 'VNINDEX' AND timestamp >= '2016-01-01'",
        label="VNINDEX Close"
    )

def load_govbond():
    return query_questdb(
        "SELECT * FROM raw_bond_yield WHERE symbol = 'VN10Y' AND timestamp > '2016-01-01'",
        label="Gov Bond 10Y"
    )

def load_money_flow():
    return query_questdb(
        "SELECT symbol, net_money_flow, timestamp FROM raw_money_flow WHERE timestamp > '2016-01-01'",
        label="Retail Money Flow"
    )

def load_foreign_trading():
    return query_questdb("""
        SELECT timestamp, symbol, netvalue 
        FROM raw_foreign_trading_detail
        WHERE timestamp > '2016-01-01'
    """, label="Foreign Net Flow")

def load_proprietary_trading():
    return query_questdb("""
        SELECT timestamp, symbol, netvalue 
        FROM raw_proprietary_trading_detail
        WHERE timestamp > '2016-01-01'
    """, label="Proprietary Net Flow")

def load_vn30_universe():
    return query_questdb("""
        SELECT symbol, timestamp
        FROM ai_ranking_universe
        WHERE universe = 'VN30'
          AND timestamp >= '2016-01-01' AND timestamp <= '2025-12-31'
    """, label="VN30 Universe Map")

def load_mid_universe():
    return query_questdb("""
        SELECT DISTINCT symbol, timestamp 
        FROM raw_historical_list 
        WHERE is_vnmid = 1
    """, label="MID Universe Map")

def load_small_universe():
    return query_questdb("""
        SELECT DISTINCT symbol, timestamp 
        FROM raw_historical_list 
        WHERE is_vnsml = 1
    """, label="SMALL Universe Map")

# ============================================================
# MEMBERSHIP & INDEX BUILDERS
# ============================================================

# Map symbols to their monthly inclusion in a specified universe (e.g., VN100 list changes over time)
def build_membership_map(universe_df, end_date):
    df = universe_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["period"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)),
        freq="M"
    )
    # Group symbols by month
    base_map = {p: set(g["symbol"].astype(str)) for p, g in df.groupby("period")}
    if not base_map:
        return {}
    
    # Fill gaps in membership map between start and end dates
    periods_sorted = sorted(base_map.keys())
    start_p = periods_sorted[0]
    last_p = periods_sorted[-1]
    end_p = pd.Period(pd.to_datetime(end_date), freq="M")
    filled = {}
    last_set = None
    for p in pd.period_range(start=start_p, end=last_p, freq="M"):
        if p in base_map:
            last_set = base_map[p]
        if last_set is not None:
            filled[p] = last_set
    if last_set is None:
        last_set = base_map[last_p]
    if end_p > last_p:
        for p in pd.period_range(start=last_p + 1, end=end_p, freq="M"):
            filled[p] = last_set
    return filled

# Calculate the mean of a metric ONLY for stocks currently in the membership list for each month
def aggregate_by_membership(panel, membership_map):
    if panel.empty or not membership_map:
        return pd.Series(dtype=float)
    panel = panel.copy()
    panel.index = pd.to_datetime(panel.index)
    panel.sort_index(inplace=True)
    out_parts = []
    months = panel.index.to_period("M")
    # Iterate through each month and filter symbols based on the membership map
    for month, month_block in panel.groupby(months):
        members = membership_map.get(month, None)
        if not members:
            continue
        # Only take columns (symbols) that are in the universe for THIS month
        cols = month_block.columns.intersection(pd.Index(sorted(members)))
        if len(cols) == 0:
            continue
        sub = month_block[cols]
        # Calculate cross-sectional mean across members
        res = sub.mean(axis=1, skipna=True)
        out_parts.append(res)
    if not out_parts:
        return pd.Series(dtype=float)
    return pd.concat(out_parts).sort_index()

# Builds an Equal-Weighted (EW) index level from constituent prices
def build_ew_index(vn_df, membership_map=None):
    prices = vn_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    daily_returns = prices.pct_change().dropna(how="all")
    if membership_map:
        ew_returns = aggregate_by_membership(daily_returns, membership_map)
    else:
        ew_returns = daily_returns.mean(axis=1)
    ew_level = (1 + ew_returns.dropna()).cumprod() * 100.0
    out = pd.DataFrame({"close": ew_level}, index=ew_level.index)
    out.index.name = "timestamp"
    return out.reset_index()

# Builds a Capitalization-Weighted (CW) index level using (Price * Volume) as a rolling weight proxy
def build_cw_index(vn_df, membership_map=None):
    df = vn_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    prices = df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    volume = df.pivot(index="timestamp", columns="symbol", values="volume").fillna(0)
    # Weights are calculated as Price * Volume from the PREVIOUS day to avoid look-ahead bias
    weights = (prices * volume).shift(1)
    returns = prices.pct_change()
    cw_returns = pd.Series(index=returns.index, dtype=float)
    for date, row in returns.iterrows():
        w = weights.loc[date] if date in weights.index else None
        if w is None or w.sum() == 0: continue
        valid = ~(row.isna() | w.isna())
        if valid.any():
            # Weighted average return
            cw_returns.loc[date] = np.average(row[valid], weights=w[valid])
    cw_level = (1 + cw_returns.dropna()).cumprod() * 100.0
    out = pd.DataFrame({"close": cw_level}, index=cw_level.index)
    out.index.name = "timestamp"
    return out.reset_index()

# ============================================================
# SYNTH TREND — BULL LABEL
# ============================================================

# Detects underlying price trends (regimes) using L1 trend filtering
# Returns 1 for Bullish (above trend) and 0 for Bearish
def trend_filtering(data, lambda_value):
    try:
        n = np.size(data)
        x_ret = data.reshape(n)
        Dfull = np.diag([1] * n) - np.diag([1] * (n - 1), 1)
        D = Dfull[0:(n - 1),]
        beta = cp.Variable(n)
        lambd = cp.Parameter(nonneg=True)
        def tf_obj(x, b, lbd):
            return cp.norm(x - b, 2) ** 2 + lbd * cp.norm(cp.matmul(D, b), 1)
        problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))
        lambd.value = lambda_value
        try:
            problem.solve(solver='CLARABEL')
        except:
            problem.solve()
        return beta.value
    except Exception as e:
        print(f"Error in trend_filtering: {e}")
        return None

def regime_switch(betas, threshold=1e-5):
    n = len(betas)
    init_points = [0]
    curr_reg = (betas[0] > threshold)
    for i in range(n):
        if (betas[i] > threshold) != curr_reg:
            curr_reg = not curr_reg
            init_points.append(i)
    init_points.append(n)
    return init_points

def indicator_function(x):
    return np.where(x > 0, 1, 0)

def detect_regime(dataset, lambda_value=DAILY_LAMBDA):
    returns = dataset.pct_change().dropna() * 100.0
    if len(returns) < 50:
        return None
    betas = trend_filtering(returns.values, lambda_value)
    if betas is not None:
        curr_reg = indicator_function(betas - 1e-5)
        return pd.Series(curr_reg, index=returns.index, name="market_trend_regime")
    else:
        return None

def define_bull_base_labels(df_price, lambda_value=DAILY_LAMBDA):
    df = df_price.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return detect_regime(df["close"], lambda_value)

# ============================================================
# COMPONENT MEASUREMENT FUNCTIONS
# ============================================================

# --- Pillar: Market Breadth ---
# Measures what percentage of symbols are trading above their Moving Average (EMA 50)
def measure_market_breadth(vn100_df, membership_map=None, window=50):
    prices = vn100_df.pivot(index="timestamp", columns="symbol", values="close")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().ffill()
    flags = pd.concat([
        (prices[col] > prices[col].ewm(span=window, adjust=False).mean()).astype(int).rename(col)
        for col in prices.columns
    ], axis=1)
    if membership_map:
        breadth = aggregate_by_membership(flags, membership_map)
    else:
        breadth = flags.mean(axis=1)
    return pd.DataFrame({"breadth": breadth}, index=breadth.index)

# --- Pillar: Momentum ---
# Measures trend momentum using MACD Histogram of an Equal-Weighted index
def measure_momentum(ew_index_df):
    df = ew_index_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    price = df["close"].astype(float)
    ema_fast = price.ewm(span=12, adjust=False).mean()
    ema_slow = price.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return pd.DataFrame({"momentum": macd_hist}, index=price.index)

# --- Pillar: Sentiment ---
# Combines Bond Yields (Safe Haven vs Stock Returns) and High/Low Strength for sentiment
def measure_sentiment(vnindex_df, vn100_df, bond_df):
    idx = vnindex_df.copy()
    idx["timestamp"] = pd.to_datetime(idx["timestamp"])
    idx = idx.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    vn100 = vn100_df.copy()
    vn100["timestamp"] = pd.to_datetime(vn100["timestamp"])
    prices = vn100.pivot(index="timestamp", columns="symbol", values="close").sort_index().ffill()
    highs = vn100.pivot(index="timestamp", columns="symbol", values="high").sort_index().ffill()
    lows = vn100.pivot(index="timestamp", columns="symbol", values="low").sort_index().ffill()
    bond = bond_df.copy()
    bond["timestamp"] = pd.to_datetime(bond["timestamp"])
    bond = bond.sort_values("timestamp").drop_duplicates("timestamp", keep="last").set_index("timestamp")
    bond["yield_pct"] = pd.to_numeric(bond["close"], errors="coerce")
    common = idx.index.intersection(prices.index).intersection(bond.index)
    idx, prices, highs, lows, bond = idx.loc[common], prices.loc[common], highs.loc[common], lows.loc[common], bond.loc[common]
    sent_breadth = np.sign(prices.pct_change()).sum(axis=1)
    n_common = len(common)
    strength_w = min(252, max(n_common // 2, 20))
    strength_m = max(strength_w // 4, 10)
    max_h = highs.shift(1).rolling(strength_w, min_periods=strength_m).max()
    new_h = (highs > max_h).astype(int).sum(axis=1)
    min_h = lows.shift(1).rolling(strength_w, min_periods=strength_m).min()
    new_l = (lows < min_h).astype(int).sum(axis=1)
    strength = new_h - new_l
    stock_r = idx["close"].pct_change(20)
    bond_pv = 1000 / (1 + bond["yield_pct"] / 100) ** 10
    bond_r = bond_pv.pct_change(20)
    safe_h = stock_r - bond_r
    fg = pd.DataFrame({"breadth": sent_breadth, "safe_haven": safe_h, "strength": strength}).ffill().dropna()
    z_w = min(252, max(len(fg) // 3, 20))
    z_m = max(z_w // 4, 10)
    fg_z = (fg - fg.rolling(z_w, min_periods=z_m).mean()) / fg.rolling(z_w, min_periods=z_m).std()
    raw_s = fg_z.dropna().sum(axis=1)
    z_w2 = min(252, max(len(raw_s) // 3, 15))
    z_m2 = max(z_w2 // 4, 5)
    raw_s = (raw_s - raw_s.rolling(z_w2, min_periods=z_m2).mean()) / raw_s.rolling(z_w2, min_periods=z_m2).std()
    raw_s = raw_s.dropna()
    if raw_s.empty: return pd.DataFrame({"sentiment": 0.0}, index=fg.index)
    kf = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]], initial_state_mean=raw_s.iloc[0], initial_state_covariance=1, observation_covariance=1, transition_covariance=1e-3)
    smoothed, _ = kf.smooth(raw_s.values)
    return pd.DataFrame({"sentiment": smoothed.flatten()}, index=raw_s.index)

# --- Pillar: Market Cycle ---
# Uses the KST (Know Sure Thing) indicator to identify market "seasons" (Bull/Bear/Neutral transitions)
def measure_market_cycle(vn_df, membership_map=None):
    def _get_kst(close):
        def roc(s, n): return ((s - s.shift(n)) / s.shift(n)) * 100.0
        r1, r2, r3, r4 = roc(close, 10).rolling(10).mean(), roc(close, 15).rolling(10).mean(), roc(close, 20).rolling(10).mean(), roc(close, 30).rolling(15).mean()
        return 1.0 * r1 + 2.0 * r2 + 3.0 * r3 + 4.0 * r4
    def _season(kst):
        dk = kst.diff()
        s = pd.Series(0.0, index=kst.index)
        s[(kst >= 0) & (dk > 0)] = 1.0; s[(kst < 0) & (dk > 0)] = 0.5; s[(kst >= 0) & (dk < 0)] = -0.5; s[(kst < 0) & (dk < 0)] = -1.0
        return s
    prices = vn_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    seasons = pd.DataFrame(index=prices.index)
    for col in prices.columns: seasons[col] = _season(_get_kst(prices[col]))
    market_cycle = aggregate_by_membership(seasons, membership_map) if membership_map else seasons.mean(axis=1)
    return pd.DataFrame({"market_cycle": market_cycle}, index=market_cycle.index)

# --- Pillar: Money Flow ---
# Aggregates raw transactional money flow across the universe
def measure_money_flow(money_flow_df, membership_map=None, smoothing_window=50):
    df = money_flow_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    flow_wide = df.pivot_table(index="timestamp", columns="symbol", values="net_money_flow", aggfunc="sum").fillna(0)
    flow_smoothed = flow_wide.rolling(smoothing_window, min_periods=5).sum().dropna(how="all")
    if membership_map:
        out_parts = []
        months = flow_smoothed.index.to_period("M")
        for month, month_block in flow_smoothed.groupby(months):
            members = membership_map.get(month, None)
            if not members: continue
            cols = month_block.columns.intersection(pd.Index(sorted(members)))
            if len(cols) > 0: out_parts.append(month_block[cols].sum(axis=1))
        market_flow = pd.concat(out_parts).sort_index() if out_parts else flow_smoothed.sum(axis=1)
    else: market_flow = flow_smoothed.sum(axis=1)
    return pd.DataFrame({"money_flow": market_flow.dropna()}, index=market_flow.dropna().index)

def measure_institutional_flow(flow_df, membership_map=None, smoothing_window=50, label="foreign"):
    df = flow_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    flow_wide = df.pivot_table(index="timestamp", columns="symbol", values="netvalue", aggfunc="sum").fillna(0)
    flow_smoothed = flow_wide.rolling(smoothing_window, min_periods=5).sum()
    if membership_map:
        out_parts = []
        months = flow_smoothed.index.to_period("M")
        for month, month_block in flow_smoothed.groupby(months):
            members = membership_map.get(month, None)
            if not members: continue
            cols = month_block.columns.intersection(pd.Index(sorted(members)))
            if len(cols) > 0: out_parts.append(month_block[cols].sum(axis=1))
        market_flow = pd.concat(out_parts).sort_index() if out_parts else flow_smoothed.sum(axis=1)
    else: market_flow = flow_smoothed.sum(axis=1)
    return pd.DataFrame({f"{label}_flow": market_flow.dropna()}, index=market_flow.dropna().index)

# --- Pillar: Dispersion ---
# Measures internal correlation/volatility spread among universe constituents
def measure_dispersion_index(vn100_df, membership_map=None):
    close = vn100_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    volume = vn100_df.pivot(index="timestamp", columns="symbol", values="volume").fillna(0)
    close.index = pd.to_datetime(close.index); volume.index = pd.to_datetime(volume.index)
    returns = np.log(close / close.shift(1)).dropna(how="all")
    weights = (close * volume).shift(1)
    dispersion_series = pd.Series(index=returns.index, dtype=float)
    for i in range(len(returns)):
        date = returns.index[i]
        daily_r = returns.iloc[i]
        daily_w = weights.loc[date] if date in weights.index else None
        if daily_w is None: continue
        if membership_map:
            members = membership_map.get(date.to_period("M"), None)
            if members:
                avail = daily_r.index.intersection(pd.Index(sorted(members)))
                daily_r, daily_w = daily_r[avail], daily_w[avail]
        valid = ~(daily_r.isna() | daily_w.isna() | (daily_w == 0))
        r, w = daily_r[valid], daily_w[valid]
        if len(r) < 10: continue
        w_avg = np.average(r, weights=w)
        dispersion_series.iloc[i] = np.sqrt(np.average((r - w_avg)**2, weights=w))
    return pd.DataFrame({"dispersion": dispersion_series.dropna()}, index=dispersion_series.dropna().index)

# ============================================================
# NOISE REDUCTION & SIGNAL FUNCTIONS
# ============================================================

def compute_z_score(series, window=LOOKBACK_PERIOD):
    eff_w = min(window, max(len(series) // 3, 15))
    min_p = max(eff_w // 4, 5)
    mu, sigma = series.rolling(window=eff_w, min_periods=min_p).mean(), series.rolling(window=eff_w, min_periods=min_p).std()
    return ((series - mu) / sigma).clip(-3, 3)

# Computes weights for the composite index using rolling PCA (Principal Component Analysis)
# The first principal component (PC1) capturing the maximum shared variance among features is used for weighting
def compute_pca_weights(df, cols, window=PCA_WINDOW):
    n_rows = len(df)
    eff_w = min(window, max(n_rows // 2, 20))
    min_c = max(eff_w // 2, 10)
    weights_df = pd.DataFrame(index=df.index, columns=cols)
    if n_rows < min_c:
        return weights_df.fillna(1.0 / len(cols))
    for i in range(eff_w, n_rows):
        chunk = df[cols].iloc[i - eff_w : i].dropna()
        if len(chunk) < min_c: continue
        scaler = StandardScaler()
        scaled = scaler.fit_transform(chunk)
        # PCA(1) identifies the single largest 'theme' in the data
        pca = PCA(n_components=1)
        pca.fit(scaled)
        loadings = pca.components_[0]
        # Ensure weights are positive (loadings can flip sign arbitrarily)
        if np.sum(loadings) < 0: loadings = -loadings
        weights_df.iloc[i] = loadings
    return weights_df.ffill().fillna(1.0 / len(cols))

def compute_garch_rolling(ew_index_df, reestimate_freq=GARCH_REFIT_FREQ, burn_in=GARCH_BURN_IN):
    df = ew_index_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    rets = (np.log(df["close"].astype(float) / df["close"].astype(float).shift(1)).dropna()) * 100.0
    sigmas = pd.Series(index=df.index, dtype=float)
    eff_b = min(burn_in, max(len(rets) // 2, 30))
    eff_r = min(reestimate_freq, max(len(rets) // 4, 10))
    if len(rets) < 30:
        simple_vol = rets.rolling(min(20, len(rets)), min_periods=5).std() / 100.0
        sigmas = simple_vol * np.sqrt(252)
        return pd.DataFrame({"timestamp": df["timestamp"], "sigma_ann": sigmas}).set_index("timestamp")
    for i in range(eff_b, len(df), eff_r):
        end_idx = min(i + eff_r, len(df))
        train = rets.iloc[:i]
        try:
            am = arch_model(train, mean="ARX", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
            res = am.fit(disp="off", show_warning=False)
            am_next = arch_model(rets.iloc[:end_idx], mean="ARX", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
            fixed_res = am_next.fix(res.params)
            if i == eff_b: sigmas.iloc[1 : i + 1] = fixed_res.conditional_volatility.iloc[:i].values / 100.0
            sigmas.iloc[i + 1 : end_idx + 1] = fixed_res.conditional_volatility.iloc[i:end_idx].values / 100.0
        except: continue
    if sigmas.isna().all(): sigmas = rets.rolling(min(20, len(rets)), min_periods=5).std() / 100.0
    return pd.DataFrame({"timestamp": df["timestamp"], "sigma_ann": sigmas * np.sqrt(252)}).set_index("timestamp")

def calculate_signal_agreement(df, cols):
    return (np.sign(df[cols]).sum(axis=1) / len(cols)).abs()

# Converts raw health scores into binary 0/1 signals using Volatility-Weighted Hysteresis
# Hysteresis prevents 'flickering' signals by requiring a minimum score to turn ON and a negative score to turn OFF
def apply_vol_weighted_hysteresis(scores, vol_series, consensus_series=None, base_buffer=BASE_BUFFER, min_hold=MIN_HOLD):
    states = np.zeros(len(scores)); last_state = 0; days_held = 0
    # Increase the required gap (buffer) when market volatility is high
    vol_mult = (vol_series / vol_series.rolling(252).mean()).fillna(1.0)
    for i in range(len(scores)):
        score = scores.iloc[i] if hasattr(scores, "iloc") else scores[i]
        buff = base_buffer * vol_mult.iloc[i]
        # Reduce buffer if multiple sub-features agree on the direction
        if consensus_series is not None: buff *= (2.0 - consensus_series.iloc[i])
        
        new_state = last_state
        if score > buff: new_state = 1
        elif score < -buff: new_state = 0
        
        # Enforce a minimum holding period to filter out noise
        if new_state != last_state:
            if days_held < min_hold: new_state = last_state; days_held += 1
            else: days_held = 1
        else: days_held += 1
        last_state = new_state
        states[i] = last_state
    return states

def winsorize_features(feat_df, lower=1, upper=99):
    out = feat_df.copy()
    for col in out.columns:
        lo = np.nanpercentile(out[col].dropna(), lower); hi = np.nanpercentile(out[col].dropna(), upper)
        out[col] = out[col].clip(lo, hi)
    return out

# ============================================================
# W1 FEATURE SELECTION FUNCTIONS
# ============================================================

def _rolling_slope(series, window):
    slopes = pd.Series(index=series.index, dtype=float)
    x = np.arange(window, dtype=float); x -= x.mean(); xss = (x ** 2).sum()
    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1 : i + 1].values
        if not np.isnan(y).any(): slopes.iloc[i] = np.dot(x, y) / xss
    return slopes

def extract_candidate_features(df, ew_index_vn30=None):
    feat = pd.DataFrame(index=df.index)
    feat["breadth_mean20"] = df["breadth_z"].rolling(20, min_periods=5).mean()
    feat["breadth_stability"] = 1 - df["breadth_z"].rolling(20, min_periods=5).std()
    feat["breadth_pct_high"] = (df["breadth"] > 0.60).astype(float).rolling(20, min_periods=5).mean() if "breadth" in df.columns else 0
    feat["momentum_mean20"] = df["momentum_z"].rolling(20, min_periods=5).mean()
    feat["momentum_signflip20"] = (np.sign(df["momentum_z"]).diff() != 0).astype(float).rolling(20, min_periods=5).sum()
    feat["sentiment_mean20"] = df["sentiment_z"].rolling(20, min_periods=5).mean()
    feat["sentiment_mania_pct"] = (df["sentiment_z"] > 1.5).astype(float).rolling(20, min_periods=5).mean()
    feat["cycle_mean20"] = df["market_cycle_z"].rolling(20, min_periods=5).mean()
    feat["cycle_skew"] = df["market_cycle_z"] - df["market_cycle_z"].rolling(63, min_periods=15).mean()
    feat["mflow_mean50"] = df["money_flow_z"].rolling(50, min_periods=10).mean()
    feat["mflow_slope20"] = _rolling_slope(df["money_flow_z"], 20)
    feat["fflow_mean50"] = df["foreign_flow_z"].rolling(50, min_periods=10).mean()
    feat["fflow_neg_pct"] = (df["foreign_flow_z"] < 0).astype(float).rolling(50, min_periods=10).mean()
    feat["pflow_mean50"] = df["prop_flow_z"].rolling(50, min_periods=10).mean()
    feat["pflow_consec_neg"] = (df["prop_flow_z"] < 0).astype(int).rolling(20, min_periods=5).sum()
    feat["dispersion_mean20"] = df["dispersion_z"].rolling(20, min_periods=5).mean()
    feat["dispersion_spike_pct"] = (df["dispersion_z"] > 1.0).astype(float).rolling(20, min_periods=5).mean()
    if ew_index_vn30 is not None and not ew_index_vn30.empty:
        ew = ew_index_vn30.copy().set_index("timestamp")
        ew.index = pd.to_datetime(ew.index)
        ema50 = ew["close"].ewm(span=50, adjust=False).mean()
        feat["ema50_dist"] = ((ew["close"] - ema50) / ema50).reindex(df.index).ffill()
    else: feat["ema50_dist"] = df["momentum_z"]
    feat["flow_composite"] = (df["money_flow_z"] + df["foreign_flow_z"] + df["prop_flow_z"]) / 3.0
    feat["sentiment_breadth_cross"] = df["sentiment_z"] * df["breadth_z"]
    feat["cycle_flow_cross"] = df["market_cycle_z"] * feat["flow_composite"]
    feat["dispersion_vol_ratio"] = df["dispersion_z"] / (df["volatility"].clip(lower=1e-6) + 1) if "volatility" in df.columns else df["dispersion_z"]
    return feat

def compute_wasserstein_scores(feat_df, labels):
    idx = feat_df.index.intersection(labels.index)
    feat_df, labels = feat_df.loc[idx], labels.loc[idx]
    w1_p = (labels == 1).sum() / len(labels); w0_p = 1 - w1_p
    rows = []
    for col in feat_df.columns:
        f = feat_df[col].dropna(); l = labels.loc[f.index]
        f0, f1, fprior = f[l == 0].values, f[l == 1].values, f.values
        skipped = len(f0) < 10 or len(f1) < 10 or f0.std() < 1e-6 or f1.std() < 1e-6
        if skipped:
            rows.append({"feature": col, "w1_score": 0.0, "w1_normalized": 0.0, "status": "Skipped", "std": 0.0})
            continue
        score = w0_p * wasserstein_distance(fprior, f0) + w1_p * wasserstein_distance(fprior, f1)
        rows.append({"feature": col, "w1_score": score, "w1_normalized": score / fprior.std(), "status": "Kept", "std": fprior.std()})
    return pd.DataFrame(rows).sort_values("w1_score", ascending=False).reset_index(drop=True)

def deduplicate_by_correlation(scores_df, feat_df, max_corr=0.85):
    scores_df = scores_df.copy(); selected = []
    for i, row in scores_df.iterrows():
        if row["status"] == "Skipped": continue
        if not selected:
            selected.append(row["feature"])
        else:
            max_r = feat_df[selected + [row["feature"]]].corr().iloc[:-1, -1].abs().max()
            if max_r <= max_corr: selected.append(row["feature"])
            else: scores_df.at[i, "status"] = "Dropped"
    return scores_df, selected

# ============================================================
# STEP 1: compute_and_save_step1(df, ew_index_vn30, output_dir)
# ============================================================

def compute_and_save_step1(df, ew_index_vn30, output_dir):
    print("--- STEP 1: Compute Raw Sub-Features ---")
    feat_df = extract_candidate_features(df, ew_index_vn30)
    feat_df = winsorize_features(feat_df)
    feat_df.to_csv(os.path.join(output_dir, "step1_all_features.csv"))
    cols = feat_df.columns; n = len(cols)
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        if i < len(axes): axes[i].plot(feat_df.index, feat_df[col]); axes[i].set_title(col)
    for j in range(n, len(axes)): fig.delaxes(axes[j])
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "step1_features_overview.png")); plt.close()
    return feat_df

# ============================================================
# STEP 2: compute_and_save_step2(feat_df, df_vnindex, output_dir)
# ============================================================

def compute_and_save_step2(feat_df, df_vnindex, output_dir):
    print("--- STEP 2: Compute W1 Scores ---")
    labels = define_bull_base_labels(df_vnindex, lambda_value=DAILY_LAMBDA)
    labels.name = "market_trend_regime"
    df_vn = df_vnindex.copy().set_index("timestamp"); df_vn.index = pd.to_datetime(df_vn.index, utc=True)
    df_vn = df_vn.join(labels, how="inner")
    df_vn[["market_trend_regime", "close"]].rename(columns={"close":"vnindex_close"}).to_csv(os.path.join(output_dir, "step2_bull_labels.csv"))
    
    # Chart
    plot_df = df_vn[df_vn.index >= "2022-01-01"]
    fig, ax = plt.subplots(figsize=(15, 7)); ax.plot(plot_df.index, plot_df["close"], color="#2c3e50")
    for i in range(len(plot_df)-1):
        ax.axvspan(plot_df.index[i], plot_df.index[i+1], color="#d4edda" if plot_df["market_trend_regime"].iloc[i]==1 else "#f8d7da", alpha=0.4)
    flip = plot_df["market_trend_regime"].diff()
    ax.scatter(plot_df[flip==1].index, plot_df[flip==1]["close"], color="green", marker="^", s=50)
    ax.scatter(plot_df[flip==-1].index, plot_df[flip==-1]["close"], color="red", marker="v", s=50)
    plt.savefig(os.path.join(output_dir, "step2_bull_regime_chart.png")); plt.close()
    
    common = feat_df.index.intersection(labels.index)
    scores_df = compute_wasserstein_scores(feat_df.loc[common], labels.loc[common])
    scores_df, kept = deduplicate_by_correlation(scores_df, feat_df.loc[common])
    
    def get_comp(f):
        for k, v in pillar_map.items():
            if f in v: return k
        return "unknown"
    scores_df["component"] = scores_df["feature"].apply(get_comp)
    scores_df.to_csv(os.path.join(output_dir, "step2_w1_scores_all_features.csv"), index=False)
    
    # BarChart
    plot_s = scores_df.sort_values("w1_score", ascending=True)
    plt.figure(figsize=(10, 8)); plt.barh(plot_s["feature"], plot_s["w1_score"], color=["#27ae60" if s=="Kept" else "#bdc3c7" for s in plot_s["status"]])
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "step2_w1_barchart.png")); plt.close()
    
    # Dist grid
    top10 = scores_df.head(10)["feature"].tolist()
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    for i, f in enumerate(top10):
        prior = feat_df[f].dropna(); bull = feat_df.loc[common].loc[labels.loc[common]==1, f].dropna()
        axes[i].hist(prior, bins=50, density=True, alpha=0.5, label='Prior', color='grey')
        axes[i].hist(bull, bins=50, density=True, alpha=0.7, label='Bull', color='green')
        axes[i].set_title(f); axes[i].legend()
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "step2_distribution_comparison.png")); plt.close()
    return scores_df, feat_df.loc[common], labels.loc[common]

# ============================================================
# STEP 3: compute_and_save_step3(w1_scores_df, output_dir)
# ============================================================

def compute_and_save_step3(w1_df, feat_matched, output_dir):
    print("--- STEP 3: Component Selection ---")
    comp_scores = w1_df[w1_df["status"] != "Skipped"].groupby("component")["w1_score"].max().sort_values(ascending=False).reset_index()
    comp_scores["rank"] = range(1, len(comp_scores) + 1); comp_scores["selected"] = comp_scores["rank"] <= 5
    comp_scores.to_csv(os.path.join(output_dir, "step3_component_w1_scores.csv"), index=False)
    
    plt.figure(figsize=(10, 6)); plt.barh(comp_scores["component"][::-1], comp_scores["w1_score"][::-1], color=["#27ae60" if s else "#bdc3c7" for s in comp_scores["selected"][::-1]])
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "step3_component_ranking.png")); plt.close()
    
    top5 = comp_scores[comp_scores["selected"]]["component"].tolist()
    final_feats = [w1_df[(w1_df["component"] == c) & (w1_df["status"] != "Skipped")].head(2) for c in top5]
    final_df = pd.concat(final_feats).sort_values("w1_score", ascending=False)
    final_df["overall_rank"] = range(1, len(final_df) + 1)
    final_df[["overall_rank", "component", "feature", "w1_score", "w1_normalized"]].to_csv(os.path.join(output_dir, "step3_selected_10_features.csv"), index=False)
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    for i, (_, r) in enumerate(final_df.iterrows()):
        axes[i].plot(feat_matched.index, feat_matched[r["feature"]])
        axes[i].set_title(f"{r['feature']} (W1: {r['w1_score']:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "step3_selected_features_grid.png")); plt.close()
    return final_df["feature"].tolist()

# ============================================================
# STEP 4: compute_and_save_step4(df, feat_df, selected_10, ew_index, labels, output_dir)
# ============================================================

def compute_and_save_step4(df_vnindex, feat_matched, selected_10, ew_index, labels_matched, output_dir):
    print("--- STEP 4: Market Health Index ---")
    z_feats = {}; noise_rows = []
    for f in selected_10:
        ser_z = winsorize_features(pd.DataFrame({f: compute_z_score(feat_matched[f])}))[f]
        z_feats[f] = ser_z
        sflip = (np.sign(ser_z).diff() != 0).mean() * 100; acorr = ser_z.autocorr(lag=1)
        noise_rows.append({"feature": f, "daily_std": ser_z.diff().std(), "sign_flip_pct": sflip, "autocorr_1": acorr, "noise_flag": "⚠ NOISY" if sflip > 10 or acorr < 0.9 else "OK"})
    pd.DataFrame(noise_rows).to_csv(os.path.join(output_dir, "step4_noise_diagnostics.csv"), index=False)
    df_z = pd.DataFrame(z_feats)
    weights_df = compute_pca_weights(df_z, selected_10)
    composite = (df_z * weights_df[selected_10]).sum(axis=1)
    
    # Use line plot for weights to avoid ValueError if loadings flip signs over time
    weights_df[selected_10].plot(figsize=(15, 7), linewidth=1.5, alpha=0.8)
    plt.title("Rolling PCA Loadings (Dynamic Feature Weights)")
    plt.ylabel("Loading Weight")
    plt.axhline(0, color='black', lw=1, ls='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step4_pca_weights_over_time.png")); plt.close()
    
    vol = compute_garch_rolling(ew_index).reindex(composite.index).ffill()["sigma_ann"]
    agr = calculate_signal_agreement(df_z, selected_10)
    signal = apply_vol_weighted_hysteresis(composite, vol, consensus_series=agr)
    
    final_df = df_z.copy(); final_df["composite_health"] = composite; final_df["volatility"] = vol; final_df["agreement"] = agr; final_df["health_signal_final"] = signal; final_df["market_trend_regime"] = labels_matched.reindex(final_df.index).ffill()
    vn_c = df_vnindex.copy().set_index("timestamp"); vn_c.index = pd.to_datetime(vn_c.index, utc=True)
    final_df["vnindex_close"] = vn_c["close"].reindex(final_df.index).ffill()
    final_df.to_csv(os.path.join(output_dir, "step4_market_health_final.csv"))
    
    # Visualization - Match plot_market_health_v2.py style
    plot_df = final_df[final_df.index >= "2024-01-01"]
    if plot_df.empty:
        plot_df = final_df.tail(252) # fallback

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Top Plot: VNINDEX with Signal Background
    ax1.plot(plot_df.index, plot_df["vnindex_close"], color="#333333", linewidth=1.5, label="VNINDEX", alpha=0.8)
    for i in range(len(plot_df)-1):
        color = "#d4edda" if plot_df["health_signal_final"].iloc[i] == 1 else "#f8d7da"
        ax1.axvspan(plot_df.index[i], plot_df.index[i+1], color=color, alpha=0.4, linewidth=0)
    
    flip = plot_df["health_signal_final"].diff()
    buys = plot_df[flip == 1]
    sells = plot_df[flip == -1]
    
    ax1.scatter(buys.index, buys["vnindex_close"], color="green", s=40, edgecolors="white", zorder=5, label="Signal ON")
    ax1.scatter(sells.index, sells["vnindex_close"], color="red", s=40, edgecolors="white", zorder=5, label="Signal OFF")
    
    ax1.set_title("VNINDEX Market Health Index (W1-Driven)", fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel("VNINDEX")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")

    # Bottom Plot: Composite Health Score
    ax2.plot(plot_df.index, plot_df["composite_health"], color="#6c5ce7", linewidth=1.5, label="Composite Health")
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    
    # Use numpy arrays, timezone-naive index, and interpolate=True to avoid TypeError
    # with certain versions of matplotlib/numpy when using 'where'
    x_vals = pd.to_datetime(plot_df.index).tz_localize(None).values
    y_vals = plot_df["composite_health"].values.astype(float)
    ax2.fill_between(x_vals, 0, y_vals, where=(y_vals > 0), color="green", alpha=0.1, interpolate=True)
    ax2.fill_between(x_vals, 0, y_vals, where=(y_vals < 0), color="red", alpha=0.1, interpolate=True)
    ax2.set_ylabel("Health Score")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step4_market_health_chart.png"), dpi=200)
    plt.close()
    
    with open(os.path.join(output_dir, "step4_summary.txt"), "w") as f_sum:
        f_sum.write(f"Total Days: {len(final_df)}\nFlips: {int((pd.Series(signal).diff()!=0).sum())}\nRegime: {'BULLISH' if signal[-1]==1 else 'BEARISH'}\n\nFeatures:\n")
        for ft in selected_10: f_sum.write(f" - {ft}\n")
        f_sum.write(f"\nLast 20 Days:\n{final_df[['vnindex_close', 'composite_health', 'health_signal_final']].tail(20).to_string()}")

# ============================================================
# MAIN
# ============================================================
# ============================================================
# MAIN PIPELINE EXECUTION
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    # Save outputs in the current application directory
    base_out = os.path.dirname(__file__)
    for d in ["step1_features", "step2_w1_scores", "step3_component_selection", "step4_market_health_index"]:
        os.makedirs(os.path.join(base_out, d), exist_ok=True)
    
    print("Fetching data from QuestDB...")
    univ100 = load_vn100_universe(); univ30 = load_vn30_universe()
    df100 = load_universe_prices(univ100, label="VN100"); df30 = load_universe_prices(univ30, label="VN30")
    vnidx = load_vnindex(); bond = load_govbond(); mflow = load_money_flow(); foreign = load_foreign_trading(); prop = load_proprietary_trading()
    
    # Handle time-varying membership for VN30 and VN100
    max_dt = vnidx["timestamp"].max()
    map100 = build_membership_map(univ100, max_dt); map30 = build_membership_map(univ30, max_dt)
    ew30 = build_ew_index(df30, map30); ew100 = build_ew_index(df100, map100)
    
    print("Computing the 8 Major Pillars...")
    b = measure_market_breadth(df30, map30).rename(columns={"breadth": "breadth_z"})
    mom = measure_momentum(ew30).rename(columns={"momentum": "momentum_z"})
    sent = measure_sentiment(vnidx, df100, bond).rename(columns={"sentiment": "sentiment_z"})
    cyc = measure_market_cycle(df100, map100).rename(columns={"market_cycle": "market_cycle_z"})
    mf = measure_money_flow(mflow, map30).rename(columns={"money_flow": "money_flow_z"})
    ff = measure_institutional_flow(foreign, label="foreign").rename(columns={"foreign_flow": "foreign_flow_z"})
    pf = measure_institutional_flow(prop, label="prop").rename(columns={"prop_flow": "prop_flow_z"})
    disp = measure_dispersion_index(df100, map100).rename(columns={"dispersion": "dispersion_z"})
    
    # Merge all pillars. We fill missing history (e.g., Prop flow) with 0 
    # to ensure the timeline starts in 2016, not just when the newest table starts.
    df_p = pd.concat([b, mom, sent, cyc, mf, ff, pf, disp], axis=1).ffill().fillna(0).dropna()
    df_z = winsorize_features(compute_z_score(winsorize_features(df_p)))
    
    # Add raw columns for specific sub-feature logic
    df_z["breadth"] = b["breadth_z"].reindex(df_z.index).ffill()
    
    # STEP 1: Generate 20+ candidate sub-features (slopes, means, crosses)
    feat_df = compute_and_save_step1(df_z, ew30, os.path.join(base_out, "step1_features"))
    
    # STEP 2: Use Wasserstein Distance (W1) to score features against historical Bull regimes
    w1_df, f_m, l_m = compute_and_save_step2(feat_df, vnidx, os.path.join(base_out, "step2_w1_scores"))
    
    # STEP 3: Select the top 10 most predictive features based on W1 and deduplication
    sel_10 = compute_and_save_step3(w1_df, f_m, os.path.join(base_out, "step3_component_selection"))
    
    # STEP 4: Build Composite Health Index using PCA and generate visual/text signals
    compute_and_save_step4(vnidx, f_m, sel_10, ew100, l_m, os.path.join(base_out, "step4_market_health_index"))
    print("DONE.")
