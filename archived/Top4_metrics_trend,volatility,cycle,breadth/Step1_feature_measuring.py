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
# NEW HELPERS FOR 4 PILLARS
# ============================================================
def _get_roc(close, n):
    return close.pct_change(n) * 100.0

def _get_kst(close, sma_windows=(10, 10, 10, 15), roc_windows=(10, 15, 20, 30)):
    rcma1 = _get_roc(close, roc_windows[0]).rolling(sma_windows[0]).mean()
    rcma2 = _get_roc(close, roc_windows[1]).rolling(sma_windows[1]).mean()
    rcma3 = _get_roc(close, roc_windows[2]).rolling(sma_windows[2]).mean()
    rcma4 = _get_roc(close, roc_windows[3]).rolling(sma_windows[3]).mean()
    kst = (1.0 * rcma1) + (2.0 * rcma2) + (3.0 * rcma3) + (4.0 * rcma4)
    return kst

def _season4_from_kst(kst):
    dkst = kst.diff()
    pos, neg = kst >= 0, kst < 0
    up, dn = dkst > 0, dkst < 0
    season = pd.Series(0.0, index=kst.index, dtype='float32')
    season[pos & up] = 1.0
    season[neg & up] = 0.5
    season[pos & dn] = -0.5
    season[neg & dn] = -1.0
    return season

def extract_signal_generation_features(ew_df, vnidx_df, univ_prices_df, membership_map):
    """
    Implements core market health pillars:
    1. market_trend: Price distance from EMA50
    2. market_cycle: Mean KST-based market season across VN100
    3. market_breadth: % of VN100 stocks above EMA200
    4. market_volatility: EGARCH-based annualized volatility of the EW index
    """
    ew_df = ew_df.copy()
    _ew_idx = pd.to_datetime(ew_df.index)
    ew_df.index = (_ew_idx.tz_convert(None) if _ew_idx.tz is not None else _ew_idx).normalize()
    
    feat = pd.DataFrame(index=ew_df.index)
    p = ew_df['close']
    
    # 1. Market Trend (EMA50) - Modified from f1_dist_long_ma to use EMA50
    ema50 = p.ewm(span=50, adjust=False).mean()
    feat['market_trend'] = (p - ema50) / ema50
    
    # Prepare wide price panel for aggregate metrics
    _prices = univ_prices_df.pivot(index='timestamp', columns='symbol', values='close').sort_index()
    _prices.index = pd.to_datetime(_prices.index)
    _prices.index = (_prices.index.tz_convert(None) if _prices.index.tz is not None else _prices.index).normalize()
    _prices = _prices.reindex(ew_df.index).ffill()
    
    # 2. Market Breadth (% of symbols above EMA200)
    _ema200_all = _prices.apply(lambda x: x.ewm(span=200, adjust=False).mean())
    _breadth_flags = (_prices > _ema200_all).astype(float)
    feat['market_breadth'] = aggregate_by_membership(_breadth_flags, membership_map)
    
    # 3. Market Cycle (KST Seasons mapped to [-1, 1])
    def get_season(s):
        kst = _get_kst(s.ffill())
        return _season4_from_kst(kst)
    
    _seasons = _prices.apply(get_season)
    feat['market_cycle'] = aggregate_by_membership(_seasons, membership_map)
    
    # 4. Market Volatility (Annualized EGARCH(1,1)-t)
    try:
        from arch import arch_model
        rets = np.log(p / p.shift(1)).dropna()
        if not rets.empty:
            rets_pct = 100.0 * rets
            am = arch_model(rets_pct, mean="ARX", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
            res = am.fit(disp="off")
            sigma_daily = res.conditional_volatility / 100.0
            feat['market_volatility'] = sigma_daily.reindex(ew_df.index) * np.sqrt(252)
        else:
            feat['market_volatility'] = np.nan
    except Exception:
        # Fallback to simple rolling volatility
        feat['market_volatility'] = p.pct_change().rolling(20).std() * np.sqrt(252)
        
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
    
    print("Computing 4 Market Health Pillars...")
    feat_df = extract_signal_generation_features(ew100, vnidx, df100, map100)

    
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

    print("Step 1 Complete: 4 Market Health Pillars computed.")

