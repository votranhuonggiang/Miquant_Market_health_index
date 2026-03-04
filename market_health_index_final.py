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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pykalman import KalmanFilter
from arch import arch_model
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================
COMPONENTS = ["sentiment_z", "breadth_z", "volatility_z", "cycle_z"]
LOOKBACK_PERIOD = 252
PCA_WINDOW = 252
WEIGHT_SMOOTH_WINDOW = 21 # 1 month smoothing for weights
OUTPUT_DIR = "output_final"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================
# DATA LOADING (Reference: market_health_w1_pipeline.py)
# ============================================================

def query_questdb(sql_query, label="Data"):
    host = os.environ.get("QUEST_DB_URL", "http://localhost:9000")
    auth = (os.getenv("QUESTDB_USERNAME"), os.getenv("QUESTDB_PASSWORD"))
    try:
        response = requests.get(
            host + "/exec", params={"query": sql_query}, auth=auth, timeout=60
        ).json()
        if "dataset" not in response or "columns" not in response:
            return pd.DataFrame()
        
        df = pd.DataFrame(
            response["dataset"],
            columns=pd.DataFrame(response["columns"])["name"].values,
        )
        if not df.empty and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
            df = df.sort_values("timestamp")
        return df
    except Exception as e:
        print(f"Error querying {label}: {e}")
        return pd.DataFrame()

def load_vn100_universe():
    return query_questdb("""
        SELECT symbol, timestamp
        FROM ai_ranking_universe
        WHERE universe = 'VN100'
          AND timestamp >= '2016-01-01'
    """, label="VN100 Universe Map")

def load_universe_prices(universe_df):
    if universe_df.empty: return pd.DataFrame()
    symbol_list = universe_df["symbol"].unique().tolist()
    in_clause = "('" + "','".join(symbol_list) + "')"
    return query_questdb(f"""
        SELECT timestamp, symbol, close, high, low, volume
        FROM raw_eod
        WHERE symbol IN {in_clause}
          AND timestamp >= '2016-01-01'
    """, label="VN100 Prices")

def load_vnindex():
    return query_questdb(
        "SELECT timestamp, close FROM raw_eod WHERE symbol = 'VNINDEX' AND timestamp >= '2016-01-01'",
        label="VNINDEX Close"
    )

def load_govbond():
    return query_questdb(
        "SELECT timestamp, close FROM raw_bond_yield WHERE symbol = 'VN10Y' AND timestamp > '2016-01-01'",
        label="Gov Bond 10Y"
    )

def build_membership_map(universe_df, end_date):
    df = universe_df.copy()
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["period"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)),
        freq="M"
    )
    base_map = {p: set(g["symbol"].astype(str)) for p, g in df.groupby("period")}
    if not base_map: return {}
    periods_sorted = sorted(base_map.keys())
    start_p, last_p = periods_sorted[0], periods_sorted[-1]
    end_p = pd.Period(pd.to_datetime(end_date), freq="M")
    filled = {}
    last_set = None
    for p in pd.period_range(start=start_p, end=last_p, freq="M"):
        if p in base_map: last_set = base_map[p]
        if last_set is not None: filled[p] = last_set
    if end_p > last_p:
        for p in pd.period_range(start=last_p + 1, end=end_p, freq="M"):
            filled[p] = last_set
    return filled

def aggregate_by_membership(panel, membership_map):
    if panel.empty or not membership_map: return pd.Series(dtype=float)
    panel = panel.copy()
    panel.index = pd.to_datetime(panel.index)
    out_parts = []
    months = panel.index.to_period("M")
    for month, month_block in panel.groupby(months):
        members = membership_map.get(month, None)
        if not members: continue
        cols = month_block.columns.intersection(pd.Index(sorted(members)))
        if len(cols) == 0: continue
        out_parts.append(month_block[cols].mean(axis=1, skipna=True))
    return pd.concat(out_parts).sort_index() if out_parts else pd.Series(dtype=float)

def build_ew_index(vn_df, membership_map=None):
    prices = vn_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    daily_returns = prices.pct_change().dropna(how="all")
    if membership_map:
        ew_returns = aggregate_by_membership(daily_returns, membership_map)
    else:
        ew_returns = daily_returns.mean(axis=1)
    ew_level = (1 + ew_returns.dropna()).cumprod() * 100.0
    return pd.DataFrame({"close": ew_level}, index=ew_level.index)

def winsorize_features(feat_df, lower=1, upper=99):
    out = feat_df.copy()
    if isinstance(out, pd.Series):
        lo = np.nanpercentile(out.dropna(), lower)
        hi = np.nanpercentile(out.dropna(), upper)
        return out.clip(lo, hi)
    for col in out.columns:
        lo = np.nanpercentile(out[col].dropna(), lower)
        hi = np.nanpercentile(out[col].dropna(), upper)
        out[col] = out[col].clip(lo, hi)
    return out

# ============================================================
# METRICS CALCULATIONS (Reference: sample files)
# ============================================================

def calculate_sentiment(vnindex_df, vn100_df, bond_df):
    # Logic from SYNTH_technical_sentiment_eod.py
    idx = vnindex_df.copy().set_index("timestamp")
    prices = vn100_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    highs = vn100_df.pivot(index="timestamp", columns="symbol", values="high").ffill()
    lows = vn100_df.pivot(index="timestamp", columns="symbol", values="low").ffill()
    bond = bond_df.copy().rename(columns={"close":"yield"}).set_index("timestamp")
    
    # Breadth component
    sent_breadth = np.sign(prices.pct_change()).sum(axis=1)
    
    # Strength component (New Highs - New Lows)
    max_h = highs.shift(1).rolling(252, min_periods=30).max().fillna(highs)
    new_h = (highs > max_h).astype(int).sum(axis=1)
    min_l = lows.shift(1).rolling(252, min_periods=30).min().fillna(lows)
    new_l = (lows < min_l).astype(int).sum(axis=1)
    strength = new_h - new_l
    
    # Safe Haven demand
    common = idx.index.intersection(bond.index)
    idx_r = idx.loc[common, "close"].pct_change(20)
    pv = 1000 / (1 + bond.loc[common, "yield"] / 100) ** 10
    bond_r = pv.pct_change(20)
    safe_haven = (idx_r - bond_r).reindex(sent_breadth.index).ffill()
    
    df_sent = pd.DataFrame({"breadth": sent_breadth, "strength": strength, "safe_haven": safe_haven}).dropna()
    # Normalize components
    df_sent = (df_sent - df_sent.rolling(252).mean()) / df_sent.rolling(252).std()
    raw_score = df_sent.sum(axis=1).dropna()
    raw_score = (raw_score - raw_score.rolling(252).mean()) / raw_score.rolling(252).std()
    raw_score = raw_score.dropna()
    
    # Kalman Filter
    kf = KalmanFilter(transition_matrices=[[1]], observation_matrices=[[1]], initial_state_mean=raw_score.iloc[0], 
                      initial_state_covariance=1, observation_covariance=0.1, transition_covariance=1e-2)
    smoothed, _ = kf.smooth(raw_score.values)
    return pd.Series(smoothed.flatten(), index=raw_score.index, name="sentiment")

def calculate_breadth(vn100_df, membership_map):
    # Logic from market_breadth.py (% above EMA50)
    prices = vn100_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    ema50 = prices.apply(lambda x: x.ewm(span=50, adjust=False).mean())
    flags = (prices > ema50).astype(float)
    breadth = aggregate_by_membership(flags, membership_map)
    return pd.Series(breadth, name="breadth")

def calculate_volatility(ew_index):
    # Logic from macd_volatility.py (GARCH annualized)
    rets = np.log(ew_index["close"] / ew_index["close"].shift(1)).dropna() * 100.0
    am = arch_model(rets, mean="ARX", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
    res = am.fit(disp="off", show_warning=False)
    sigma_daily = res.conditional_volatility / 100.0
    sigma_ann = sigma_daily * np.sqrt(252)
    return pd.Series(sigma_ann, name="volatility")

def calculate_cycle(vn100_df, membership_map):
    # Logic from market_cycle (1).py (KST seasons)
    def _get_kst(close):
        def _roc(s, n): return ((s - s.shift(n)) / s.shift(n)) * 100.0
        rcma1 = _roc(close, 10).rolling(10).mean()
        rcma2 = _roc(close, 15).rolling(10).mean()
        rcma3 = _roc(close, 20).rolling(10).mean()
        rcma4 = _roc(close, 30).rolling(15).mean()
        return (1.0 * rcma1) + (2.0 * rcma2) + (3.0 * rcma3) + (4.0 * rcma4)
    
    def _season(kst):
        dkst = kst.diff()
        s = pd.Series(0.0, index=kst.index)
        s[(kst >= 0) & (dkst > 0)] = 1.0; s[(kst < 0) & (dkst > 0)] = 0.5; s[(kst >= 0) & (dkst < 0)] = -0.5; s[(kst < 0) & (dkst < 0)] = -1.0
        return s

    prices = vn100_df.pivot(index="timestamp", columns="symbol", values="close").ffill()
    seasons = pd.DataFrame(index=prices.index)
    for col in prices.columns:
        kst = _get_kst(prices[col])
        seasons[col] = _season(kst)
    
    cycle = aggregate_by_membership(seasons, membership_map)
    return pd.Series(cycle, name="market_cycle")

# ============================================================
# COMBINATION LOGIC (Reference: synth_market_health_v2.py)
# ============================================================

def compute_z_score(series, window=LOOKBACK_PERIOD):
    mu = series.rolling(window=window).mean()
    sigma = series.rolling(window=window).std()
    return ((series - mu) / sigma).clip(-3, 3)

def compute_pca_weights(df, cols, window=PCA_WINDOW, smooth_window=WEIGHT_SMOOTH_WINDOW):
    weights_df = pd.DataFrame(index=df.index, columns=cols)
    for i in range(window, len(df)):
        chunk = df[cols].iloc[i - window : i].dropna()
        if len(chunk) < window * 0.5: continue
        scaler = StandardScaler()
        scaled = scaler.fit_transform(chunk)
        pca = PCA(n_components=1)
        pca.fit(scaled)
        loadings = pca.components_[0]
        if np.sum(loadings) < 0: loadings = -loadings
        weights_df.iloc[i] = loadings
    
    weights_df = weights_df.ffill().fillna(1.0 / len(cols))
    # Noise reduction in weighting via smoothing
    if smooth_window > 1:
        weights_df = weights_df.rolling(window=smooth_window, min_periods=1).mean()
        # Re-normalize to ensure they sum to 1 if needed, though PCA loadings don't necessarily sum to 1
        # but for contribution analysis it's better if they are scaled.
        weights_df = weights_df.div(weights_df.sum(axis=1), axis=0).fillna(1.0 / len(cols))
    return weights_df

def calculate_agreement(df, cols):
    signs = np.sign(df[cols])
    return (signs.sum(axis=1) / len(cols)).abs()

def apply_hysteresis(scores, vol_series, agreement, base_buffer=0.2, min_hold=3):
    states = np.zeros_like(scores)
    last_state = 0
    days_held = 0
    vol_mult = (vol_series / vol_series.rolling(252).mean()).fillna(1.0)
    for i in range(len(scores)):
        buff = base_buffer * vol_mult.iloc[i] * (2.0 - agreement.iloc[i])
        new_state = last_state
        if scores.iloc[i] > buff: new_state = 1
        elif scores.iloc[i] < -buff: new_state = 0
        
        if new_state != last_state:
            if days_held < min_hold: new_state = last_state; days_held += 1
            else: days_held = 1
        else: days_held += 1
        last_state = new_state
        states[i] = last_state
    return states

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_final_health(df, output_path):
    plt.style.use('default')
    plot_df = df[df.index >= "2023-01-01"]
    if plot_df.empty: plot_df = df.tail(252)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # Top: VN100 EW with signal
    ax1.plot(plot_df.index, plot_df["vn100_ew"], color="#2d3436", linewidth=2.0, label="VN100 EW", alpha=0.9)
    for i in range(len(plot_df)-1):
        color = "#d1f2e9" if plot_df["signal"].iloc[i] == 1 else "#fadbd8"
        ax1.axvspan(plot_df.index[i], plot_df.index[i+1], color=color, alpha=0.5, lw=0)
    
    flip = plot_df["signal"].diff()
    ax1.scatter(plot_df[flip==1].index, plot_df.loc[flip==1, "vn100_ew"], color="#10ac84", marker="^", s=100, label="Signal ON", edgecolors='black', zorder=5)
    ax1.scatter(plot_df[flip==-1].index, plot_df.loc[flip==-1, "vn100_ew"], color="#ee5253", marker="v", s=100, label="Signal OFF", edgecolors='black', zorder=5)
    
    ax1.set_title("VN100 EW Market Health Index (Composite V2)", fontsize=18, fontweight="bold", color="#2d3436", pad=20)
    ax1.set_ylabel("VN100 EW Level", color="#2d3436", fontsize=12)
    ax1.legend(facecolor='white', edgecolor='gray')
    ax1.grid(color='gray', linestyle='--', alpha=0.3)

    # Bottom: Composite Health
    ax2.plot(plot_df.index, plot_df["composite_health"], color="#5d5fef", linewidth=2.0, label="Composite Health")
    ax2.axhline(0, color="black", ls="--", alpha=0.5)
    ax2.fill_between(plot_df.index, 0, plot_df["composite_health"], where=(plot_df["composite_health"] > 0), color="#10ac84", alpha=0.2, interpolate=True)
    ax2.fill_between(plot_df.index, 0, plot_df["composite_health"], where=(plot_df["composite_health"] < 0), color="#ee5253", alpha=0.2, interpolate=True)
    
    ax2.set_ylabel("Health Score", color="#2d3436", fontsize=12)
    ax2.legend(facecolor='white', edgecolor='gray')
    ax2.grid(color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()

def plot_contributions(df, output_path):
    plt.style.use('default')
    plot_df = df[df.index >= "2024-01-01"]
    if plot_df.empty: plot_df = df.tail(252)
    
    contribs = pd.DataFrame(index=plot_df.index)
    for col in COMPONENTS:
        contribs[col] = plot_df[col] * plot_df[f"weight_{col}"]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    ax.stackplot(contribs.index, [contribs[c] for c in COMPONENTS], labels=COMPONENTS, colors=colors, alpha=0.6)
    ax.plot(plot_df.index, plot_df["composite_health"], color="#2d3436", lw=2, label="Composite Health")
    ax.axhline(0, color="black", ls="--", alpha=0.5)
    
    ax.set_title("Metric Contribution Analysis", fontsize=18, fontweight="bold", color="#2d3436", pad=20)
    ax.set_ylabel("Contribution Score", color="#2d3436", fontsize=12)
    ax.legend(loc="upper left", facecolor='white', edgecolor='gray')
    ax.grid(color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white')
    plt.close()

# ============================================================
# MAIN
# ============================================================

def main():
    print("Fetching data...")
    univ_map = load_vn100_universe()
    max_dt = query_questdb("SELECT max(timestamp) as ts FROM raw_eod")["ts"].iloc[0]
    m_map = build_membership_map(univ_map, max_dt)
    prices_df = load_universe_prices(univ_map)
    vnindex = load_vnindex()
    bond = load_govbond()
    
    print("Building EW Index...")
    ew_idx = build_ew_index(prices_df, m_map)
    
    print("Calculating metrics...")
    sent = calculate_sentiment(vnindex, prices_df, bond)
    breadth = calculate_breadth(prices_df, m_map)
    vol = calculate_volatility(ew_idx)
    cycle = calculate_cycle(prices_df, m_map)
    
    print("Normalizing and combining...")
    df = pd.DataFrame(index=ew_idx.index)
    df["vn100_ew"] = ew_idx["close"]
    df["vnindex"] = vnindex.set_index("timestamp")["close"].reindex(df.index).ffill()
    df["sentiment"] = sent.reindex(df.index).ffill()
    df["breadth"] = breadth.reindex(df.index).ffill()
    df["volatility_raw"] = vol.reindex(df.index).ffill()
    df["cycle"] = cycle.reindex(df.index).ffill()
    
    df["sentiment_z"] = compute_z_score(winsorize_features(df["sentiment"]))
    df["breadth_z"] = compute_z_score(winsorize_features(df["breadth"]))
    df["volatility_z"] = -compute_z_score(winsorize_features(df["volatility_raw"])) # Inverse vol for health
    df["cycle_z"] = compute_z_score(winsorize_features(df["cycle"]))
    
    df = df.dropna(subset=COMPONENTS)
    weights = compute_pca_weights(df, COMPONENTS)
    df["composite_health"] = (df[COMPONENTS] * weights[COMPONENTS]).sum(axis=1)
    for col in COMPONENTS:
        df[f"weight_{col}"] = weights[col]
    
    df["agreement"] = calculate_agreement(df, COMPONENTS)
    df["signal"] = apply_hysteresis(df["composite_health"], df["volatility_raw"], df["agreement"])
    
    print("Saving results...")
    df.to_csv(os.path.join(OUTPUT_DIR, "market_health_final.csv"))
    plot_final_health(df, os.path.join(OUTPUT_DIR, "step4_market_health_chart.png"))
    plot_contributions(df, os.path.join(OUTPUT_DIR, "contrib_market_breadth.png"))
    print(f"Done. Outputs in {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()