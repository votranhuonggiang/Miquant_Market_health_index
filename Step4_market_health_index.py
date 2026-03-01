import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG & CONSTANTS (From pipeline)
# ============================================================
PCA_WINDOW = 252
GARCH_BURN_IN = 252
GARCH_REFIT_FREQ = 63
BASE_BUFFER = 1
MIN_HOLD = 15

def query_questdb(sql_query):
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
        return df
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def load_vnindex():
    return query_questdb(
        "SELECT * FROM raw_eod WHERE symbol = 'VNINDEX' AND timestamp >= '2016-01-01'"
    )

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

# ============================================================
# INDEX BUILDER (To get EW Index for Volatility calculation)
# ============================================================
def build_membership_map(universe_df, end_date):
    df = universe_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["period"] = df["timestamp"].dt.to_period("M")
    base_map = {p: set(g["symbol"].astype(str)) for p, g in df.groupby("period")}
    if not base_map: return {}
    periods_sorted = sorted(base_map.keys())
    start_p = periods_sorted[0]
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
# NOISE REDUCTION HELPERS (From pipeline)
# ============================================================
def compute_z_score(series, window=252): # window=LOOKBACK_PERIOD
    eff_w = min(window, max(len(series) // 3, 15))
    min_p = max(eff_w // 4, 5)
    mu, sigma = series.rolling(window=eff_w, min_periods=min_p).mean(), series.rolling(window=eff_w, min_periods=min_p).std()
    return ((series - mu) / sigma).clip(-3, 3)

def winsorize_features(feat_df, lower=1, upper=99):
    out = feat_df.copy()
    for col in out.columns:
        valid = out[col].dropna()
        if valid.empty: continue
        lo = np.nanpercentile(valid, lower); hi = np.nanpercentile(valid, upper)
        out[col] = out[col].clip(lo, hi)
    return out

# ============================================================
# MARKET HEALTH LOGIC (From pipeline)
# ============================================================
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
        pca = PCA(n_components=1)
        pca.fit(scaled)
        loadings = pca.components_[0]
        if np.sum(loadings) < 0: loadings = -loadings
        weights_df.iloc[i] = loadings
    return weights_df.ffill().fillna(1.0 / len(cols))

def compute_garch_rolling(ew_index_df, reestimate_freq=GARCH_REFIT_FREQ, burn_in=GARCH_BURN_IN):
    df = ew_index_df.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index(); price = df["close"].astype(float)
    rets = (np.log(price / price.shift(1)).dropna()) * 100.0
    sigmas = pd.Series(index=price.index, dtype=float)
    eff_b = min(burn_in, max(len(rets) // 2, 30))
    eff_r = min(reestimate_freq, max(len(rets) // 4, 10))
    if len(rets) < 30:
        sigmas = rets.rolling(20, min_periods=5).std() * np.sqrt(252) / 100.0
        return pd.DataFrame({"sigma_ann": sigmas}, index=price.index)
    for i in range(eff_b, len(price), eff_r):
        end_idx = min(i + eff_r, len(price))
        train = rets.iloc[:i]
        try:
            am = arch_model(train, mean="ARX", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
            res = am.fit(disp="off", show_warning=False)
            am_next = arch_model(rets.iloc[:end_idx], mean="ARX", vol="EGARCH", p=1, o=1, q=1, dist="studentst")
            fixed_res = am_next.fix(res.params)
            if i == eff_b: sigmas.iloc[1:i+1] = fixed_res.conditional_volatility.iloc[:i].values / 100.0
            sigmas.iloc[i:end_idx] = fixed_res.conditional_volatility.iloc[i:end_idx].values / 100.0
        except: continue
    if sigmas.isna().all(): sigmas = rets.rolling(20, min_periods=5).std() / 100.0
    return pd.DataFrame({"sigma_ann": sigmas * np.sqrt(252)}, index=price.index)

def calculate_signal_agreement(df, cols):
    return (np.sign(df[cols]).sum(axis=1) / len(cols)).abs()

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

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, "step4_market_health_index")
    os.makedirs(output_dir, exist_ok=True)
    
    feat_file = os.path.join(base_dir, "step1_features", "step1_all_features.csv")
    sel_file = os.path.join(base_dir, "step3_component_selection", "step3_selected_5_features.csv")
    
    if not os.path.exists(feat_file) or not os.path.exists(sel_file):
        print("Missing input files. Run steps 1 and 3 first.")
        exit()
        
    print("Loading selected features and data...")
    sel_df = pd.read_csv(sel_file)
    selected_cols = sel_df["feature"].tolist()
    feat_df = pd.read_csv(feat_file, index_col=0)
    feat_df.index = pd.to_datetime(feat_df.index).tz_localize(None)
    
    # 1. Pre-process Features: Noise Diagnostics & Cleaning
    print(f"Applying noise reduction to: {selected_cols}")
    processed_feats = {}
    noise_rows = []
    
    for f in selected_cols:
        # Re-apply Z-score and Winsorize for the specific sample to anchor them
        ser_raw = feat_df[f].ffill()
        ser_z = compute_z_score(ser_raw)
        ser_clean = winsorize_features(pd.DataFrame({f: ser_z}))[f]
        processed_feats[f] = ser_clean
        
        # Noise Diagnostics
        sflip = (np.sign(ser_clean).diff() != 0).mean() * 100
        acorr = ser_clean.autocorr(lag=1)
        noise_rows.append({
            "feature": f, 
            "daily_std": ser_clean.diff().std(), 
            "sign_flip_pct": sflip, 
            "autocorr_1": acorr, 
            "noise_flag": "⚠ NOISY" if sflip > 10 or acorr < 0.9 else "OK"
        })
    
    # Save diagnostics
    diag_df = pd.DataFrame(noise_rows)
    diag_df.to_csv(os.path.join(output_dir, "step4_noise_diagnostics.csv"), index=False)
    print("Noise diagnostics saved to step4_noise_diagnostics.csv")
    
    df_z = pd.DataFrame(processed_feats).ffill().fillna(0)
    
    # Calculate Health Score Components
    weights_df = compute_pca_weights(df_z, selected_cols)
    composite_health = (df_z * weights_df[selected_cols]).sum(axis=1)
    
    # 2. Get Volatility for Hysteresis
    print("Fetching VN100 data for GARCH volatility...")
    univ100 = load_vn100_universe()
    prices100 = load_universe_prices(univ100)
    vnidx = load_vnindex()
    
    max_dt = vnidx["timestamp"].max()
    membership_map = build_membership_map(univ100, max_dt)
    ew_index = build_ew_index(prices100, membership_map)
    ew_index.index = pd.to_datetime(ew_index.index).tz_localize(None)
    
    print("Computing GARCH Volatility...")
    vol_df = compute_garch_rolling(ew_index)
    vol_series = vol_df["sigma_ann"].reindex(composite_health.index).ffill().fillna(vol_df["sigma_ann"].mean())
    
    # 3. Apply Signal Logic
    print("Applying volatility-weighted hysteresis signal...")
    agreement = calculate_signal_agreement(df_z, selected_cols)
    health_signal = apply_vol_weighted_hysteresis(composite_health, vol_series, agreement)
    
    # 4. Final Data Assembly
    print("Assembling final results...")
    vn_c = vnidx.set_index("timestamp").copy()
    vn_c.index = pd.to_datetime(vn_c.index).tz_localize(None)
    
    final_df = pd.DataFrame({
        "vnindex_close": vn_c["close"].reindex(composite_health.index).ffill(),
        "composite_health": composite_health,
        "health_signal": health_signal,
        "volatility": vol_series,
        "agreement": agreement
    }, index=composite_health.index)
    
    # Add the constituent feature Z-scores
    for col in selected_cols:
        final_df[col] = df_z[col]
        
    final_df.to_csv(os.path.join(output_dir, "step4_market_health_final.csv"))
    
    # 5. Visualization
    print("Generating charts...")
    plot_df = final_df[final_df.index >= "2024-01-01"]
    if plot_df.empty: plot_df = final_df.tail(252)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    
    # Top Plot: VNINDEX with Signal Background
    ax1.plot(plot_df.index, plot_df["vnindex_close"], color="#2d3436", linewidth=1.5, label="VNINDEX")
    for i in range(len(plot_df)-1):
        color = "#d4edda" if plot_df["health_signal"].iloc[i] == 1 else "#f8d7da"
        ax1.axvspan(plot_df.index[i], plot_df.index[i+1], color=color, alpha=0.4, linewidth=0)
    
    flip = plot_df["health_signal"].diff()
    ax1.scatter(plot_df[flip==1].index, plot_df[flip==1]["vnindex_close"], color="green", marker="^", s=60, label="Signal ON", zorder=5)
    ax1.scatter(plot_df[flip==-1].index, plot_df[flip==-1]["vnindex_close"], color="red", marker="v", s=60, label="Signal OFF", zorder=5)
    
    ax1.set_title("VNINDEX Market Health Index (W1-Selected Features)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("VNINDEX Level")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.2)
    
    # Bottom Plot: Health Score
    x_vals = plot_df.index.values
    y_vals = plot_df["composite_health"].values.astype(float)
    ax2.plot(plot_df.index, y_vals, color="#0984e3", linewidth=1.5, label="Health Score")
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.fill_between(x_vals, 0, y_vals, where=(y_vals > 0), color="green", alpha=0.1, interpolate=True)
    ax2.fill_between(x_vals, 0, y_vals, where=(y_vals < 0), color="red", alpha=0.1, interpolate=True)
    ax2.set_ylabel("Composite Health Score")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step4_market_health_chart.png"), dpi=200)
    plt.close()
    
    # 6. Weights Visualization
    plt.figure(figsize=(12, 6))
    weights_df[selected_cols].plot(ax=plt.gca(), linewidth=1.5, alpha=0.8)
    plt.title("Dynamic Feature Weights (Rolling PCA Loadings)")
    plt.ylabel("Weight")
    plt.axhline(0, color='black', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step4_feature_weights_over_time.png"))
    plt.close()
    
    print(f"Step 4 Complete. Output saved to {output_dir}")
