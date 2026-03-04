import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, accuracy_score,
    f1_score, classification_report, confusion_matrix
)
import xgboost as xgb
import lightgbm as lgb
import cvxpy as cp
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
FORECAST_HORIZON    = 1     # Days ahead to predict
WALK_FORWARD_START  = "2020-01-01"
WINDOW_YEARS        = 5     # Train on the last 5 years
REFIT_FREQ          = 21    # Refit models every 21 trading days (approx monthly)
THRESHOLD           = 0.5   # Probability center for decision
HYSTERESIS_BUFFER   = 0.05  # Probability must be > 0.55 to turn ON, < 0.45 to turn OFF
MIN_HOLD_DAYS       = 5     # Minimum days to stay in a state before flipping
TREND_LAMBDA        = 5     # Lambda for L1 Trend Filtering label

# ============================================================
# PATHS
# ============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(BASE_DIR, "step4_market_health_index", "step4_market_health_final.csv")
LABEL_FILE  = os.path.join(BASE_DIR, "step2_w1_score", "step2_bull_labels.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "step5_predictive_modelling")
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ============================================================
# L1 TREND FILTERING (LABELING)
# ============================================================
def trend_filtering(data: np.ndarray, lambda_value: float) -> np.ndarray:
    """L1 Trend Filtering via CVXPY."""
    n = np.size(data)
    x_ret = data.reshape(n)

    # Difference matrix D
    Dfull = np.diag([1] * n) - np.diag([1] * (n - 1), 1)
    D = Dfull[0:(n - 1),]

    beta = cp.Variable(n)
    lambd = cp.Parameter(nonneg=True)

    def tf_obj(x, beta, lambd):
        return cp.norm(x - beta, 2) ** 2 + lambd * cp.norm(cp.matmul(D, beta), 1)

    problem = cp.Problem(cp.Minimize(tf_obj(x_ret, beta, lambd)))
    lambd.value = lambda_value
    
    # Attempt to solve with CLARABEL if available, fallback to default
    try:
        problem.solve(solver='CLARABEL')
    except:
        problem.solve()

    return beta.value


def detect_regime(series: pd.Series, lambda_value: float = TREND_LAMBDA) -> pd.Series:
    """
    Returns Bullish (1) or Bearish (0) regime based on L1 trend filter.
    """
    returns = series.pct_change().fillna(0.0) * 100
    betas = trend_filtering(returns.values, lambda_value)
    # Binary: 1 if beta > 0 (bullish return trend), else 0
    regime = np.where(betas > 0, 1, 0)
    return pd.Series(regime, index=series.index)


# ============================================================
# STEP A — BUILD LABEL: From Step 2 Bull Labels CSV
# ============================================================
def build_label(df: pd.DataFrame, price_col: str, label_file: str) -> pd.Series:
    """
    y(t) = Regime(t + FORECAST_HORIZON)
    Regime is pulled from STEP 2 output (market_trend_regime).
    """
    print(f"  > Loading regime labels from: {os.path.basename(label_file)}")
    df_labels = pd.read_csv(label_file)
    df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])
    df_labels.set_index('timestamp', inplace=True)
    
    # Align labels with the current dataframe index
    regime = df_labels['market_trend_regime'].reindex(df.index).ffill().fillna(0)
    
    # Forward shift: if horizon > 0, we want to know what the regime will be N days later
    y = regime.shift(-FORECAST_HORIZON)
    
    return regime, y


# ============================================================
# STEP B — BUILD FEATURE MATRIX X
# ============================================================
PILLAR_COLS = ["market_trend", "market_breadth", "market_cycle", "market_volatility"]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    X(t) = lagged (by 1 day) Z-scored pillars
    """
    feat = pd.DataFrame(index=df.index)
    for col in PILLAR_COLS:
        if col in df.columns:
            feat[col] = df[col].shift(1)          # lag 1 day to avoid lookahead
    return feat.dropna()


# ============================================================
# SIGNAL LOGIC (HYSTERESIS & MIN-HOLD)
# ============================================================
def apply_probability_hysteresis(probs: pd.Series, vol_series: pd.Series, 
                                 threshold: float = THRESHOLD, 
                                 base_buffer: float = HYSTERESIS_BUFFER, 
                                 min_hold: int = MIN_HOLD_DAYS) -> pd.Series:
    """
    Reduces signal noise by requiring a buffer and a minimum hold time.
    Logic:
      ON  if prob > (threshold + buff)
      OFF if prob < (threshold - buff)
      ELSE keep last state
    """
    n = len(probs)
    states = np.zeros(n)
    last_state = 0
    days_held = 0
    
    # Scale buffer by relative volatility (higher vol = wider buffer)
    vol_mult = (vol_series / vol_series.rolling(252).mean()).fillna(1.0)
    
    for i in range(n):
        p = probs.iloc[i]
        buff = base_buffer * vol_mult.iloc[i]
        
        new_state = last_state
        if p > (threshold + buff):
            new_state = 1
        elif p < (threshold - buff):
            new_state = 0
            
        # Enforce minimum holding period
        if new_state != last_state:
            if days_held < min_hold:
                new_state = last_state
                days_held += 1
            else:
                days_held = 1
        else:
            days_held += 1
            
        last_state = new_state
        states[i] = last_state
        
    return pd.Series(states, index=probs.index)


def binarize(y: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Note: In testing, we use probability threshold (0.5), but for training labels we use raw value threshold (0.0)."""
    return (y > threshold).astype(int)


def plot_prediction_chart(df_raw: pd.DataFrame, 
                           X_eval: pd.DataFrame, 
                           y_pred_bin: pd.Series, 
                           y_pred_prob: pd.Series, 
                           model_name: str, 
                           output_path: str,
                           target_price_col: str,
                           horizon: int,
                           eval_start: str):
    """Generates the dual-axis market health prediction chart for Walk-Forward period."""
    # Build predicted signal series on the evaluation window
    pred_signal = pd.Series(y_pred_bin, index=X_eval.index, name="pred_signal")
    pred_prob   = pd.Series(y_pred_prob, index=X_eval.index, name="pred_prob")

    # Combine available prices and predicted signal for plot
    plot_start = eval_start
    df_plot = df_raw[[target_price_col, "composite_health"]].copy()
    df_plot = df_plot[df_plot.index >= plot_start]
    df_plot["pred_signal"] = np.nan
    df_plot.loc[pred_signal.index, "pred_signal"] = pred_signal.values

    # Forward-fill predicted signal for background coloring
    df_plot["pred_signal_ff"] = df_plot["pred_signal"].ffill().fillna(0)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 12),
        gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )

    # ── Top: Price with signal background ─────────────────────────────────────
    ax1.plot(df_plot.index, df_plot[target_price_col],
             color="#2d3436", linewidth=1.5, label=f"{target_price_col.upper()}", zorder=3)

    # Walk-Forward period shading
    eval_start_dt = pd.Timestamp(eval_start)
    ax1.axvspan(eval_start_dt, df_plot.index[-1],
                color="#dfe6e9", alpha=0.3, zorder=1, label="Walk-Forward Evaluation")

    # Predicted signal background (green/red)
    for i in range(len(df_plot) - 1):
        color = "#d4edda" if df_plot["pred_signal_ff"].iloc[i] == 1 else "#f8d7da"
        ax1.axvspan(df_plot.index[i], df_plot.index[i + 1],
                    color=color, alpha=0.55, linewidth=0, zorder=2)

    # Signal flip markers
    pred_signal_clean = df_plot["pred_signal"].dropna()
    flip = pred_signal_clean.diff()
    on_dates  = flip[flip == 1].index
    off_dates = flip[flip == -1].index
    vn_on  = df_raw[target_price_col].reindex(on_dates).dropna()
    vn_off = df_raw[target_price_col].reindex(off_dates).dropna()
    ax1.scatter(vn_on.index,  vn_on.values,  color="green", marker="^", s=70,
                label="Predicted Signal ON",  zorder=5)
    ax1.scatter(vn_off.index, vn_off.values, color="red",   marker="v", s=70,
                label="Predicted Signal OFF", zorder=5)

    ax1.set_title(
        f"Market Health — {model_name} Rolling Prediction (Horizon: {horizon}d)",
        fontsize=15, fontweight="bold"
    )
    ax1.set_ylabel(f"{target_price_col.upper()} Level")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.2)

    # ── Bottom: Raw Composite Health score & predicted probability ─────────────
    y_health = df_plot["composite_health"].values.astype(float)
    ax2.plot(df_plot.index, y_health, color="#0984e3", linewidth=1.3,
             label="Composite Health Score", zorder=3)
    ax2.fill_between(df_plot.index, 0, y_health,
                     where=(y_health > 0), color="green", alpha=0.08, interpolate=True)
    ax2.fill_between(df_plot.index, 0, y_health,
                     where=(y_health < 0), color="red",   alpha=0.08, interpolate=True)

    # Overlay predicted probability
    ax2_r = ax2.twinx()
    ax2_r.plot(pred_prob.index, pred_prob.values, color="#e17055", linewidth=1.0,
               alpha=0.7, linestyle="--", label="Predicted Prob (Bullish)")
    ax2_r.axhline(0.5, color="#e17055", linewidth=0.7, alpha=0.4, linestyle=":")
    ax2_r.set_ylabel("Predicted Probability", fontsize=9, color="#e17055")
    ax2_r.set_ylim(0, 1)
    ax2_r.tick_params(axis="y", labelcolor="#e17055")

    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5, linestyle="--")
    ax2.set_ylabel("Composite Health Score")
    ax2.legend(loc="upper left", fontsize=8)
    ax2_r.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved {os.path.basename(output_path)}")


# ============================================================
# HELPER: PRETTY PRINTING
# ============================================================
def print_section(title: str):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # ── 0. Load Step4 output ───────────────────────────────────────────────────
    print("Loading Step 4 final output...")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Run Step 4 first.")
        exit(1)

    df_raw = pd.read_csv(INPUT_FILE, index_col=0)
    df_raw.index = pd.to_datetime(df_raw.index)
    print(f"  Loaded {len(df_raw):,} rows from {df_raw.index[0].date()} to {df_raw.index[-1].date()}")

    # ── 1. Build Label ─────────────────────────────────────────────────────────
    # ── 1. Build Label ─────────────────────────────────────────────────────────
    print_section("STEP A: LOADING & REGIME LABELING")
    
    # Choose VNINDEX as the price reference for PE and Regime
    price_col = "vnindex_close"
    if price_col not in df_raw.columns:
        price_col = "vn100_ew_close"
    print(f"  > Using '{price_col}' as reference for Regime Detection (TF).")

    raw_label, y_all = build_label(df_raw, price_col, LABEL_FILE)

    # ── 2. Build Features ──────────────────────────────────────────────────────
    print_section("STEP B: FEATURE ENGINEERING (Lagged 1 Day)")
    X_all = build_features(df_raw)

    # ── 3. Align X and y on common index ──────────────────────────────────────
    common_idx = X_all.index.intersection(y_all.dropna().index)
    X_all = X_all.loc[common_idx]
    y_all = y_all.loc[common_idx]
    # For training labels, the regime is already binary (0 or 1)
    y_binary = y_all.dropna().astype(int)

    # ── 4. OUTPUT 1: Pillar + PE + Volatility CSV & PNG ───────────────────────
    feature_cols = X_all.columns.tolist()
    print(f"  > Features identified: {feature_cols}")
    print(f"  > Total synchronized samples: {len(X_all):,}")
    
    print("\n  [Saving Inputs...]")
    all_features_df = X_all.copy()
    all_features_df.to_csv(os.path.join(OUTPUT_DIR, "step5_all_features.csv"))
    print(f"  Saved step5_all_features.csv with {len(all_features_df.columns)} features.")

    # Plot features overview
    fig_cols = feature_cols
    n_cols_plot = len(fig_cols)
    fig, axes = plt.subplots(n_cols_plot, 1, figsize=(15, 3 * n_cols_plot), sharex=True)
    if n_cols_plot == 1:
        axes = [axes]
    colors = ["#0984e3", "#00b894", "#6c5ce7", "#e17055", "#fdcb6e", "#fd79a8"]
    for i, col in enumerate(fig_cols):
        ax = axes[i]
        ax.plot(all_features_df.index, all_features_df[col], linewidth=1.0,
                color=colors[i % len(colors)], label=col)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4, linestyle="--")
        ax.set_ylabel(col, fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)
    axes[0].set_title("Step 5: Input Features (Lagged)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step5_features_overview.png"), dpi=180)
    plt.close()
    print("  > step5_features_overview.png [SAVED]")

    # ── 5. Walk-Forward Rolling Training ──────────────────────────────────────
    print_section("STEP C: WALK-FORWARD ROLLING PREDICTION")
    print(f"  Starting Walk-Forward simulation from {WALK_FORWARD_START}")
    print(f"  Training Window: {WINDOW_YEARS} years | Refit Frequency: {REFIT_FREQ} days")

    wf_indices = X_all.index[X_all.index >= WALK_FORWARD_START]
    
    # Storage for OOS predictions
    oos_probs_lr  = pd.Series(index=wf_indices, dtype=float)
    oos_probs_xgb = pd.Series(index=wf_indices, dtype=float)
    oos_probs_lgb = pd.Series(index=wf_indices, dtype=float)

    # Scaler
    scaler = StandardScaler()

    # Loop with Refit Frequency
    for i in range(0, len(wf_indices), REFIT_FREQ):
        current_date = wf_indices[i]
        
        # Define Training window: [current_date - 5 years, current_date - 1 day]
        train_start = current_date - pd.DateOffset(years=WINDOW_YEARS)
        train_end   = current_date - pd.Timedelta(days=1)
        
        X_train_wf = X_all.loc[train_start:train_end]
        y_train_wf = y_binary.loc[X_train_wf.index]
        
        if len(X_train_wf) < 252: # Minimum 1 year of data to train
            print(f"  ! Skipping {current_date.date()} — insufficient training data ({len(X_train_wf)} rows)")
            continue

        # Segment to predict: [current_date, next refit date]
        pred_end_idx = min(i + REFIT_FREQ, len(wf_indices))
        pred_indices = wf_indices[i:pred_end_idx]
        X_test_wf    = X_all.loc[pred_indices]

        # Standard Scaling
        X_tr_sc = scaler.fit_transform(X_train_wf)
        X_te_sc = scaler.transform(X_test_wf)

        # 1. Logistic Regression
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(X_tr_sc, y_train_wf)
        oos_probs_lr.loc[pred_indices] = lr.predict_proba(X_te_sc)[:, 1]

        # 2. XGBoost
        # scale_pos_weight = count_neg / count_pos
        xgb_weight = (y_train_wf == 0).sum() / (y_train_wf == 1).sum() if (y_train_wf == 1).sum() > 0 else 1.0
        model_xgb = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            objective="binary:logistic", random_state=42, eval_metric="logloss",
            scale_pos_weight=xgb_weight, n_jobs=-1
        )
        model_xgb.fit(X_tr_sc, y_train_wf)
        oos_probs_xgb.loc[pred_indices] = model_xgb.predict_proba(X_te_sc)[:, 1]

        # 3. LightGBM
        model_lgb = lgb.LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=42, verbosity=-1, class_weight="balanced", n_jobs=-1
        )
        model_lgb.fit(X_tr_sc, y_train_wf)
        oos_probs_lgb.loc[pred_indices] = model_lgb.predict_proba(X_te_sc)[:, 1]

        if i % (REFIT_FREQ * 5) == 0:
            print(f"  > Refitted at {current_date.date()} (Train window: {X_train_wf.index[0].date()} to {X_train_wf.index[-1].date()})")

    # Drop any NaNs (from skips)
    oos_probs_lr.dropna(inplace=True)
    oos_probs_xgb.dropna(inplace=True)
    oos_probs_lgb.dropna(inplace=True)
    
    # Align common indices
    eval_idx = oos_probs_lr.index
    y_eval_bin = y_binary.loc[eval_idx]
    X_eval = X_all.loc[eval_idx]

    # ── 6. OUTPUT 1: Walk-Forward Data Export ─────────────────────────────────
    print("\n  [Exporting Evaluation Sets...]")
    eval_df = X_eval.copy()
    eval_df["y_binary"] = y_eval_bin
    eval_df["prob_lr"]  = oos_probs_lr
    eval_df["prob_xgb"] = oos_probs_xgb
    eval_df["prob_lgb"] = oos_probs_lgb
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "step5_walk_forward_oos_data.csv"))
    print(f"  Saved step5_walk_forward_oos_data.csv ({len(eval_df):,} rows)")

    # ── 7.5. Apply Hysteresis Filter (Noise Reduction) ────────────────────────
    print(f"\n  [Post-Processing: Noise Suppression]")
    print(f"  > Applying Hysteresis (Buffer: {HYSTERESIS_BUFFER}, Min Hold: {MIN_HOLD_DAYS}d)")
    
    # Get volatility series for the test window to drive hysteresis
    vol_eval = df_raw.loc[eval_idx, "volatility"]

    y_pred_lr_bin  = apply_probability_hysteresis(oos_probs_lr, vol_eval)
    y_pred_xgb_bin = apply_probability_hysteresis(oos_probs_xgb, vol_eval)
    y_pred_lgb_bin = apply_probability_hysteresis(oos_probs_lgb, vol_eval)

    # ── 8. OUTPUT 3: Evaluation & Comparison ──────────────────────────────────
    print_section("OUTPUT 3: SCOREBOARD & MODEL GRADING")

    def get_metrics(y_true, y_pred, model_name):
        return {
            "model": model_name,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, zero_division=0)
        }

    metrics_lr  = get_metrics(y_eval_bin, y_pred_lr_bin,  "Logistic Regression (Rolling)")
    metrics_xgb = get_metrics(y_eval_bin, y_pred_xgb_bin, "XGBoost (Rolling)")
    metrics_lgb = get_metrics(y_eval_bin, y_pred_lgb_bin, "LightGBM (Rolling)")

    results_df = pd.DataFrame([metrics_lr, metrics_xgb, metrics_lgb])
    results_df["eval_start"] = WALK_FORWARD_START
    results_df["window_years"] = WINDOW_YEARS
    results_df["forecast_horizon_days"] = FORECAST_HORIZON
    results_df["hysteresis_buffer"]     = HYSTERESIS_BUFFER
    results_df["min_hold_days"]         = MIN_HOLD_DAYS

    print("\n  [Comparative Metrics]")
    pd.options.display.float_format = '{:,.4f}'.format
    print(results_df[["model", "precision", "recall", "accuracy", "f1_score"]].to_string(index=False))

    print(f"\n  [Logistic Regression Diagnostic Report]")
    print("  " + classification_report(y_eval_bin, y_pred_lr_bin, zero_division=0).replace("\n", "\n  "))

    print(f"\n  [XGBoost Diagnostic Report]")
    print("  " + classification_report(y_eval_bin, y_pred_xgb_bin, zero_division=0).replace("\n", "\n  "))
    
    print(f"\n  [LightGBM Diagnostic Report]")
    print("  " + classification_report(y_eval_bin, y_pred_lgb_bin, zero_division=0).replace("\n", "\n  "))

    results_df.to_csv(os.path.join(OUTPUT_DIR, "step5_evaluation_metrics_comparison.csv"), index=False)
    print("  Saved step5_evaluation_metrics_comparison.csv")

    # ── 9. OUTPUT 4: Charts (Logistic vs XGBoost) ────────────────────────────
    print_section("OUTPUT 4: VISUALIZATION & DIAGNOSTICS")
    
    target_price_col = "vn100_ew_close" if "vn100_ew_close" in df_raw.columns else "vnindex_close"

    # A. Baseline Chart (Logistic Regression)
    plot_prediction_chart(
        df_raw, X_eval, y_pred_lr_bin, oos_probs_lr, 
        model_name="Logistic Regression", 
        output_path=os.path.join(OUTPUT_DIR, "step5_market_health_logistic_prediction_chart.png"),
        target_price_col=target_price_col, 
        horizon=FORECAST_HORIZON,
        eval_start=WALK_FORWARD_START
    )

    # B. Challenger 1 Chart (XGBoost)
    plot_prediction_chart(
        df_raw, X_eval, y_pred_xgb_bin, oos_probs_xgb, 
        model_name="XGBoost", 
        output_path=os.path.join(OUTPUT_DIR, "step5_market_health_xgb_prediction_chart.png"),
        target_price_col=target_price_col, 
        horizon=FORECAST_HORIZON,
        eval_start=WALK_FORWARD_START
    )

    # C. Challenger 2 Chart (LightGBM)
    plot_prediction_chart(
        df_raw, X_eval, y_pred_lgb_bin, oos_probs_lgb, 
        model_name="LightGBM", 
        output_path=os.path.join(OUTPUT_DIR, "step5_market_health_lgb_prediction_chart.png"),
        target_price_col=target_price_col, 
        horizon=FORECAST_HORIZON,
        eval_start=WALK_FORWARD_START
    )

    # Secondary copy for standard reference
    import shutil
    shutil.copyfile(
        os.path.join(OUTPUT_DIR, "step5_market_health_xgb_prediction_chart.png"),
        os.path.join(OUTPUT_DIR, "step5_market_health_prediction_chart.png")
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print_section("PIPELINE SUMMARY")
    
    # identify Champion based on F1
    best_idx = results_df['f1_score'].idxmax()
    best_model_row = results_df.iloc[best_idx]
    
    print(f"  CHAMPION MODEL : {best_model_row['model'].upper()}")
    print(f"  TEST ACCURACY  : {best_model_row['accuracy']*100:.2f}%")
    print(f"  F1-SCORE       : {best_model_row['f1_score']:.4f}")
    
    print(f"\n  Files generated in: {OUTPUT_DIR}")
    print(f"  1. step5_all_features.csv          — input features over full history")
    print(f"  2. step5_features_overview.png     — feature time-series plot")
    print(f"  3. step5_walk_forward_oos_data.csv — rolling results with actual labels")
    print(f"  5. step5_evaluation_metrics_comparison.csv       — comparative analysis")
    print(f"  6. step5_market_health_logistic_prediction_chart.png — Base model chart")
    print(f"  7. step5_market_health_xgb_prediction_chart.png   — XGBoost chart")
    print(f"  8. step5_market_health_lgb_prediction_chart.png   — LightGBM chart")
    print(f"{'='*60}")
