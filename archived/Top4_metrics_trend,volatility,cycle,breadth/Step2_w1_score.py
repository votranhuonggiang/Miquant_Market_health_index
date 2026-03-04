import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.stats import wasserstein_distance
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG & HELPERS (Borrowed from pipeline)
# ============================================================
DAILY_LAMBDA = 5

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

def load_vnindex():
    return query_questdb(
        "SELECT * FROM raw_eod WHERE symbol = 'VNINDEX' AND timestamp >= '2016-01-01'"
    )

# ============================================================
# BULL LABELING LOGIC (L1 Trend Filtering)
# ============================================================
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
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return detect_regime(df["close"], lambda_value)

# ============================================================
# W1 SCORING LOGIC
# ============================================================
def compute_wasserstein_scores(feat_df, labels):
    if feat_df.empty or labels.empty:
        return pd.DataFrame(columns=["feature", "w1_score", "w1_normalized", "status", "std"])
    
    idx = feat_df.index.intersection(labels.index)
    if len(idx) == 0:
        return pd.DataFrame(columns=["feature", "w1_score", "w1_normalized", "status", "std"])
        
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
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    
    # Paths
    base_dir = os.path.dirname(__file__) # application/
    output_dir = os.path.join(base_dir, "step2_w1_score")
    os.makedirs(output_dir, exist_ok=True)
    
    input_features_path = os.path.join(base_dir, "step1_features", "step1_all_features.csv")
    
    if not os.path.exists(input_features_path):
        # Try one level up if not found (in case script is run from a subfolder)
        input_features_path = os.path.join(os.path.dirname(base_dir), "step1_features", "step1_all_features.csv")
        if not os.path.exists(input_features_path):
            print(f"Error: Could not find features. Tried:\n1. {os.path.join(base_dir, 'step1_features', 'step1_all_features.csv')}\n2. {input_features_path}")
            exit()
        
    print(f"Loading features from {input_features_path}...")
    feat_df = pd.read_csv(input_features_path, index_col=0)
    feat_df.index = pd.to_datetime(feat_df.index).tz_localize(None)
    
    print("Fetching VNINDEX for Bull/Bear labeling...")
    vnidx = load_vnindex()
    if vnidx.empty:
        print("Error: Could not fetch VNINDEX data.")
        exit()
        
    print("Defining Bull Base Labels...")
    labels = define_bull_base_labels(vnidx, lambda_value=DAILY_LAMBDA)
    labels.name = "market_trend_regime"
    
    # Save labels and VNINDEX close for reference
    df_vn = vnidx.copy().set_index("timestamp")
    df_vn.index = pd.to_datetime(df_vn.index).tz_localize(None)
    df_vn = df_vn.join(labels, how="inner")
    df_vn[["market_trend_regime", "close"]].rename(columns={"close":"vnindex_close"}).to_csv(
        os.path.join(output_dir, "step2_bull_labels.csv")
    )
    
    print("Computing W1 Scores for 40 features...")
    common = feat_df.index.intersection(labels.index)
    scores_df = compute_wasserstein_scores(feat_df.loc[common], labels.loc[common])
    
    print("Deduplicating features by correlation...")
    scores_df, kept_features = deduplicate_by_correlation(scores_df, feat_df.loc[common])
    
    # Save the final scores
    scores_df.to_csv(os.path.join(output_dir, "step2_w1_scores_all_features.csv"), index=False)
    print(f"Saved W1 scores to {os.path.join(output_dir, 'step2_w1_scores_all_features.csv')}")
    
    # ------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------
    
    # ------------------------------------------------------------
    # VISUALIZATIONS (Premium Design)
    # ------------------------------------------------------------
    plt.style.use('ggplot') # Use a cleaner base style
    
    # 1. Bull Regime Chart
    print("Generating premium Bull Regime chart...")
    plot_df = df_vn[df_vn.index >= "2020-01-01"] # Show more history
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot price line with a sleek signature color
        ax.plot(plot_df.index, plot_df["close"], color="#2c3e50", linewidth=2, label="VNINDEX Close", zorder=3)
        
        # Fill regimes with sophisticated transparency
        for i in range(len(plot_df)-1):
            # Bullish = soft emerald, Bearish = soft rose
            color = "#e8f5e9" if plot_df["market_trend_regime"].iloc[i] == 1 else "#ffebee"
            ax.axvspan(plot_df.index[i], plot_df.index[i+1], color=color, alpha=0.9, linewidth=0)
        
        # Signal arrows for regime flips
        flip = plot_df["market_trend_regime"].diff()
        bull_signals = plot_df[flip == 1]
        bear_signals = plot_df[flip == -1]
        
        ax.scatter(bull_signals.index, bull_signals["close"], color="#27ae60", marker="^", s=100, 
                   label="Bull Regime Start", edgecolors='white', linewidths=1.5, zorder=5)
        ax.scatter(bear_signals.index, bear_signals["close"], color="#e74c3c", marker="v", s=100, 
                   label="Bear Regime Start", edgecolors='white', linewidths=1.5, zorder=5)
        
        # Refined titles and labels
        ax.set_title("VNINDEX Trend Regimes: L1 Trend Filtering", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Date", fontsize=12, labelpad=10)
        ax.set_ylabel("Price Index Level", fontsize=12, labelpad=10)
        
        # Custom legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            plt.Line2D([0], [0], color='#2c3e50', lw=2, label='VNINDEX Close'),
            Patch(facecolor='#e8f5e9', label='Bull Regime Zone'),
            Patch(facecolor='#ffebee', label='Bear Regime Zone'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#27ae60', markersize=12, label='Bull Trigger'),
            plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#e74c3c', markersize=12, label='Bear Trigger')
        ]
        ax.legend(handles=legend_elements, loc='upper left', frameon=True, shadow=True, facecolor='white')
        
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "step2_bull_regime_chart.png"), dpi=200)
        plt.close()
    
    # 2. W1 Bar Chart
    print("Generating refined W1 Bar Chart...")
    # Only plot Kept and Dropped (exclude Skipped to focus on candidates)
    plot_s = scores_df[scores_df["status"] != "Skipped"].sort_values("w1_score", ascending=True).tail(25)
    
    plt.figure(figsize=(12, 12))
    colors = ["#27ae60" if s=="Kept" else "#e74c3c" for s in plot_s["status"]]
    
    bars = plt.barh(plot_s["feature"], plot_s["w1_score"], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add labels on the ends of the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 va='center', fontsize=9, fontweight='bold', color='#34495e')
        
    plt.title("Top Feature Information Content (Wasserstein W1 Score)", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Normalized W1 Score", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Legend for Keep/Drop
    from matplotlib.lines import Line2D
    legend_custom = [Line2D([0], [0], color='#27ae60', lw=8, label='Selected (Low Redundancy)'),
                     Line2D([0], [0], color='#e74c3c', lw=8, label='Dropped (High Correlation)')]
    plt.legend(handles=legend_custom, loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step2_w1_barchart.png"), dpi=200)
    plt.close()
    
    # 3. Distribution Comparison (Premium Grid)
    print("Generating high-fidelity distribution comparisons...")
    # Plot top 12 kept features for clarity
    kept_scores = scores_df[scores_df["status"] == "Kept"].sort_values("w1_score", ascending=False).head(12)
    cols_to_plot = kept_scores["feature"].tolist()
    
    if cols_to_plot:
        n_features = len(cols_to_plot)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()
        
        import seaborn as sns
        
        for i, f in enumerate(cols_to_plot):
            # Extract data
            common_idx = feat_df.index.intersection(labels.index)
            bull_data = feat_df.loc[common_idx].loc[labels.loc[common_idx]==1, f].dropna()
            bear_data = feat_df.loc[common_idx].loc[labels.loc[common_idx]==0, f].dropna()
            
            # Plot KDEs/Histograms
            sns.histplot(bull_data, ax=axes[i], color="#27ae60", label='Bull Regime', kde=True, stat="density", alpha=0.4, linewidth=0)
            sns.histplot(bear_data, ax=axes[i], color="#e74c3c", label='Bear Regime', kde=True, stat="density", alpha=0.4, linewidth=0)
            
            # Cosmetics
            axes[i].set_title(f"Feature: {f}\n(W1: {kept_scores.iloc[i]['w1_score']:.4f})", fontsize=12, fontweight='bold')
            axes[i].legend(fontsize=9, loc='upper right')
            axes[i].set_xlabel("Z-Score", fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
        # Clean up empty subplots
        for j in range(len(cols_to_plot), len(axes)): 
            fig.delaxes(axes[j])
            
        plt.suptitle("Statistical Divergence: Feature Distribution across Market Regimes", fontsize=20, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "step2_distribution_comparison.png"), bbox_inches='tight', dpi=180)
        plt.close()
    
    print(f"Step 2 Visualizations updated with premium aesthetics. Output saved to {output_dir}")
    
    print(f"Step 2 Complete. Output saved to {output_dir}")
