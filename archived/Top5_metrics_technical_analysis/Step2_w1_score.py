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
    
    # 1. Bull Regime Chart
    print("Generating Bull Regime chart...")
    plot_df = df_vn[df_vn.index >= "2022-01-01"]
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(plot_df.index, plot_df["close"], color="#2c3e50", linewidth=1.5)
        for i in range(len(plot_df)-1):
            color = "#d4edda" if plot_df["market_trend_regime"].iloc[i] == 1 else "#f8d7da"
            ax.axvspan(plot_df.index[i], plot_df.index[i+1], color=color, alpha=0.4)
        
        flip = plot_df["market_trend_regime"].diff()
        ax.scatter(plot_df[flip==1].index, plot_df[flip==1]["close"], color="green", marker="^", s=60, label="Bull Signal")
        ax.scatter(plot_df[flip==-1].index, plot_df[flip==-1]["close"], color="red", marker="v", s=60, label="Bear Signal")
        ax.set_title("VNINDEX Bull/Bear Regime (L1 Trend Filtering)", fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "step2_bull_regime_chart.png"))
        plt.close()
    
    # 2. W1 Bar Chart
    print("Generating W1 Bar Chart for all features...")
    plot_s = scores_df.sort_values("w1_score", ascending=True)
    plt.figure(figsize=(12, 15)) # Increased height for 40 features
    colors = ["#27ae60" if s=="Kept" else ("#e74c3c" if s=="Dropped" else "#bdc3c7") for s in plot_s["status"]]
    plt.barh(plot_s["feature"], plot_s["w1_score"], color=colors)
    plt.title("Features by W1 Score (Kept/Dropped/Skipped)", fontsize=14)
    plt.xlabel("W1 Score (Wasserstein Distance)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "step2_w1_barchart.png"))
    plt.close()
    
    # 3. Distribution Comparison (All 40 Features - Sorted by W1 Score)
    print("Generating Distribution Comparisons for all features (sorted by W1)...")
    cols_to_plot = scores_df["feature"].tolist()
    if cols_to_plot:
        n_features = len(cols_to_plot)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        for i, f in enumerate(cols_to_plot):
            prior = feat_df[f].dropna()
            bull = feat_df.loc[common].loc[labels.loc[common]==1, f].dropna() if not common.empty else pd.Series()
            bear = feat_df.loc[common].loc[labels.loc[common]==0, f].dropna() if not common.empty else pd.Series()
            
            axes[i].hist(prior, bins=40, density=True, alpha=0.3, label='Full Sample', color='grey')
            if not bull.empty:
                axes[i].hist(bull, bins=40, density=True, alpha=0.5, label='Bull Regime', color='green')
            if not bear.empty:
                axes[i].hist(bear, bins=40, density=True, alpha=0.5, label='Bear Regime', color='red')
            
            axes[i].set_title(f"Dist: {f}", fontsize=10)
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.2)
        
        for j in range(len(cols_to_plot), len(axes)): fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "step2_distribution_comparison.png"))
        plt.close()
    
    print(f"Step 2 Complete. Output saved to {output_dir}")
