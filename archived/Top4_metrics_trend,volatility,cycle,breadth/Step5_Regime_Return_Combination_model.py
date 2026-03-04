import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from typing import Union, Optional, Any
import warnings
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# Define Types commonly used in quantitative modules
PD_TYPE = Union[pd.DataFrame, pd.Series]
DATE_TYPE = Union[str, Any]

def check_dir_exist(filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def filter_date_range(df: PD_TYPE, start_date: DATE_TYPE = None, end_date: DATE_TYPE = None) -> PD_TYPE:
    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df

def raise_labels_into_proba(labels: np.ndarray, n_c: int = 2) -> np.ndarray:
    out = np.zeros((len(labels), n_c))
    for i, l in enumerate(labels):
        if 0 <= int(l) < n_c:
            out[i, int(l)] = 1.0
    return out

# ============================================================
# 1. USER-PROVIDED PLOTTING MODULE
# ============================================================
ALPHA_LINE = .8
ALPHA_FILL = .3
AXES_TYPE = Optional[plt.Axes]

def matplotlib_setting():
    plt.rcParams['figure.figsize'] = (24, 12)
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams['font.size'] = 22
    try:
        plt.rcParams['font.family'] = 'cmr10'
        plt.rcParams['axes.formatter.use_mathtext'] = True
    except: pass
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

matplotlib_setting()

def convert_yaxis_to_percent(ax: plt.Axes) -> None:
    def to_percent(x, position): 
        return f"{x * 100:.1f}%"
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

def plot_cumret(ret_df: Union[PD_TYPE, dict], 
                start_date: DATE_TYPE = None, 
                end_date: DATE_TYPE = None, 
                ax: AXES_TYPE = None, 
                ylabel_ret="Cumulative Returns",
                ) -> plt.Axes:
    if ax is None: _, ax = plt.subplots(figsize=(24, 12))
    df = pd.DataFrame(ret_df)
    df.index = pd.to_datetime(df.index)
    df = filter_date_range(df, start_date, end_date)
    df.cumsum(axis=0).plot(ax=ax, linewidth=3)
    ax.set_ylabel(ylabel_ret)
    convert_yaxis_to_percent(ax)
    return ax

def fill_between(ser: pd.Series,
                 start_date: DATE_TYPE = None, 
                 end_date: DATE_TYPE = None, 
                 ax: AXES_TYPE = None, 
                 color: Optional[str] = None, 
                 fill_between_label: Optional[str] = None) -> plt.Axes:
    if ax is None: _, ax = plt.subplots(figsize=(24, 4))
    ser = filter_date_range(ser, start_date, end_date)
    ax.fill_between(ser.index, ser, step="pre", alpha=0.3, color=color, label=fill_between_label)
    return ax

def plot_regimes(regimes: PD_TYPE, 
                 n_c: int = 2, 
                 start_date: DATE_TYPE = None, 
                 end_date: DATE_TYPE = None, 
                 ax: AXES_TYPE = None, 
                 colors_regimes: Optional[list] = ['g', 'r'], 
                 labels_regimes: Optional[list] = ['Bull', 'Bear'],
                 ) -> plt.Axes:
    df = filter_date_range(pd.DataFrame(regimes), start_date, end_date)
    if df.shape[1] == 1:
        probs = raise_labels_into_proba(df.values.flatten(), n_c)
        df = pd.DataFrame(probs, index=df.index)
    if ax is None: _, ax = plt.subplots(figsize=(24, 4))
    for i in range(n_c):
        fill_between(df.iloc[:, i], ax=ax, color=colors_regimes[i], fill_between_label=labels_regimes[i])
    return ax

def plot_regimes_and_cumret(regimes: PD_TYPE, 
                            ret_df: Union[PD_TYPE, dict], 
                            n_c: int = 2, 
                            start_date: DATE_TYPE = None, 
                            end_date: DATE_TYPE = None, 
                            ax: AXES_TYPE = None, 
                            colors_regimes: Optional[list] = ['g', 'r'], 
                            labels_regimes: Optional[list] = ['Bull', 'Bear'],
                            ylabel_ret="Cumulative Returns",
                            legend_loc="upper left"
                            ) -> tuple[plt.Axes, plt.Axes]:
    ax = plot_cumret(ret_df, start_date=start_date, end_date=end_date, ax=ax, ylabel_ret=ylabel_ret)
    ax2 = ax.twinx()
    # Hiding the probability axis labels/ticks as per user request
    ax2.get_yaxis().set_visible(False)
    ax2.set_ylim(0, 1.1)
    plot_regimes(regimes, n_c, start_date=start_date, end_date=end_date, ax=ax2, colors_regimes=colors_regimes, labels_regimes=labels_regimes)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc)
    return (ax, ax2)

# ============================================================
# 2. JOINT REGIME-SWITCHING MODEL (LOGISTIC REGRESSION)
# ============================================================
class JointJumpRegimeModel:
    """
    Implements a Joint Jump Model for Regime-Switching.
    The model jointly optimizes regime feature centers and return distributions.
    
    Objective:
    min sum [ 0.5*||x_t - c_s_t||^2 + beta * ( (r_t - mu_s_t)^2 / (2*sigma_s_t^2) + log(sigma_s_t) ) ] + sum [ lambda * (s_t != s_t-1) ]
    """
    def __init__(self, n_regimes=2, beta=1.0, lambda_jump=1.0, max_iter=50, tol=1e-6):
        self.K = n_regimes
        self.beta = beta
        self.lambda_jump = lambda_jump
        self.max_iter = max_iter
        self.tol = tol
        
        # Parameters to be learned
        self.centroids = None # c_k
        self.mu = None        # mu_k
        self.sigma = None     # sigma_k
        self.regimes = None
        self.total_objective = None
        self.tm = None

    def _initialize_params(self, X, r):
        T, N = X.shape
        # Initialize regimes using KMeans on returns or features
        from sklearn.cluster import KMeans
        # Use returns for a better initial regime guess for market context
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init=10).fit(r.reshape(-1, 1))
        s = kmeans.labels_
        self.update_parameters(X, r, s)
        return s

    def compute_loss_matrix(self, X, r):
        T, N = X.shape
        L = np.zeros((T, self.K))
        for k in range(self.K):
            # Feature distance component: 0.5 * ||x_t - c_k||^2
            dist_feat = 0.5 * np.sum((X - self.centroids[k])**2, axis=1)
            
            # Return likelihood component: beta * ( (r_t - mu_k)^2 / (2*sigma_k^2) + log(sigma_k) )
            # Add small epsilon to sigma for stability
            sigma_k = max(self.sigma[k], 1e-6)
            dist_ret = self.beta * (((r - self.mu[k])**2 / (2 * sigma_k**2)) + np.log(sigma_k))
            
            L[:, k] = dist_feat + dist_ret
        return L

    def dynamic_programming(self, L):
        T, K = L.shape
        D = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        
        D[0] = L[0]
        for t in range(1, T):
            for k in range(K):
                # Transition costs from all regimes j at t-1 to regime k at t
                costs = D[t-1] + self.lambda_jump
                costs[k] -= self.lambda_jump # No penalty if staying in the same regime
                
                best_prev = np.argmin(costs)
                D[t, k] = L[t, k] + costs[best_prev]
                psi[t, k] = best_prev
        
        # Backtrack to find optimal sequence
        s = np.zeros(T, dtype=int)
        s[T-1] = np.argmin(D[T-1])
        for t in range(T-2, -1, -1):
            s[t] = psi[t+1, s[t+1]]
            
        return s, D[T-1, s[T-1]]

    def update_parameters(self, X, r, s):
        T, N = X.shape
        self.centroids = np.zeros((self.K, N))
        self.mu = np.zeros(self.K)
        self.sigma = np.zeros(self.K)
        
        for k in range(self.K):
            mask = (s == k)
            if mask.any():
                self.centroids[k] = np.mean(X[mask], axis=0)
                self.mu[k] = np.mean(r[mask])
                # Small floor for sigma^2 to ensure stability
                var_k = np.var(r[mask])
                self.sigma[k] = np.sqrt(max(var_k, 1e-9))
            else:
                # Handle empty regimes by assigning global means or random data
                self.centroids[k] = np.mean(X, axis=0)
                self.mu[k] = np.mean(r)
                self.sigma[k] = np.std(r)

    def fit(self, X, r):
        s = self._initialize_params(X, r)
        prev_obj = float('inf')
        
        for i in range(self.max_iter):
            # Step 1: Compute Loss Matrix
            L = self.compute_loss_matrix(X, r)
            
            # Step 2: Regime Assignment (DP)
            s, total_obj = self.dynamic_programming(L)
            
            # Step 3: Parameter Update
            self.update_parameters(X, r, s)
            
            # Check convergence
            if abs(prev_obj - total_obj) < self.tol:
                break
            prev_obj = total_obj
            
        self.regimes = s
        self.total_objective = total_obj
        
        # Calculate empirical transition matrix
        self.tm = np.zeros((self.K, self.K))
        for t in range(1, len(s)):
            self.tm[s[t-1], s[t]] += 1
        row_sums = self.tm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.tm = self.tm / row_sums
        
        return self

    def predict_probs(self, X, r):
        """
        Calculates regime probabilities based on Joint Likelihood.
        Since this is a Jump Model with DP, the 'probability' is usually 
        binary in assignment, but we can provide a softmax-style likelihood 
        score for visualization if needed.
        """
        L = self.compute_loss_matrix(X, r)
        # Convert loss (negative log-likelihood) to probability
        # Use the negative loss as the log-likelihood
        probs = np.exp(-L + np.max(-L, axis=1, keepdims=True))
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

# ============================================================
# 3. MAIN WORKFLOW
# ============================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "step4_market_health_index", "step4_market_health_final.csv")
    out_dir = os.path.join(base_dir, "step5_regime_return_combination_model")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(input_file):
        print(f"Error: Missing {input_file}")
        return

    # 1. Loading & Prep
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('date').set_index('date')
    
    # Filter for timestamp from 2020 until now as requested
    df = df[df.index >= "2020-01-01"]
    
    # Use log returns for VN100 EW
    if 'vn100_ew_close' in df.columns:
        df['market_return'] = np.log(df['vn100_ew_close'] / df['vn100_ew_close'].shift(1))
        df['price_for_regime'] = df['vn100_ew_close']
    else:
        potential_cols = [c for c in df.columns if 'ew' in c.lower() and 'close' in c.lower()]
        if potential_cols:
            print(f"Using {potential_cols[0]} for VN100 EW returns")
            df['market_return'] = np.log(df[potential_cols[0]] / df[potential_cols[0]].shift(1))
            df['price_for_regime'] = df[potential_cols[0]]
        else:
            print("Warning: vn100_ew_close not found. Using vnindex_close as fallback.")
            df['market_return'] = np.log(df['vnindex_close'] / df['vnindex_close'].shift(1))
            df['price_for_regime'] = df['vnindex_close']
            
    df = df.dropna(subset=['market_return'])

    cols = ['market_trend', 'market_volatility', 'market_cycle', 'market_breadth']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    r = df['market_return'].values

    # 2. Training (Joint Jump Model)
    print("Training Joint Feature-Return Regime Model...")
    # beta=10 balances feature distance and return likelihood
    # lambda_jump=0.5 penalizes regime switches
    model = JointJumpRegimeModel(n_regimes=2, beta=10.0, lambda_jump=50).fit(X, r)

    # Label Bull vs Bear based on regime means
    bull_id = np.argmax(model.mu)
    bear_id = 1 - bull_id
    
    # 3. Probabilities & Forecast
    probs = model.predict_probs(X, r)
    df['regime_label'] = ["Bull Market" if reg == bull_id else "Bear Market" for reg in model.regimes]
    df['bull_probability'] = probs[:, bull_id]
    df['bear_probability'] = probs[:, bear_id]
    
    # Expected return is the probability-weighted mean of each regime
    df['expected_return'] = df['bull_probability'] * model.mu[bull_id] + \
                            df['bear_probability'] * model.mu[bear_id]
    
    # Save Final CSV
    output_csv = os.path.join(out_dir, "step5_market_regime_forecast.csv")
    df[['price_for_regime', 'market_return', 'regime_label', 'bull_probability', 'bear_probability', 'expected_return']].to_csv(output_csv)
    print(f"Results saved to {output_csv}")

    # 4. Visualization
    print("Generating Visualizations...")
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_regimes_and_cumret(
        regimes=pd.Series(model.regimes, index=df.index),
        ret_df=df[['market_return']],
        n_c=2,
        start_date="2020-01-01",
        ax=ax,
        colors_regimes=['green', 'red'] if bull_id == 0 else ['red', 'green'],
        labels_regimes=["Bull", "Bear"] if bull_id == 0 else ["Bear", "Bull"],
        ylabel_ret="VN100 EW Cumulative Return (Log)"
    )
    plt.title(f"Joint Market Regime Identification (Beta={model.beta}, Lambda={model.lambda_jump})")
    plt.savefig(os.path.join(out_dir, "regime_performance_combined.png"))
    plt.close()

    # Transition Matrix Visualization
    tm = model.tm
    tm_ordered = np.array([[tm[bull_id, bull_id], tm[bull_id, bear_id]],
                           [tm[bear_id, bull_id], tm[bear_id, bear_id]]])
    
    tm_df = pd.DataFrame(tm_ordered, index=['From Bull', 'From Bear'], columns=['To Bull', 'To Bear'])
    ax = tm_df.plot(kind='bar', figsize=(10, 7), color=['#2ecc71', '#e74c3c'], width=0.8)
    plt.title("Regime Transition Probabilities", fontsize=24)
    plt.ylabel("Probability", fontsize=18); plt.ylim(0, 1.1); plt.xticks(rotation=0); plt.grid(axis='y', linestyle='--', alpha=0.7)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=16, fontweight='bold')
    plt.legend(title="Target State", loc='upper right')
    plt.savefig(os.path.join(out_dir, "transition_matrix_bars.png"))
    plt.close()

    # 5. Report
    p_last = np.array([df['bull_probability'].iloc[-1], df['bear_probability'].iloc[-1]])
    p_next = p_last @ tm_ordered

    report = ["=== JOINT REGIME-SWITCHING MODEL REPORT (VN100 EW) ===\n"]
    report.append(f"Model Parameters: beta={model.beta}, lambda={model.lambda_jump}")
    report.append(f"Total Objective: {model.total_objective:.4f}\n")
    
    for name, rid in [("Bull Market", bull_id), ("Bear Market", bear_id)]:
        mu, sigma = model.mu[rid], model.sigma[rid]
        n_samples = np.sum(model.regimes == rid)
        report.append(f"{name}:")
        report.append(f"  Mean Log Return: {mu*100:.4f}%")
        report.append(f"  Volatility (Std): {sigma*100:.4f}%")
        report.append(f"  Sample Size:      {n_samples}")
        report.append(f"  Centroid L2 Norm: {np.linalg.norm(model.centroids[rid]):.4f}")
        report.append("")

    report.append("=== TRANSITION PROBABILITIES ===")
    report.append(f"P(Bull | Bull): {tm_ordered[0,0]:.4f}")
    report.append(f"P(Bear | Bull): {tm_ordered[0,1]:.4f}")
    report.append(f"P(Bear | Bear): {tm_ordered[1,1]:.4f}")
    report.append(f"P(Bull | Bear): {tm_ordered[1,0]:.4f}")
    report.append("")
    report.append("=== NEXT DAY REGIME FORECAST ===")
    report.append(f"Probability Bull: {p_next[0]:.4f}")
    report.append(f"Probability Bear: {p_next[1]:.4f}")
    
    with open(os.path.join(out_dir, "real_data_model_report.txt"), "w") as f:
        f.write("\n".join(report))
    
    print("\n".join(report))
    print(f"\nOptimization completed. Outputs saved in {out_dir}")

if __name__ == "__main__":
    main()
