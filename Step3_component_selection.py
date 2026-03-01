import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# MAIN EXECUTION: STEP 3 - FEATURE SELECTION
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    
    # Paths
    base_dir = os.path.dirname(__file__) # application/
    output_dir = os.path.join(base_dir, "step3_component_selection")
    os.makedirs(output_dir, exist_ok=True)
    
    step2_dir = os.path.join(base_dir, "step2_w1_score")
    step1_dir = os.path.join(base_dir, "step1_features")
    
    scores_path = os.path.join(step2_dir, "step2_w1_scores_all_features.csv")
    features_path = os.path.join(step1_dir, "step1_all_features.csv")
    
    if not os.path.exists(scores_path):
        print(f"Error: Could not find scores at {scores_path}")
        exit()
    if not os.path.exists(features_path):
        print(f"Error: Could not find features at {features_path}")
        exit()
        
    print(f"Loading W1 scores from {scores_path}...")
    scores_df = pd.read_csv(scores_path)
    
    print(f"Loading feature data from {features_path}...")
    feat_df = pd.read_csv(features_path, index_col=0)
    feat_df.index = pd.to_datetime(feat_df.index)
    
    print("--- Selecting Top 5 Predictive Features ---")
    
    # Filter features that were 'Kept' (high W1 score and not highly correlated)
    kept_df = scores_df[scores_df["status"] == "Kept"]
    
    # Fallback if none kept (highly unlikely with 40 features)
    if kept_df.empty:
        print("  WARNING: No features marked as 'Kept'. Falling back to top 5 by raw score.")
        final_selection = scores_df.head(5).copy()
    else:
        # Take the top 5 from the deduplicated list
        final_selection = kept_df.head(5).copy()
        
    final_selection["rank"] = range(1, len(final_selection) + 1)
    
    # Save the selected features
    output_file = os.path.join(output_dir, "step3_selected_5_features.csv")
    final_selection[["rank", "feature", "w1_score", "w1_normalized", "std"]].to_csv(output_file, index=False)
    print(f"Saved selected features to {output_file}")
    
    # ------------------------------------------------------------
    # VISUALIZATION: Top 5 Logic Checks
    # ------------------------------------------------------------
    selected_list = final_selection["feature"].tolist()
    print(f"Final Selection: {selected_list}")
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(final_selection.iterrows()):
        feat_name = row["feature"]
        if feat_name in feat_df.columns:
            ax = axes[i]
            ax.plot(feat_df.index, feat_df[feat_name], color="#27ae60", linewidth=1)
            ax.set_title(f"Rank {row['rank']}: {feat_name}\nW1 Score: {row['w1_score']:.4f}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', alpha=0.5, linestyle='--')
            
    # Remove unused subplot
    for j in range(len(final_selection), len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "step3_selected_features_grid.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved selection grid plot to {chart_path}")
    
    print("\nSelection Summary:")
    for _, r in final_selection.iterrows():
        print(f" {r['rank']}. {r['feature']:20} | W1: {r['w1_score']:.4f}")
    
    print("\nStep 3 Complete.")
