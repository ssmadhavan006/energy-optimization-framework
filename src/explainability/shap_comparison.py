import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SHAPComparison")

def plot_feature_importance_comparison(
    roughness_importance: pd.DataFrame,
    time_importance: pd.DataFrame,
    energy_importance: pd.DataFrame,
    figsize=(14, 8)
) -> str:
    """
    Side-by-side grouped bar chart comparing relative feature importance
    across all three targets.
    """
    # Get union of top 8 features from each target
    top_r = roughness_importance.head(8)["feature_name"].tolist()
    top_t = time_importance.head(8)["feature_name"].tolist()
    top_e = energy_importance.head(8)["feature_name"].tolist()
    
    union_features = list(set(top_r + top_t + top_e))
    
    # Calculate normalization values
    r_max = roughness_importance["mean_abs_shap"].max()
    t_max = time_importance["mean_abs_shap"].max()
    e_max = energy_importance["mean_abs_shap"].max()
    
    plot_data = []
    for feat in union_features:
        r_row = roughness_importance[roughness_importance["feature_name"] == feat]
        t_row = time_importance[time_importance["feature_name"] == feat]
        e_row = energy_importance[energy_importance["feature_name"] == feat]
        
        r_val = r_row.iloc[0]["mean_abs_shap"] if not r_row.empty else 0.0
        t_val = t_row.iloc[0]["mean_abs_shap"] if not t_row.empty else 0.0
        e_val = e_row.iloc[0]["mean_abs_shap"] if not e_row.empty else 0.0
        
        r_norm = r_val / r_max if r_max > 0 else 0.0
        t_norm = t_val / t_max if t_max > 0 else 0.0
        e_norm = e_val / e_max if e_max > 0 else 0.0
        
        plot_data.append({"Feature": feat, "Target": "Surface Roughness", "Relative Importance": r_norm})
        plot_data.append({"Feature": feat, "Target": "Machining Time", "Relative Importance": t_norm})
        plot_data.append({"Feature": feat, "Target": "Energy (SEC)", "Relative Importance": e_norm})
        
    df_plot = pd.DataFrame(plot_data)
    
    # Sort features by average importance for better visual structure
    avg_imp = df_plot.groupby("Feature")["Relative Importance"].mean().sort_values(ascending=False)
    df_plot["Feature"] = pd.Categorical(df_plot["Feature"], categories=avg_imp.index, ordered=True)
    df_plot = df_plot.sort_values("Feature")
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(
        x="Feature",
        y="Relative Importance",
        hue="Target",
        data=df_plot,
        palette=["#34495e", "#e74c3c", "#2ecc71"],
        ax=ax
    )
    
    ax.set_title("Relative Feature Importance Across Prediction Targets", fontsize=14, fontweight='bold')
    ax.set_xlabel("Machining and Kinematic Features", fontsize=11)
    ax.set_ylabel("Normalized SHAP Importance (0 to 1)", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Prediction Target")
    plt.tight_layout()
    
    output_dir = Path("outputs/figures/shap")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "cross_target_importance.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved cross-target importance plot to {save_path}")
    return str(save_path)

def plot_conflict_analysis(
    roughness_importance: pd.DataFrame,
    energy_importance: pd.DataFrame,
    figsize=(10, 8)
) -> str:
    """
    Scatter plot revealing feature trade-offs between Energy SEC and Surface Roughness.
    """
    # Merge importance dataframes
    df_merged = pd.merge(
        energy_importance.rename(columns={"mean_abs_shap": "energy_shap", "rank": "energy_rank"}),
        roughness_importance.rename(columns={"mean_abs_shap": "roughness_shap", "rank": "roughness_rank"}),
        on="feature_name",
        how="outer"
    ).fillna(0.0)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    x = df_merged["energy_shap"]
    y = df_merged["roughness_shap"]
    
    ax.scatter(x, y, color="#8e44ad", s=80, alpha=0.8, edgecolors="none")
    
    # Annotate points with feature name
    for i, row in df_merged.iterrows():
        ax.annotate(
            row["feature_name"],
            (row["energy_shap"], row["roughness_shap"]),
            textcoords="offset points",
            xytext=(5, 5),
            ha='left',
            fontsize=9
        )
        
    # Draw quadrant lines at medians
    x_mid = x.median()
    y_mid = y.median()
    
    ax.axvline(x_mid, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y_mid, color='gray', linestyle='--', alpha=0.5)
    
    # Annotate quadrants
    ax.text(ax.get_xlim()[1]*0.9, ax.get_ylim()[1]*0.9, "Optimize\nBoth", ha='right', va='top', fontsize=10, color='darkgreen', fontweight='bold')
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.1, ax.get_ylim()[1]*0.9, "Roughness\nOnly", ha='left', va='top', fontsize=10, color='darkblue', fontweight='bold')
    ax.text(ax.get_xlim()[1]*0.9, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1, "Energy\nOnly", ha='right', va='bottom', fontsize=10, color='darkorange', fontweight='bold')
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1]-ax.get_xlim()[0])*0.1, ax.get_ylim()[0] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.1, "Low\nImpact", ha='left', va='bottom', fontsize=10, color='gray', fontweight='bold')
    
    ax.set_title("Feature Conflict Analysis: Energy vs Surface Roughness", fontsize=12, fontweight='bold')
    ax.set_xlabel("Feature Importance for Energy (Mean |SHAP|)", fontsize=11)
    ax.set_ylabel("Feature Importance for Roughness (Mean |SHAP|)", fontsize=11)
    plt.tight_layout()
    
    output_dir = Path("outputs/figures/shap")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "energy_roughness_conflict.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved energy vs roughness conflict plot to {save_path}")
    return str(save_path)

def generate_paper_shap_table(
    roughness_importance: pd.DataFrame,
    time_importance: pd.DataFrame,
    energy_importance: pd.DataFrame
) -> str:
    """
    Generates LaTeX and CSV tables summarizing top features across targets.
    """
    # Collect all unique features
    all_feats = set(roughness_importance["feature_name"]).union(
        time_importance["feature_name"]
    ).union(energy_importance["feature_name"])
    
    table_rows = []
    for feat in all_feats:
        r_row = roughness_importance[roughness_importance["feature_name"] == feat]
        t_row = time_importance[time_importance["feature_name"] == feat]
        e_row = energy_importance[energy_importance["feature_name"] == feat]
        
        r_rank = int(r_row.iloc[0]["rank"]) if not r_row.empty else 999
        r_val = r_row.iloc[0]["mean_abs_shap"] if not r_row.empty else 0.0
        
        t_rank = int(t_row.iloc[0]["rank"]) if not t_row.empty else 999
        t_val = t_row.iloc[0]["mean_abs_shap"] if not t_row.empty else 0.0
        
        e_rank = int(e_row.iloc[0]["rank"]) if not e_row.empty else 999
        e_val = e_row.iloc[0]["mean_abs_shap"] if not e_row.empty else 0.0
        
        avg_rank = np.mean([r_rank, t_rank, e_rank])
        
        table_rows.append({
            "Feature": feat,
            "Energy Rank": e_rank,
            "Energy SHAP": e_val,
            "Roughness Rank": r_rank,
            "Roughness SHAP": r_val,
            "Time Rank": t_rank,
            "Time SHAP": t_val,
            "Avg Rank": avg_rank
        })
        
    df_table = pd.DataFrame(table_rows)
    df_table = df_table.sort_values("Avg Rank").reset_index(drop=True)
    
    # Save CSV
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "shap_importance_table.csv"
    df_table.to_csv(csv_path, index=False)
    
    # Generate LaTeX
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{SHAP Feature Importance Rankings Across Multi-Objective Prediction Targets}",
        "\\label{tab:shap_importance}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Feature & \\multicolumn{2}{c}{Energy (SEC)} & \\multicolumn{2}{c}{Surface Roughness} & \\multicolumn{2}{c}{Machining Time} \\\\",
        " & Rank & Mean $|$SHAP$|$ & Rank & Mean $|$SHAP$|$ & Rank & Mean $|$SHAP$|$ \\\\",
        "\\hline"
    ]
    
    for i, row in df_table.head(10).iterrows():
        feat = row["Feature"].replace("_", "\\_")
        
        e_rank = str(row["Energy Rank"]) if row["Energy Rank"] != 999 else "-"
        e_val = f"{row['Energy SHAP']:.4f}"
        if row["Energy Rank"] == 1:
            e_rank = f"\\textbf{{{e_rank}}}"
            e_val = f"\\textbf{{{e_val}}}"
            
        r_rank = str(row["Roughness Rank"]) if row["Roughness Rank"] != 999 else "-"
        r_val = f"{row['Roughness SHAP']:.4f}"
        if row["Roughness Rank"] == 1:
            r_rank = f"\\textbf{{{r_rank}}}"
            r_val = f"\\textbf{{{r_val}}}"
            
        t_rank = str(row["Time Rank"]) if row["Time Rank"] != 999 else "-"
        t_val = f"{row['Time SHAP']:.4f}"
        if row["Time Rank"] == 1:
            t_rank = f"\\textbf{{{t_rank}}}"
            t_val = f"\\textbf{{{t_val}}}"
            
        latex_lines.append(f"{feat} & {e_rank} & {e_val} & {r_rank} & {r_val} & {t_rank} & {t_val} \\\\")
        
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    tex_path = results_dir / "shap_importance_table.tex"
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))
        
    logger.info(f"Saved LaTeX shap table to {tex_path} and CSV table to {csv_path}")
    return str(tex_path)

def generate_all_comparison_plots(results: dict) -> list:
    """
    Calls all comparison plotting functions using stored SHAP analyzers.
    
    Args:
        results: Dict containing {target_name: SHAPAnalyzer instance}.
        
    Returns:
        List of strings showing output file paths.
    """
    logger.info("Generating all cross-target SHAP comparison outputs...")
    
    r_imp = results["roughness"].get_feature_importance()
    t_imp = results["time"].get_feature_importance()
    e_imp = results["energy"].get_feature_importance()
    
    paths = [
        plot_feature_importance_comparison(r_imp, t_imp, e_imp),
        plot_conflict_analysis(r_imp, e_imp),
        generate_paper_shap_table(r_imp, t_imp, e_imp)
    ]
    
    return paths
