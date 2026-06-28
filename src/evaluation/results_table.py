import logging
import json
from pathlib import Path
import pandas as pd

logger = logging.getLogger("ResultsTable")

def build_comparison_df(all_results: list) -> pd.DataFrame:
    """
    Converts list of metric dicts into a formatted DataFrame.
    Columns: target, model, r2, rmse, mae, mape, n_samples
    Sorted by: target asc, r2 desc
    """
    df = pd.DataFrame(all_results)
    required_cols = ["target", "model", "r2", "rmse", "mae", "mape", "n_samples"]
    
    # Ensure all required columns are present
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            
    df = df[required_cols]
    df = df.sort_values(by=["target", "r2"], ascending=[True, False]).reset_index(drop=True)
    return df

def save_results(df: pd.DataFrame, prefix: str) -> None:
    """
    Saves the DataFrame to CSV and JSON format under outputs/results/metrics/.
    """
    out_dir = Path("outputs/results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / f"{prefix}.csv"
    json_path = out_dir / f"{prefix}.json"
    
    df.to_csv(csv_path, index=False)
    
    # Convert df to dictionary format for json saving
    data_dict = df.to_dict(orient="records")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)
        
    logger.info(f"Saved results summary to {csv_path} and {json_path}")

def print_latex_table(df: pd.DataFrame) -> None:
    """
    Prints LaTeX table string for direct paper inclusion (booktabs style).
    Bolds the best R2 per target group.
    """
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Model Performance Comparison across Targets}")
    latex_lines.append("\\begin{tabular}{llrrrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Target & Model & $R^2$ & RMSE & MAE & MAPE (\\%) \\\\")
    latex_lines.append("\\midrule")
    
    targets = df["target"].unique()
    
    for target in targets:
        df_target = df[df["target"] == target]
        if df_target.empty:
            continue
            
        # Find index of best R2 in this target group
        best_idx = df_target["r2"].idxmax()
        
        for idx, row in df_target.iterrows():
            is_best = (idx == best_idx)
            
            # Format model name beautifully
            model_name = row['model'].replace('_', ' ').title()
            
            # Apply bold to R2 if best
            r2_val = f"\\textbf{{{row['r2']:.4f}}}" if is_best else f"{row['r2']:.4f}"
            
            # Format MAPE
            mape_val = "$-\\dagger$" if pd.isna(row['mape']) or row['mape'] is None else f"{row['mape']:.2f}"
            
            # Format row
            latex_lines.append(
                f"{row['target'].capitalize()} & {model_name} & {r2_val} & "
                f"{row['rmse']:.4f} & {row['mae']:.4f} & {mape_val} \\\\"
            )
        latex_lines.append("\\hline")
        
    # Remove last \hline and close tabular
    if latex_lines[-1] == "\\hline":
        latex_lines.pop()
        
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\footnotesize{$\\dagger$ MAPE suppressed for targets containing near-zero values where relative error is undefined.}")
    latex_lines.append("\\end{table}")
    
    print("\n--- LATEX TABLE FOR PAPER INCLUSION ---")
    print("\n".join(latex_lines))
    print("---------------------------------------\n")
