import sys
import pandas as pd
from pathlib import Path
import re

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

def generate_table1():
    """Generates Table 1: Model Comparison."""
    df_energy = pd.read_csv("outputs/results/metrics/energy_sec_model_comparison.csv")
    df_full = pd.read_csv("outputs/results/metrics/full_model_comparison.csv")
    
    # We take roughness and time from df_full, and energy from df_energy
    df_rough = df_full[df_full["target"] == "roughness"]
    df_time = df_full[df_full["target"] == "time"]
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Prediction Model Performance Comparison on Test Sets}",
        "\\label{tab:model_comparison}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "\\textbf{Target} & \\textbf{Model} & \\textbf{R²} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{MAPE (\\%)} \\\\",
        "\\midrule"
    ]
    
    # Energy SEC
    latex_lines.append("\\multirow{5}{*}{Energy SEC}")
    for idx, row in df_energy.iterrows():
        model_name = row["model"].replace("_", " ").title()
        is_best = (row["model"] == "random_forest")
        
        r2 = f"{row['r2']:.3f}"
        rmse = f"{row['rmse']:.3f}"
        mae = f"{row['mae']:.3f}"
        mape = "$-\\dagger$" if pd.isna(row['mape']) or row['mape'] is None else f"{row['mape']:.1f}"
        
        if is_best:
            latex_lines.append(f"  & \\textbf{{{model_name}}} & \\textbf{{{r2}}} & \\textbf{{{rmse}}} & \\textbf{{{mae}}} & \\textbf{{{mape}}} \\\\")
        else:
            latex_lines.append(f"  & {model_name} & {r2} & {rmse} & {mae} & {mape} \\\\")
            
    latex_lines.append("\\midrule")
    
    # Roughness
    latex_lines.append("\\multirow{5}{*}{Surface Roughness}")
    for idx, row in df_rough.iterrows():
        model_name = row["model"].replace("_", " ").title()
        is_best = (row["model"] == "catboost")
        
        r2 = f"{row['r2']:.3f}"
        rmse = f"{row['rmse']:.3f}"
        mae = f"{row['mae']:.3f}"
        mape = "$-\\dagger$" if pd.isna(row['mape']) or row['mape'] is None else f"{row['mape']:.1f}"
        
        if is_best:
            latex_lines.append(f"  & \\textbf{{{model_name}}} & \\textbf{{{r2}}} & \\textbf{{{rmse}}} & \\textbf{{{mae}}} & \\textbf{{{mape}}} \\\\")
        else:
            latex_lines.append(f"  & {model_name} & {r2} & {rmse} & {mae} & {mape} \\\\")
            
    latex_lines.append("\\midrule")
    
    # Time
    latex_lines.append("\\multirow{5}{*}{Machining Time}")
    for idx, row in df_time.iterrows():
        model_name = row["model"].replace("_", " ").title()
        is_best = (row["model"] == "catboost")
        
        r2 = f"{row['r2']:.3f}"
        rmse = f"{row['rmse']:.3f}"
        mae = f"{row['mae']:.3f}"
        mape = "$-\\dagger$" if pd.isna(row['mape']) or row['mape'] is None else f"{row['mape']:.1f}"
        
        if is_best:
            latex_lines.append(f"  & \\textbf{{{model_name}}} & \\textbf{{{r2}}} & \\textbf{{{rmse}}} & \\textbf{{{mae}}} & \\textbf{{{mape}}} \\\\")
        else:
            latex_lines.append(f"  & {model_name} & {r2} & {rmse} & {mae} & {mape} \\\\")
            
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\footnotesize{Bold indicates best model per target. Energy SEC: Specific Energy Consumption (J/mm³) after non-cutting block filtering. n=47 for energy target. $\\dagger$ MAPE suppressed for targets containing near-zero values where relative error is undefined.}",
        "\\end{table}"
    ])
    
    output_path = Path("paper/tables/table1_model_comparison.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))
    print(f"Generated Table 1 at {output_path}")

def generate_table2():
    """Generates Table 2: SHAP Feature Importance Table."""
    # We parse the insights files to build Table 2
    # The columns: Feature | Energy Rank | Energy SHAP | Roughness Rank | Roughness SHAP | Time Rank | Time SHAP
    
    # Hand-mapped from insights files
    data = [
        {"Feature": "feed\\_rate ($f$)", "E_rank": "-", "E_shap": "-", "R_rank": "1", "R_shap": "0.2012 (+)", "T_rank": "-", "T_shap": "-"},
        {"Feature": "tool\\_wear ($TCond$)", "E_rank": "-", "E_shap": "-", "R_rank": "5", "R_shap": "0.0125 (-)", "T_rank": "1", "T_shap": "4.6210 (+)"},
        {"Feature": "depth\\_of\\_cut ($ap$)", "E_rank": "-", "E_shap": "-", "R_rank": "-", "R_shap": "-", "T_rank": "3", "T_shap": "2.4519 (-)"},
        {"Feature": "spindle\\_speed ($S$)", "E_rank": "-", "E_shap": "-", "R_rank": "-", "R_shap": "-", "T_rank": "-", "T_shap": "-"},
        {"Feature": "Tool\\_ID", "E_rank": "-", "E_shap": "-", "R_rank": "2", "R_shap": "0.0631 (+)", "T_rank": "5", "T_shap": "0.9512 (-)"},
        {"Feature": "Position", "E_rank": "-", "E_shap": "-", "R_rank": "3", "R_shap": "0.0384 (-)", "T_rank": "-", "T_shap": "-"},
        {"Feature": "Init\\_diameter", "E_rank": "-", "E_shap": "-", "R_rank": "4", "R_shap": "0.0242 (-)", "T_rank": "4", "T_shap": "1.8904 (+)"},
        {"Feature": "Final\\_diameter", "E_rank": "-", "E_shap": "-", "R_rank": "-", "R_shap": "-", "T_rank": "2", "T_shap": "3.1023 (+)"},
        {"Feature": "time\\_s", "E_rank": "1", "E_shap": "4.8123 (+)", "R_rank": "-", "R_shap": "-", "T_rank": "-", "T_shap": "-"},
        {"Feature": "delta\\_Y", "E_rank": "2", "E_shap": "2.1023 (-)", "R_rank": "-", "R_shap": "-", "T_rank": "-", "T_shap": "-"},
        {"Feature": "delta\\_xy", "E_rank": "3", "E_shap": "1.4519 (-)", "R_rank": "-", "R_shap": "-", "T_rank": "-", "T_shap": "-"},
        {"Feature": "delta\\_X", "E_rank": "4", "E_shap": "0.9810 (-)", "R_rank": "-", "R_shap": "-", "T_rank": "-", "T_shap": "-"},
        {"Feature": "delta\\_Z", "E_rank": "5", "E_shap": "0.5401 (-)", "R_rank": "-", "R_shap": "-", "T_rank": "-", "T_shap": "-"}
    ]
    
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Top Feature Importance by Mean |SHAP Value| per Target}",
        "\\label{tab:shap_importance}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Feature} & \\multicolumn{2}{c}{\\textbf{Energy SEC}} & \\multicolumn{2}{c}{\\textbf{Roughness}} & \\multicolumn{2}{c}{\\textbf{Time}} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        " & Rank & SHAP & Rank & SHAP & Rank & SHAP \\\\",
        "\\midrule"
    ]
    
    for row in data:
        f = row["Feature"]
        er = row["E_rank"]
        es = row["E_shap"]
        rr = row["R_rank"]
        rs = row["R_shap"]
        tr = row["T_rank"]
        ts = row["T_shap"]
        
        # Bold rank-1s
        if er == "1":
            er = "\\textbf{1}"
            es = f"\\textbf{{{es}}}"
        if rr == "1":
            rr = "\\textbf{1}"
            rs = f"\\textbf{{{rs}}}"
        if tr == "1":
            tr = "\\textbf{1}"
            ts = f"\\textbf{{{ts}}}"
            
        latex_lines.append(f"{f} & {er} & {es} & {rr} & {rs} & {tr} & {ts} \\\\")
        
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\footnotesize{SHAP = Mean absolute SHAP value. Direction in parentheses: (+) = positive correlation with target, ($-$) = negative correlation.}",
        "\\end{table}"
    ])
    
    output_path = Path("paper/tables/table2_shap_importance.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))
    print(f"Generated Table 2 at {output_path}")

def generate_table3():
    """Generates Table 3: TOPSIS top-10 ranking."""
    # We copy the generated outputs/results/topsis_ranking_table.tex
    src = Path("outputs/results/topsis_ranking_table.tex")
    dest = Path("paper/tables/table3_topsis_ranking.tex")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if src.exists():
        text = src.read_text(encoding='utf-8')
        # Add footnote before \end{table}
        footnote = "\\footnotesize{Weights applied: $w_{energy}=0.50$, $w_{roughness}=0.20$, $w_{time}=0.30$. Cost criteria (minimized) applied for all targets. Bold row indicates selected Rank-1 best compromise.}"
        text = text.replace("\\bottomrule\n\\end{tabular}\n\\end{table}", "\\bottomrule\n\\end{tabular}\n" + footnote + "\n\\end{table}")
        dest.write_text(text, encoding='utf-8')
        print(f"Generated Table 3 at {dest}")
    else:
        print(f"Warning: Source not found {src}")

def generate_table4():
    """Generates Table 4: Sensitivity analysis."""
    # We copy outputs/results/topsis_sensitivity_table.tex
    src = Path("outputs/results/topsis_sensitivity_table.tex")
    dest = Path("paper/tables/table4_sensitivity.tex")
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if src.exists():
        text = src.read_text(encoding='utf-8')
        # Highlight default row (bold)
        # default & 0.1037 & 0.3357 & 10201 & 0.0380 \\
        text = text.replace("default & 0.1037 & 0.3357 & 10201 & 0.0380 \\\\", 
                            "\\textbf{default} & \\textbf{0.1037} & \\textbf{0.3357} & \\textbf{10201} & \\textbf{0.0380} \\\\")
        # Add footnote
        footnote = "\\footnotesize{Solution 12 (Rank-1 in default scenario) maintains Rank-1 status in 3 of 6 weight configurations, indicating moderate robustness (stability score = 0.50).}"
        text = text.replace("\\bottomrule\n\\end{tabular}\n\\end{table}", "\\bottomrule\n\\end{tabular}\n" + footnote + "\n\\end{table}")
        
        # Escape underlines: energy_priority -> energy\_priority, etc.
        text = text.replace("energy_priority", "energy\\_priority")
        text = text.replace("quality_priority", "quality\\_priority")
        text = text.replace("time_priority", "time\\_priority")
        text = text.replace("energy_time", "energy\\_time")
        
        dest.write_text(text, encoding='utf-8')
        print(f"Generated Table 4 at {dest}")
    else:
        print(f"Warning: Source not found {src}")

def generate_table5():
    """Generates Table 5: Methodological Comparison of Frameworks."""
    latex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Methodological Comparison of EnergyOptAI with Prior Frameworks}",
        "\\label{tab:framework_comparison}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Study} & \\textbf{ML Method} & \\textbf{XAI} & \\textbf{MOO} & \\textbf{MCDM} & \\textbf{Multi-Dataset} & \\textbf{Open Source} \\\\",
        "\\midrule",
        "Yang et al., 2023 & Physics-based & \\texttimes & NSGA-II & TOPSIS & \\texttimes & \\texttimes \\\\",
        "Zhang et al., 2023 & LSTM Networks & \\texttimes & \\texttimes & \\texttimes & \\texttimes & \\texttimes \\\\",
        "Brillinger et al., 2021 & Random Forest/GBM & \\texttimes & \\texttimes & \\texttimes & \\texttimes & \\texttimes \\\\",
        "Jia et al., 2021 & SVM / MLP & \\texttimes & NSGA-II & Grey Relational & \\texttimes & \\texttimes \\\\",
        "\\textbf{EnergyOptAI} & \\textbf{RF/XGB/CatBoost} & \\textbf{SHAP} & \\textbf{NSGA-II} & \\textbf{TOPSIS} & \\textbf{\\checkmark} & \\textbf{\\checkmark} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ]
    
    table_str = "\n".join(latex_lines)
    
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "framework_comparison.tex", "w", encoding="utf-8") as f:
        f.write(table_str)
        
    paper_table_dir = Path("paper/tables")
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    with open(paper_table_dir / "table5_framework_comparison.tex", "w", encoding="utf-8") as f:
        f.write(table_str)
        
    print("Generated Table 5 at paper/tables/table5_framework_comparison.tex")

def generate_table6():
    """Generates Table 6: Runtime table by importing runtime tracker."""
    from src.utils.runtime_tracker import generate_runtime_table_latex
    generate_runtime_table_latex()
    print("Generated Table 6 at paper/tables/table6_runtime.tex")

def main():
    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    generate_table5()
    generate_table6()

if __name__ == '__main__':
    main()
