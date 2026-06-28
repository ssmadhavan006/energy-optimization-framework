import os
from pathlib import Path
import re

def main():
    sections = [
        ("Abstract", "paper/sections/00_abstract.md"),
        ("1. Introduction", "paper/sections/01_introduction.md"),
        ("2. Related Work", "paper/sections/02_related_work.md"),
        ("3. Methodology", "paper/sections/03_methodology.md"),
        ("4. Datasets and Preprocessing", "paper/sections/04_datasets.md"),
        ("5. Experimental Results", "paper/sections/05_results.md"),
        ("6. Discussion and Limitations", "paper/sections/06_discussion.md"),
        ("7. Conclusion", "paper/sections/07_conclusion.md")
    ]
    
    manuscript_lines = [
        "# EnergyOptAI: An Explainable Machine Learning, Multi-Objective Optimization, and MCDM Framework for CNC Machining Parameter Selection",
        "",
        "**Author**: Antigravity AI Pair Programmer & Research Team",
        "",
        "---",
        ""
    ]
    
    for title, path_str in sections:
        p = Path(path_str)
        if p.exists():
            content = p.read_text(encoding='utf-8')
            manuscript_lines.append(f"# {title}")
            manuscript_lines.append("")
            manuscript_lines.append(content)
            manuscript_lines.append("")
            manuscript_lines.append("---")
            manuscript_lines.append("")
        else:
            print(f"Warning: Section file not found {p}")
            
    assembled_text = "\n".join(manuscript_lines)
    
    # Insert figures and tables references at logical paragraphs
    # Figure 1: Framework Overview (Stage 1)
    assembled_text = assembled_text.replace(
        "As shown in Figure~\\ref{fig:framework}, the pipeline begins by loading",
        "As shown in the framework overview (Figure 1), the pipeline begins by loading"
    )
    assembled_text = assembled_text.replace(
        "# 3. Methodology\n\nThe",
        "# 3. Methodology\n\n![Figure 1: EnergyOptAI Integrated Framework Architecture Flowchart](figures/Fig1_framework_overview.png)\n\n*Figure 1: EnergyOptAI Integrated Framework Architecture Flowchart.*\n\nThe"
    )
    
    # Figure 2: Target distributions
    assembled_text = assembled_text.replace(
        "# 4. Datasets and Preprocessing\n\nTo train",
        "# 4. Datasets and Preprocessing\n\n![Figure 2: Target Variable Distributions across Mendeley and Kaggle datasets](figures/Fig2_target_distributions.png)\n\n*Figure 2: Target Variable Distributions across Mendeley and Kaggle datasets.*\n\nTo train"
    )
    
    # Table 1: Model Comparison
    assembled_text = assembled_text.replace(
        "Table~\\ref{tab:model_comparison} summarizes the test set accuracy metrics.",
        "Table 1 summarizes the test set accuracy metrics.\n\n[Table 1: Prediction Model Performance Comparison on Test Sets — see paper/tables/table1_model_comparison.tex]"
    )
    
    # Figure 3 & 4: Model comparison and Actual vs Predicted
    assembled_text = assembled_text.replace(
        "illustrating their fit across the test sets.",
        "illustrating their fit across the test sets.\n\n![Figure 3: Model Performance R² Comparison Across Targets](figures/Fig3_model_comparison.png)\n\n*Figure 3: Model Performance R² Comparison Across Targets.*\n\n![Figure 4: Actual vs. Predicted Scatter Plots for Champion Models](figures/Fig4_actual_vs_predicted.png)\n\n*Figure 4: Actual vs. Predicted Scatter Plots for Champion Models.*"
    )
    
    # Table 2 & Figure 5 & 7: SHAP
    assembled_text = assembled_text.replace(
        "Table~\\ref{tab:shap_importance} lists the mean absolute SHAP values and correlations.",
        "Table 2 lists the mean absolute SHAP values and correlations.\n\n[Table 2: Top Feature Importance by Mean |SHAP Value| per Target — see paper/tables/table2_shap_importance.tex]"
    )
    assembled_text = assembled_text.replace(
        "Reference Figure 5 (SHAP importance panels).\nReference Figure 7 (conflict plot).",
        "![Figure 5: Global Feature Importance (Mean Absolute SHAP Value) Comparison](figures/Fig5_shap_importance.png)\n\n*Figure 5: Global Feature Importance (Mean Absolute SHAP Value) Comparison.*\n\n![Figure 6: SHAP Beeswarm Summary Plot for Surface Roughness](figures/Fig6_shap_beeswarm_roughness.png)\n\n*Figure 6: SHAP Beeswarm Summary Plot for Surface Roughness.*\n\n![Figure 7: Feature Conflict Analysis between Energy and Surface Roughness](figures/Fig7_feature_conflict.png)\n\n*Figure 7: Feature Conflict Analysis between Energy and Surface Roughness.*"
    )
    assembled_text = assembled_text.replace(
        "Figure~\\ref{fig:conflict}",
        "Figure 7"
    )
    
    # Figure 8 & 9: Pareto
    assembled_text = assembled_text.replace(
        "reference convergence plot Figure~\\ref{fig:convergence}",
        "reference convergence plot (Figure 8)"
    )
    assembled_text = assembled_text.replace(
        "illustrating the trade-off curve between cycle time and surface roughness. Figure~\\ref{fig:pareto_projections} displays",
        "illustrating the trade-off curve between cycle time and surface roughness.\n\n![Figure 8: NSGA-II Hypervolume Convergence Curve](figures/Fig8_nsga2_convergence.png)\n\n*Figure 8: NSGA-II Hypervolume Convergence Curve.*\n\n![Figure 9: Pareto Front 2D Projections](figures/Fig9_pareto_projections.png)\n\n*Figure 9: Pareto Front 2D Projections.*\n\nFigure 9 displays"
    )
    assembled_text = assembled_text.replace(
        "Reference Figures 8 and 9.",
        ""
    )
    
    # Table 3 & Figure 10 & 11: TOPSIS
    assembled_text = assembled_text.replace(
        "Table~\\ref{tab:topsis_ranking} lists the top 10 ranked solutions.",
        "Table 3 lists the top 10 ranked solutions.\n\n[Table 3: TOPSIS Multi-Criteria Decision Making (MCDM) Ranking of Pareto Solutions — see paper/tables/table3_topsis_ranking.tex]"
    )
    assembled_text = assembled_text.replace(
        "Table~\\ref{tab:recommendation}",
        "Table 3"
    )
    assembled_text = assembled_text.replace(
        "Table 4 for sensitivity.",
        "Table 4 for sensitivity.\n\n[Table 4: TOPSIS Decision Sensitivity: Recommended Parameters Across Weight Scenarios — see paper/tables/table4_sensitivity.tex]"
    )
    assembled_text = assembled_text.replace(
        "Reference Figure 10.",
        "![Figure 10: Radar Chart Comparison of Top TOPSIS Solutions](figures/Fig10_topsis_radar.png)\n\n*Figure 10: Radar Chart Comparison of Top TOPSIS Solutions.*"
    )
    assembled_text = assembled_text.replace(
        "Figure~\\ref{fig:sensitivity_heatmap} illustrates these rank shifts.",
        "Figure 11 illustrates these rank shifts.\n\n![Figure 11: TOPSIS Rank Sensitivity Heatmap to Weight Variations](figures/Fig11_sensitivity_heatmap.png)\n\n*Figure 11: TOPSIS Rank Sensitivity Heatmap to Weight Variations.*"
    )
    
    # Figure 12: Tool Wear Pareto Shift
    assembled_text = assembled_text.replace(
        "the Pareto front shifts dynamically (Figure \\ref{fig:tool_wear_pareto_shift}).",
        "the Pareto front shifts dynamically (Figure 12).\n\n![Figure 12: Pareto front shifts under different tool wear scenario states](figures/Fig12_tool_wear_pareto_shift.png)\n\n*Figure 12: Pareto front shifts under different tool wear scenario states.*"
    )
    
    # Figure 13: Proximity Distances
    assembled_text = assembled_text.replace(
        "margins of the model's high-confidence envelope. This highlights the absolute necessity of our OOD checks",
        "margins of the model's high-confidence envelope (Figure 13). This highlights the absolute necessity of our OOD checks\n\n![Figure 13: Nearest-neighbor Euclidean distance distributions of Pareto coordinates](figures/Fig13_proximity_distances.png)\n\n*Figure 13: Nearest-neighbor Euclidean distance distributions of Pareto coordinates.*"
    )
    
    # Table 5: Framework Comparison
    assembled_text = assembled_text.replace(
        "Table \\ref{tab:framework_comparison}",
        "Table 5\n\n[Table 5: Methodological Comparison of State-of-the-Art Frameworks — see paper/tables/table5_framework_comparison.tex]"
    )
    
    # Table 6: Runtimes
    # Let's insert a reference to Table 6 in Section 6.5 Practical Deployment Workflow
    assembled_text = assembled_text.replace(
        "automated optimization loop.",
        "automated optimization loop.\n\n[Table 6: EnergyOptAI Stage Execution Runtimes — see paper/tables/table6_runtime.tex]"
    )
    
    # Table 7: Ablation Study
    assembled_text = assembled_text.replace(
        "Table~\\ref{tab:ablation} details the results:",
        "Table 7 details the results:\n\n[Table 7: Ablation Study of Framework Components — see paper/tables/table7_ablation.tex]"
    )
    
    # Save the manuscript
    output_path = Path("paper/manuscript.md")
    output_path.write_text(assembled_text, encoding='utf-8')
    print(f"Assembled manuscript saved to {output_path}")
    
    # Verify word counts
    words = len(assembled_text.split())
    print(f"Word count: {words}")
    
    # Count figures references
    figs = re.findall(r'Fig\d+_', assembled_text)
    print(f"Figure references: {len(set(figs))}")

if __name__ == '__main__':
    main()
