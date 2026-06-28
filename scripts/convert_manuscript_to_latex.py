import re
from pathlib import Path

def convert_to_latex():
    src_path = Path("paper/manuscript.md")
    dest_path = Path("paper/submission_package/manuscript.tex")
    
    if not src_path.exists():
        print(f"Error: {src_path} not found.")
        return
        
    text = src_path.read_text(encoding="utf-8")
    
    # -------------------------------------------------------------
    # 1. Section Header Replacements
    # -------------------------------------------------------------
    # Abstract
    text = text.replace("# Abstract", "\\section*{Abstract}")
    
    # Main sections: # 1. Introduction -> \section{Introduction}
    text = re.sub(r'# \d+\.\s+([^\n]+)', r'\\section{\1}', text)
    
    # Subsections: ### 2.1 ... -> \subsection{...}
    text = re.sub(r'### \d+\.\d+\s+([^\n]+)', r'\\subsection{\1}', text)
    
    # Subsubsections: #### 3.1.1 ... -> \subsubsection{...}
    text = re.sub(r'#### \d+\.\d+\.\d+\s+([^\n]+)', r'\\subsubsection{\1}', text)
    
    # -------------------------------------------------------------
    # 2. Text Formatting Replacements
    # -------------------------------------------------------------
    # Bold: **text** -> \textbf{text}
    text = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', text)
    
    # Italics: *text* -> \textit{text}
    text = re.sub(r'\*([a-zA-Z0-9\s]+)\*', r'\\textit{\1}', text)
    
    # Bullet points: list matching
    lines = text.splitlines()
    in_list = False
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("* ") or stripped.startswith("- "):
            if not in_list:
                new_lines.append("\\begin{itemize}")
                in_list = True
            item_text = stripped[2:]
            new_lines.append(f"  \\item {item_text}")
        else:
            if in_list:
                new_lines.append("\\end{itemize}")
                in_list = False
            new_lines.append(line)
    if in_list:
        new_lines.append("\\end{itemize}")
    text = "\n".join(new_lines)
    
    # -------------------------------------------------------------
    # 3. Figure Embeddings Conversion
    # -------------------------------------------------------------
    fig_labels = {
        "Fig1_framework_overview.png": "fig:framework",
        "Fig2_target_distributions.png": "fig:distributions",
        "Fig3_model_comparison.png": "fig:comparison",
        "Fig4_actual_vs_predicted.png": "fig:actual_vs_predicted",
        "Fig5_shap_importance.png": "fig:shap_importance",
        "Fig6_shap_beeswarm_roughness.png": "fig:shap_beeswarm",
        "Fig7_feature_conflict.png": "fig:conflict",
        "Fig8_nsga2_convergence.png": "fig:nsga2_convergence",
        "Fig9_pareto_projections.png": "fig:pareto_projections",
        "Fig10_topsis_radar.png": "fig:topsis_radar",
        "Fig11_sensitivity_heatmap.png": "fig:sensitivity_heatmap",
        "Fig12_tool_wear_pareto_shift.png": "fig:tool_wear_pareto_shift",
        "Fig13_proximity_distances.png": "fig:proximity_distances"
    }
    
    # Match markdown figures: ![Figure N: Caption](figures/FigN_name.png)
    # followed by optional figure caption block
    fig_pattern = r'!\[Figure (\d+):\s*([^\]]+)\]\(figures/(Fig\d+_[^)]+\.png)\)\s*(?:\n\s*\*Figure \1:\s*([^*]+)\.\*)?'
    
    def fig_repl(match):
        num = match.group(1)
        caption = match.group(2).strip()
        filename = match.group(3).strip()
        label = fig_labels.get(filename, f"fig:figure{num}")
        
        return f"""\\begin{{figure}}[htbp]
\\centering
\\includegraphics[width=0.9\\textwidth]{{figures/{filename}}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{figure}}"""
        
    text = re.sub(fig_pattern, fig_repl, text)
    
    # -------------------------------------------------------------
    # 4. Table Placeholders Conversion
    # -------------------------------------------------------------
    # Pattern: [Table N: Caption — see paper/tables/tableN_name.tex]
    tab_pattern = r'\[Table (\d+):[^\s]*\s*([^\s]+)\s*—\s*see\s*paper/tables/(table\d+_[^\]]+\.tex)\]'
    def tab_repl(match):
        filename = match.group(3).strip()
        return f"\\input{{tables/{filename}}}"
        
    text = re.sub(tab_pattern, tab_repl, text)
    
    # -------------------------------------------------------------
    # 5. Assemble LaTeX Document
    # -------------------------------------------------------------
    latex_doc = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{multirow}}
\\usepackage{{hyperref}}
\\usepackage{{lineno}}    % line numbers
\\usepackage{{natbib}}    % for references
\\linenumbers

\\title{{EnergyOptAI: An Explainable Machine Learning Framework with Multi-Objective Optimization for Energy-Efficient CNC Machining Parameter Selection}}
\\author{{EnergyOptAI Research Team}}
\\date{{June 28, 2026}}

\\begin{{document}}
\\maketitle

{text}

\\bibliographystyle{{elsarticle-num}}
\\bibliography{{references}}
\\end{{document}}"""

    # Strip double title section created from markdown header
    latex_doc = re.sub(r'\\section\{EnergyOptAI: An Explainable Machine Learning, Multi-Objective Optimization, and MCDM Framework for CNC Machining Parameter Selection\}', '', latex_doc)
    latex_doc = re.sub(r'---\s*\n', '', latex_doc) # strip hr lines

    dest_path.write_text(latex_doc, encoding="utf-8")
    print(f"Generated manuscript.tex at {dest_path}")

if __name__ == "__main__":
    convert_to_latex()
