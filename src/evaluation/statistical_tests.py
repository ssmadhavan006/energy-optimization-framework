import os
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
from typing import Dict, List, Any

logger = logging.getLogger("StatisticalTester")

class StatisticalTester:
    """
    Runs Wilcoxon signed-rank tests on cross-validation fold scores
    to determine if differences in model performance are statistically significant.
    """
    
    def __init__(self, metrics_dir: Path = Path("outputs/results/metrics")):
        self.metrics_dir = Path(metrics_dir)
        
    def load_cv_scores(self, target: str, model: str) -> List[float]:
        """Loads CV RMSE scores for a specific target and model."""
        path = self.metrics_dir / f"{target}_{model}_cv_folds.json"
        if not path.exists():
            raise FileNotFoundError(f"CV fold metrics file not found at {path}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Extract RMSE from each fold
        rmse_scores = [fold["rmse"] for fold in data]
        return rmse_scores

    def wilcoxon_pairwise(
        self,
        scores_a: List[float],
        scores_b: List[float],
        model_a_name: str,
        model_b_name: str,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Runs Wilcoxon signed-rank test between two lists of fold scores."""
        scores_a = np.asarray(scores_a)
        scores_b = np.asarray(scores_b)
        
        diff = scores_a - scores_b
        if np.all(diff == 0.0):
            # Identical scores: cannot compute wilcoxon
            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "interpretation": "Identical performance"
            }
            
        try:
            # alternative="two-sided"
            stat, p = wilcoxon(scores_a, scores_b)
            significant = p < alpha
            interpretation = (
                f"Significant difference (p={p:.4f})" if significant 
                else f"No significant difference (p={p:.4f})"
            )
            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "statistic": float(stat),
                "p_value": float(p),
                "significant": significant,
                "interpretation": interpretation
            }
        except Exception as e:
            logger.error(f"Failed to compute Wilcoxon test: {e}")
            return {
                "model_a": model_a_name,
                "model_b": model_b_name,
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "interpretation": f"Error: {e}"
            }

    def run_all_pairwise(self, target: str, models: List[str]) -> pd.DataFrame:
        """Runs Wilcoxon test for all pairs of models for a target."""
        scores = {}
        for m in models:
            try:
                scores[m] = self.load_cv_scores(target, m)
            except Exception as e:
                logger.warning(f"Could not load CV scores for {m} on {target}: {e}")
                
        results = []
        loaded_models = list(scores.keys())
        
        for i in range(len(loaded_models)):
            for j in range(i + 1, len(loaded_models)):
                m_a = loaded_models[i]
                m_b = loaded_models[j]
                
                res = self.wilcoxon_pairwise(scores[m_a], scores[m_b], m_a, m_b)
                
                # Determine direction (which model has lower mean RMSE)
                mean_a = np.mean(scores[m_a])
                mean_b = np.mean(scores[m_b])
                better_model = m_a if mean_a < mean_b else m_b
                
                res["target"] = target
                res["mean_a"] = float(mean_a)
                res["mean_b"] = float(mean_b)
                res["better_model"] = better_model
                res["direction"] = f"{better_model} is better"
                results.append(res)
                
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by="p_value").reset_index(drop=True)
        return df

    def generate_significance_table_latex(self, df: pd.DataFrame, target_name: str) -> str:
        """Generates a LaTeX table for Wilcoxon pairwise results."""
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{Pairwise Wilcoxon Signed-Rank Test Results on 5-Fold CV RMSE ({target_name.capitalize()})}}",
            f"\\label{{tab:wilcoxon_{target_name}}}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "\\textbf{Model A} & \\textbf{Model B} & \\textbf{Mean A} & \\textbf{Mean B} & \\textbf{Statistic} & \\textbf{p-value} \\\\",
            "\\midrule"
        ]
        
        for _, row in df.iterrows():
            ma = row["model_a"].replace("_", " ").title()
            mb = row["model_b"].replace("_", " ").title()
            
            p_val = row["p_value"]
            # Formatting p-value with asterisks for significance
            if p_val < 0.05:
                p_str = f"\\textbf{{{p_val:.4f}*}}"
            else:
                p_str = f"{p_val:.4f}"
                
            mean_a = f"{row['mean_a']:.4f}"
            mean_b = f"{row['mean_b']:.4f}"
            stat = f"{row['statistic']:.1f}"
            
            latex_lines.append(f"{ma} & {mb} & {mean_a} & {mean_b} & {stat} & {p_str} \\\\")
            
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\footnotesize{* p < 0.05 indicates statistical significance. Cost criteria (RMSE) analyzed fold-by-fold.}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)
