import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.optimization.topsis import TOPSIS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Sensitivity")

class TOPSISSensitivityAnalyzer:
    """
    Tests stability of TOPSIS ranking across weight scenarios.
    Demonstrates the robustness of the MCDM recommendations.
    """

    WEIGHT_SCENARIOS = {
        "balanced":          [0.333, 0.333, 0.334],
        "energy_priority":   [0.600, 0.200, 0.200],
        "quality_priority":  [0.200, 0.600, 0.200],
        "time_priority":     [0.200, 0.200, 0.600],
        "energy_time":       [0.450, 0.100, 0.450],
        "default":           [0.500, 0.200, 0.300]
    }

    def __init__(
        self,
        pareto_F: np.ndarray,
        pareto_X: np.ndarray,
        decision_space: dict
    ):
        self.pareto_F = np.array(pareto_F, dtype=float)
        self.pareto_X = np.array(pareto_X, dtype=float)
        self.decision_space = decision_space
        self.results = None

    def run_all_scenarios(self) -> pd.DataFrame:
        """Runs TOPSIS for each weight scenario and compiles the results."""
        rows = []
        
        for name, weights in self.WEIGHT_SCENARIOS.items():
            topsis = TOPSIS(
                weights=np.array(weights),
                benefit_criteria=[False, False, False],
                criteria_names=["energy_sec", "roughness_ra", "time_s"]
            )
            topsis.fit(self.pareto_F)
            
            best_sol = topsis.get_best_solution()
            best_idx = best_sol["solution_idx"]
            
            # Extract decision variables
            decision_vars = {}
            for i, key in enumerate(self.decision_space.keys()):
                decision_vars[key] = float(self.pareto_X[best_idx, i])
                
            rows.append({
                "scenario": name,
                "w_energy": weights[0],
                "w_roughness": weights[1],
                "w_time": weights[2],
                "rank1_solution_idx": best_idx,
                "rank1_energy": best_sol["energy_sec"],
                "rank1_roughness": best_sol["roughness_ra"],
                "rank1_time": best_sol["time_s"],
                "rank1_closeness": best_sol["closeness"],
                "rank1_decision_vars": json.dumps(decision_vars)
            })
            
        self.results = pd.DataFrame(rows)
        return self.results

    def compute_rank_stability(self) -> dict:
        """Computes stability metrics for the top solutions across weight scenarios."""
        if self.results is None:
            self.run_all_scenarios()
            
        # Count occurrences of solutions in rank 1
        best_indices = self.results["rank1_solution_idx"].tolist()
        unique_indices, counts = np.unique(best_indices, return_counts=True)
        
        most_stable_idx = int(unique_indices[np.argmax(counts)])
        stable_count = int(np.max(counts))
        total_scenarios = len(self.WEIGHT_SCENARIOS)
        stability_score = stable_count / total_scenarios
        
        return {
            "most_stable_solution_idx": most_stable_idx,
            "appears_in_rank1_count": stable_count,
            "total_scenarios": total_scenarios,
            "stability_score": stability_score
        }

    def plot_sensitivity_heatmap(self, output_dir: Path) -> str:
        """Generates heatmap showing ranks of top-5 overall solutions across weight scenarios."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.results is None:
            self.run_all_scenarios()
            
        # Let's identify the top 5 solutions across all scenarios
        # We can rank solutions for each scenario and build a matrix: (scenarios x top_solutions)
        all_ranks = []
        solution_set = set()
        
        for name, weights in self.WEIGHT_SCENARIOS.items():
            topsis = TOPSIS(
                weights=np.array(weights),
                benefit_criteria=[False, False, False],
                criteria_names=["energy_sec", "roughness_ra", "time_s"]
            )
            topsis.fit(self.pareto_F)
            ranking = topsis.get_ranking()
            
            # Keep track of top 5 solutions in each scenario
            for idx in ranking.head(5)["solution_idx"]:
                solution_set.add(int(idx))
                
        # Build heatmap matrix
        solutions_list = list(solution_set)
        matrix_data = []
        
        for name, weights in self.WEIGHT_SCENARIOS.items():
            topsis = TOPSIS(
                weights=np.array(weights),
                benefit_criteria=[False, False, False],
                criteria_names=["energy_sec", "roughness_ra", "time_s"]
            )
            topsis.fit(self.pareto_F)
            # Map index to rank
            idx_to_rank = dict(zip(topsis.get_ranking()["solution_idx"], topsis.get_ranking()["rank"]))
            
            row = []
            for sol_idx in solutions_list:
                row.append(idx_to_rank.get(sol_idx, 999))
            matrix_data.append(row)
            
        df_heatmap = pd.DataFrame(matrix_data, index=list(self.WEIGHT_SCENARIOS.keys()), columns=[f"Sol {s}" for s in solutions_list])
        
        # Sort columns by mean rank (ascending = better)
        mean_ranks = df_heatmap.mean(axis=0).sort_values()
        df_heatmap = df_heatmap[mean_ranks.index]
        
        # Take only top 5 columns
        df_heatmap = df_heatmap.iloc[:, :5]
        
        sns.set_theme(style="white")
        plt.figure(figsize=(10, 6))
        
        # Heatmap: lower rank (closer to 1) is greener, higher is redder
        ax = sns.heatmap(df_heatmap, annot=True, fmt="d", cmap="RdYlGn_r", vmin=1, vmax=10, cbar_kws={'label': 'TOPSIS Rank'})
        
        plt.title("TOPSIS Rank Sensitivity to Weight Scenario Variations", fontsize=12, fontweight="bold")
        plt.xlabel("Pareto-Optimal Solution ID", fontsize=11)
        plt.ylabel("Weight Scenario", fontsize=11)
        plt.tight_layout()
        
        save_path = output_dir / "sensitivity_heatmap.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved sensitivity heatmap to {save_path}")
        return str(save_path)

    def plot_sensitivity_parallel_coords(self, output_dir: Path) -> str:
        """Generates line-based parallel coordinates plot comparing weight scenarios."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.results is None:
            self.run_all_scenarios()
            
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        # Build coordinates DataFrame
        coords_data = []
        for idx, row in self.results.iterrows():
            coords_data.append({
                "Scenario": row["scenario"],
                "w_energy": row["w_energy"],
                "w_roughness": row["w_roughness"],
                "w_time": row["w_time"],
                "Energy (J/mm3)": row["rank1_energy"],
                "Roughness (um)": row["rank1_roughness"],
                "Time (s)": row["rank1_time"]
            })
        df_coords = pd.DataFrame(coords_data)
        
        # Plot using pd.plotting
        pd.plotting.parallel_coordinates(df_coords, class_column="Scenario", colormap="viridis", linewidth=2.5)
        
        plt.title("TOPSIS Weight Scenario & Recommended Performance Coordinates", fontsize=12, fontweight="bold")
        plt.ylabel("Value Range", fontsize=11)
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        save_path = output_dir / "sensitivity_parallel.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved sensitivity parallel coordinates plot to {save_path}")
        return str(save_path)

    def generate_sensitivity_table_latex(self) -> str:
        """Generates LaTeX table showing recommended parameters across scenarios."""
        if self.results is None:
            self.run_all_scenarios()
            
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{TOPSIS Decision Sensitivity: Recommended Parameters Across Weight Scenarios}",
            "\\label{tab:topsis_sensitivity}",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Scenario & Feed Rate ($f$) & Depth of Cut ($a_p$) & Spindle Speed ($S$) & Tool Wear ($TCond$) \\\\",
            " & (mm/rev) & (mm) & (rpm) & (mm) \\\\",
            "\\midrule"
        ]
        
        for idx, row in self.results.iterrows():
            name = row["scenario"].replace("_", "\\_")
            decision_vars = json.loads(row["rank1_decision_vars"])
            
            f = f"{decision_vars['feed_rate']:.4f}"
            ap = f"{decision_vars['depth_of_cut']:.4f}"
            S = f"{decision_vars['spindle_speed']:.0f}"
            TCond = f"{decision_vars['tool_condition']:.4f}"
            
            line = f"{name} & {f} & {ap} & {S} & {TCond} \\\\"
            latex_lines.append(line)
            
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        tex_path = Path("outputs/results") / "topsis_sensitivity_table.tex"
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(latex_lines))
            
        logger.info(f"Saved LaTeX sensitivity table to {tex_path}")
        return str(tex_path)
