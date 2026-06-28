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
logger = logging.getLogger("TOPSIS")

class TOPSIS:
    """
    Technique for Order Preference by Similarity to Ideal Solution (TOPSIS).
    Implements Hwang & Yoon (1981) vector normalization method.
    """

    def __init__(
        self,
        weights: np.ndarray,
        benefit_criteria: list,
        criteria_names: list
    ):
        self.weights = np.array(weights, dtype=float)
        self.benefit_criteria = benefit_criteria
        self.criteria_names = criteria_names
        
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.ideal_best = None
        self.ideal_worst = None
        self.d_plus = None
        self.d_minus = None
        self.closeness = None
        self.ranks = None
        self.decision_matrix = None
        self.is_fitted = False

    def fit(self, decision_matrix: np.ndarray) -> 'TOPSIS':
        """Computes TOPSIS rankings on the given decision matrix (solutions x criteria)."""
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        n_solutions, n_criteria = self.decision_matrix.shape
        
        # Validate inputs
        assert n_criteria == len(self.weights), "Decision matrix columns must match weights length"
        assert abs(np.sum(self.weights) - 1.0) < 1e-5, f"Weights must sum to 1.0, got {np.sum(self.weights)}"
        assert not np.any(np.isnan(self.decision_matrix)), "Decision matrix contains NaN values"

        # 1. Normalize (vector normalization)
        column_norms = np.sqrt(np.sum(self.decision_matrix**2, axis=0))
        # Avoid division by zero
        column_norms[column_norms == 0] = 1.0
        self.normalized_matrix = self.decision_matrix / column_norms

        # 2. Apply weights
        self.weighted_matrix = self.normalized_matrix * self.weights

        # 3. Determine ideal best (A+) and ideal worst (A-)
        self.ideal_best = np.zeros(n_criteria)
        self.ideal_worst = np.zeros(n_criteria)
        
        for j in range(n_criteria):
            is_benefit = self.benefit_criteria[j]
            col_vals = self.weighted_matrix[:, j]
            if is_benefit:
                self.ideal_best[j] = np.max(col_vals)
                self.ideal_worst[j] = np.min(col_vals)
            else:
                # Cost criterion (minimize)
                self.ideal_best[j] = np.min(col_vals)
                self.ideal_worst[j] = np.max(col_vals)

        # 4. Compute Euclidean distances to A+ and A-
        self.d_plus = np.sqrt(np.sum((self.weighted_matrix - self.ideal_best)**2, axis=1))
        self.d_minus = np.sqrt(np.sum((self.weighted_matrix - self.ideal_worst)**2, axis=1))

        # 5. Compute closeness coefficient C
        denom = self.d_plus + self.d_minus
        denom[denom == 0] = 1e-12
        self.closeness = self.d_minus / denom

        # 6. Rank by closeness descending (higher C = better)
        # scipy.stats.rankdata equivalent:
        # Sort indices of closeness in descending order to assign ranks
        sorted_indices = np.argsort(-self.closeness)
        self.ranks = np.zeros(n_solutions, dtype=int)
        for rank, idx in enumerate(sorted_indices):
            self.ranks[idx] = rank + 1

        self.is_fitted = True
        logger.info(f"TOPSIS fit completed for {n_solutions} solutions.")
        return self

    def get_ranking(self) -> pd.DataFrame:
        """Returns the full ranking table as a DataFrame sorted by rank ascending."""
        if not self.is_fitted:
            raise ValueError("TOPSIS model is not fitted yet. Call fit() first.")
            
        data = {
            "rank": self.ranks,
            "solution_idx": np.arange(len(self.ranks)),
            "closeness": self.closeness,
            "d_plus": self.d_plus,
            "d_minus": self.d_minus
        }
        
        # Add original criteria columns
        for j, name in enumerate(self.criteria_names):
            data[name] = self.decision_matrix[:, j]
            
        df = pd.DataFrame(data)
        return df.sort_values("rank").reset_index(drop=True)

    def get_best_solution(self) -> dict:
        """Returns the rank-1 solution as a dictionary."""
        df = self.get_ranking()
        row = df.iloc[0]
        res = {
            "rank": 1,
            "solution_idx": int(row["solution_idx"]),
            "closeness": float(row["closeness"])
        }
        for name in self.criteria_names:
            res[name] = float(row[name])
        return res

    def get_top_k(self, k: int = 5) -> pd.DataFrame:
        """Returns top-k solutions."""
        return self.get_ranking().head(k)

    def to_latex(self) -> str:
        """Returns LaTeX table of top-10 ranked solutions in booktabs style."""
        df_top = self.get_ranking().head(10)
        
        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{TOPSIS Multi-Criteria Decision Making (MCDM) Ranking of Pareto Solutions}",
            "\\label{tab:topsis_ranking}",
            "\\begin{tabular}{cccccc}",
            "\\toprule",
            "Rank & Solution ID & Closeness ($C_i$) & " + " & ".join(self.criteria_names) + " \\\\",
            "\\midrule"
        ]
        
        for idx, row in df_top.iterrows():
            r = int(row["rank"])
            sol_id = int(row["solution_idx"])
            c_val = f"{row['closeness']:.4f}"
            
            # Format criteria
            vals = []
            for name in self.criteria_names:
                vals.append(f"{row[name]:.4f}")
                
            line = f"{r} & {sol_id} & {c_val} & " + " & ".join(vals) + " \\\\"
            if r == 1:
                # Bold the best solution row
                line = f"\\textbf{{{r}}} & \\textbf{{{sol_id}}} & \\textbf{{{c_val}}} & " + " & ".join([f"\\textbf{{{v}}}" for v in vals]) + " \\\\"
            latex_lines.append(line)
            
        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(latex_lines)

    def plot_closeness_distribution(self, output_dir: Path, figsize=(10, 5)) -> str:
        """Saves a histogram of closeness coefficient values across solutions."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        
        sns.histplot(self.closeness, bins=15, kde=True, color="purple")
        
        # Highlight best solution
        best_c = np.max(self.closeness)
        plt.axvline(best_c, color="gold", linestyle="--", linewidth=2.5, label=f"Rank-1 Solution (C={best_c:.4f})")
        
        plt.title("TOPSIS Closeness Coefficient Distribution Across Pareto Front", fontsize=12, fontweight="bold")
        plt.xlabel("Closeness Coefficient (Ci)", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.legend()
        plt.tight_layout()
        
        save_path = output_dir / "closeness_distribution.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved closeness distribution plot to {save_path}")
        return str(save_path)

    def plot_ranking_radar(self, top_k: int = 5, output_dir: Path = None, figsize=(10, 8)) -> str:
        """Radar chart comparing top-k solutions across objectives."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_ranking = self.get_ranking().head(top_k)
        
        # Normalize criteria to [0, 1] range where 0 is best (cost criteria)
        # Since all criteria are cost, we normalize such that 0 is minimum (best) and 1 is maximum (worst)
        n_crit = len(self.criteria_names)
        angles = np.linspace(0, 2 * np.pi, n_crit, endpoint=False).tolist()
        angles += angles[:1]  # close loop
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Find min/max for scaling
        min_vals = np.min(self.decision_matrix, axis=0)
        max_vals = np.max(self.decision_matrix, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0
        
        colors = sns.color_palette("muted", top_k)
        
        for idx, row in df_ranking.iterrows():
            rank = int(row["rank"])
            sol_idx = int(row["solution_idx"])
            
            # Extract criteria values and normalize to [0, 1] (0 = best, 1 = worst)
            vals = []
            for j, name in enumerate(self.criteria_names):
                v = row[name]
                norm_v = (v - min_vals[j]) / ranges[j]
                vals.append(norm_v)
            vals += vals[:1]  # close loop
            
            linewidth = 3.0 if rank == 1 else 1.5
            linestyle = "-" if rank == 1 else "--"
            label = f"Rank {rank} (Sol {sol_idx}): " + ("Best Balance" if rank == 1 else f"Option {rank}")
            
            ax.plot(angles, vals, color=colors[idx], linewidth=linewidth, linestyle=linestyle, label=label)
            ax.fill(angles, vals, color=colors[idx], alpha=0.1)
            
        # Draw labels
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Criteria names on axes
        plt.xticks(angles[:-1], self.criteria_names, fontsize=10, fontweight="bold")
        
        # Y labels
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["Best (0.0)", "0.5", "0.75", "Worst (1.0)"], color="grey", fontsize=8)
        plt.ylim(0, 1)
        
        plt.title("Radar Chart Comparison of Top TOPSIS Solutions\n(Values normalized: 0.0=Best, 1.0=Worst)", y=1.08, fontsize=12, fontweight="bold")
        plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        
        save_path = output_dir / "ranking_radar.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved ranking radar plot to {save_path}")
        return str(save_path)

    def plot_pareto_with_ranking(self, pareto_F: np.ndarray, output_dir: Path, top_k: int = 5) -> str:
        """2D scatter plot showing the Pareto front colored by TOPSIS closeness."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        xs = pareto_F[:, 0]  # Energy SEC
        ys = pareto_F[:, 1]  # Roughness
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 8))
        
        # Scatter colored by closeness
        sc = plt.scatter(xs, ys, c=self.closeness, cmap="viridis", s=60, alpha=0.9, edgecolors="none")
        cbar = plt.colorbar(sc)
        cbar.set_label("TOPSIS Closeness Coefficient (Ci)", fontsize=11)
        
        # Highlight top_k
        df_ranking = self.get_ranking().head(top_k)
        for idx, row in df_ranking.iterrows():
            rank = int(row["rank"])
            sol_idx = int(row["solution_idx"])
            sol_x = row[self.criteria_names[0]]
            sol_y = row[self.criteria_names[1]]
            
            if rank == 1:
                # Gold star for rank 1
                plt.scatter(sol_x, sol_y, color="gold", marker="*", s=250, edgecolors="black", zorder=5, label="Rank-1 Best Compromise")
                plt.annotate(f"Rank 1", (sol_x, sol_y), textcoords="offset points", xytext=(0, 12), ha='center', fontweight='bold', fontsize=10)
            else:
                plt.scatter(sol_x, sol_y, color="red", marker="o", s=100, edgecolors="black", zorder=4)
                plt.annotate(f"Rank {rank}", (sol_x, sol_y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
                
        plt.title("Pareto Front colored by TOPSIS Closeness (Energy vs Roughness)", fontsize=12, fontweight="bold")
        plt.xlabel("Energy SEC (J/mm3)", fontsize=11)
        plt.ylabel("Surface Roughness Ra (um)", fontsize=11)
        plt.legend(loc="upper right")
        plt.tight_layout()
        
        save_path = output_dir / "pareto_with_ranking.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved Pareto with ranking plot to {save_path}")
        return str(save_path)
