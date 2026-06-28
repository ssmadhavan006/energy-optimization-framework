import logging
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.indicators.hv import HV

from src.optimization.surrogate_adapter import SurrogateAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NSGA2Optimizer")

class EnergyOptNSGA2:
    """
    NSGA-II multi-objective optimizer for CNC machining parameters.
    Uses ML surrogate models as objective functions.
    """

    def __init__(
        self,
        adapter: SurrogateAdapter,
        decision_space: dict,
        pop_size: int = 100,
        n_gen: int = 200,
        seed: int = 42,
        verbose: bool = True,
        save_history: bool = True,
        tool_condition_fixed: float = 0.053
    ):
        self.adapter = adapter
        self.decision_space = decision_space
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.verbose = verbose
        self.save_history = save_history
        self.tool_condition_fixed = tool_condition_fixed
        
        self.problem = None
        self.algorithm = None
        self.result = None
        
        self.pareto_X = None
        self.pareto_F = None
        self.hypervolume = 0.0
        self.n_pareto_solutions = 0
        self.convergence_history = []
        self.runtime_seconds = 0.0

    def _build_pymoo_problem(self) -> Problem:
        """Creates a pymoo Problem subclass wrapping SurrogateAdapter."""
        optimized_keys = [k for k in self.decision_space.keys() if k != "tool_condition"]
        n_var = len(optimized_keys)
        
        xl = np.array([self.decision_space[k]["bounds"][0] for k in optimized_keys])
        xu = np.array([self.decision_space[k]["bounds"][1] for k in optimized_keys])
        
        adapter = self.adapter
        t_cond_val = self.tool_condition_fixed
        
        class CNCMachiningProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=n_var,
                    n_obj=3,
                    n_constr=0,
                    xl=xl,
                    xu=xu,
                    elementwise_evaluation=False
                )
                
            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = adapter.predict_batch(x, t_cond_val)

        self.problem = CNCMachiningProblem()
        return self.problem

    def _build_algorithm(self) -> NSGA2:
        """Configures NSGA-II algorithm with standard parameters (Deb et al., 2002)."""
        self.algorithm = NSGA2(
            pop_size=self.pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        return self.algorithm

    def run(self) -> dict:
        """Executes NSGA-II optimization and computes hypervolume."""
        logger.info(f"Initializing NSGA-II problem (pop_size={self.pop_size}, generations={self.n_gen})...")
        self._build_pymoo_problem()
        self._build_algorithm()
        
        termination = get_termination("n_gen", self.n_gen)
        
        start_time = time.time()
        
        self.result = minimize(
            self.problem,
            self.algorithm,
            termination,
            seed=self.seed,
            save_history=self.save_history,
            verbose=self.verbose
        )
        
        self.runtime_seconds = time.time() - start_time
        
        X_res = self.result.X
        if X_res is not None:
            X_res_2d = np.atleast_2d(X_res)
            t_cond_col = np.full((X_res_2d.shape[0], 1), self.tool_condition_fixed)
            self.pareto_X = np.hstack([X_res_2d, t_cond_col])
        else:
            self.pareto_X = None
            
        self.pareto_F = self.result.F
        self.n_pareto_solutions = len(self.pareto_X) if self.pareto_X is not None else 0
        
        # Calculate hypervolume convergence history if history was saved
        self.convergence_history = []
        if self.save_history and self.result.history is not None:
            # We need a fixed reference point to calculate hypervolume consistently across generations
            # Let's use 1.1 * max observed objectives in the final Pareto front
            ref_point = 1.1 * np.max(self.pareto_F, axis=0)
            hv_indicator = HV(ref_point=ref_point)
            
            for gen_idx, entry in enumerate(self.result.history):
                # Extract Pareto front at this generation
                opt_F = entry.opt.get("F")
                if opt_F is not None and len(opt_F) > 0:
                    val = hv_indicator(opt_F)
                    self.convergence_history.append({"generation": gen_idx + 1, "hypervolume": float(val)})
            
            # Compute final hypervolume
            self.hypervolume = hv_indicator(self.pareto_F)
        else:
            # Fallback hypervolume
            ref_point = 1.1 * np.max(self.pareto_F, axis=0)
            self.hypervolume = HV(ref_point=ref_point)(self.pareto_F)
            
        logger.info(f"NSGA-II completed in {self.runtime_seconds:.2f} seconds.")
        logger.info(f"Pareto front size: {self.n_pareto_solutions} solutions")
        logger.info(f"Final Hypervolume: {self.hypervolume:.4f}")
        
        return {
            "pareto_X": self.pareto_X,
            "pareto_F": self.pareto_F,
            "hypervolume": self.hypervolume,
            "n_solutions": self.n_pareto_solutions,
            "convergence_history": self.convergence_history,
            "runtime_seconds": self.runtime_seconds
        }

    def save_results(self, output_dir: Path) -> dict:
        """Saves Pareto front coordinates, objectives, history, and summary configuration."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save X coordinates
        var_names = list(self.decision_space.keys())
        df_X = pd.DataFrame(self.pareto_X, columns=var_names)
        x_path = output_dir / "pareto_X.csv"
        df_X.to_csv(x_path, index=False)
        
        # Save F objectives
        obj_names = ["energy_sec", "roughness_ra", "time_s"]
        df_F = pd.DataFrame(self.pareto_F, columns=obj_names)
        f_path = output_dir / "pareto_F.csv"
        df_F.to_csv(f_path, index=False)
        
        # Save convergence history
        conv_path = output_dir / "convergence.csv"
        if self.convergence_history:
            df_conv = pd.DataFrame(self.convergence_history)
            df_conv.to_csv(conv_path, index=False)
            
        # Get baseline prediction
        baseline = self.adapter.get_baseline_prediction()
        
        # Calculate maximum improvement vs baseline across all solutions
        best_energy_saving = ((baseline["energy_sec"] - np.min(self.pareto_F[:, 0])) / baseline["energy_sec"]) * 100
        best_roughness_saving = ((baseline["roughness_ra"] - np.min(self.pareto_F[:, 1])) / baseline["roughness_ra"]) * 100
        best_time_saving = ((baseline["time_s"] - np.min(self.pareto_F[:, 2])) / baseline["time_s"]) * 100
        
        summary = {
            "hypervolume": float(self.hypervolume),
            "n_solutions": int(self.n_pareto_solutions),
            "pop_size": int(self.pop_size),
            "n_gen": int(self.n_gen),
            "runtime_seconds": float(self.runtime_seconds),
            "objective_ranges": {
                "energy_sec": [float(np.min(self.pareto_F[:, 0])), float(np.max(self.pareto_F[:, 0]))],
                "roughness_ra": [float(np.min(self.pareto_F[:, 1])), float(np.max(self.pareto_F[:, 1]))],
                "time_s": [float(np.min(self.pareto_F[:, 2])), float(np.max(self.pareto_F[:, 2]))]
            },
            "baseline_comparison": {
                "baseline": baseline,
                "max_reductions": {
                    "energy_sec_pct": float(best_energy_saving),
                    "roughness_ra_pct": float(best_roughness_saving),
                    "time_s_pct": float(best_time_saving)
                }
            }
        }
        
        summary_path = output_dir / "nsga2_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
            
        logger.info(f"NSGA-II results exported successfully to {output_dir}")
        return {
            "pareto_X": str(x_path),
            "pareto_F": str(f_path),
            "convergence": str(conv_path),
            "summary": str(summary_path)
        }

    def plot_pareto_3d(self, output_dir: Path) -> str:
        """Generates 3D scatter plot of the Pareto front (interactive HTML + static PNG)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        xs = self.pareto_F[:, 0]  # Energy SEC
        ys = self.pareto_F[:, 1]  # Roughness Ra
        zs = self.pareto_F[:, 2]  # Time
        
        baseline = self.adapter.get_baseline_prediction()
        
        # 1. Interactive Plotly HTML version
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=xs,
                        colorscale='Viridis',
                        colorbar=dict(title="Energy SEC (J/mm³)"),
                        opacity=0.8
                    ),
                    name="Pareto Solutions"
                ),
                go.Scatter3d(
                    x=[baseline["energy_sec"]],
                    y=[baseline["roughness_ra"]],
                    z=[baseline["time_s"]],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='star'
                    ),
                    name="Baseline Configuration"
                )
            ])
            
            fig.update_layout(
                title="NSGA-II 3D Pareto Front",
                scene=dict(
                    xaxis_title="Energy SEC (J/mm³)",
                    yaxis_title="Surface Roughness Ra (μm)",
                    zaxis_title="Machining Time (s)"
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )
            html_path = output_dir / "pareto_3d.html"
            fig.write_html(str(html_path))
            logger.info(f"Saved interactive 3D Pareto HTML to {html_path}")
        except Exception as e:
            logger.warning(f"Could not generate interactive Plotly plot: {e}")
            
        # 2. Static Matplotlib PNG version
        sns.set_theme(style="white")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        sc = ax.scatter(xs, ys, zs, c=xs, cmap='viridis', s=40, alpha=0.8, edgecolors='none')
        cbar = plt.colorbar(sc, pad=0.1)
        cbar.set_label("Energy SEC (J/mm³)", fontsize=10)
        
        # Plot baseline
        ax.scatter(
            [baseline["energy_sec"]],
            [baseline["roughness_ra"]],
            [baseline["time_s"]],
            color='red',
            marker='*',
            s=150,
            label="Current Average"
        )
        
        ax.set_title("3D Pareto Front (Energy SEC vs Roughness vs Time)", fontsize=12, fontweight='bold')
        ax.set_xlabel("Energy SEC (J/mm³)", fontsize=10)
        ax.set_ylabel("Surface Roughness Ra (μm)", fontsize=10)
        ax.set_zlabel("Machining Time (s)", fontsize=10)
        ax.legend()
        
        plt.tight_layout()
        save_path = output_dir / "pareto_3d_static.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved static 3D Pareto plot to {save_path}")
        return str(save_path)

    def plot_pareto_2d_projections(self, output_dir: Path) -> str:
        """Generates 2D projection scatter subplots showing Pareto trade-offs and baseline comparison."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        xs = self.pareto_F[:, 0]  # Energy SEC
        ys = self.pareto_F[:, 1]  # Roughness
        zs = self.pareto_F[:, 2]  # Time
        
        baseline = self.adapter.get_baseline_prediction()
        
        # Identify knee point (balanced solution)
        # Normalize objectives first to [0, 1] range for distance calculation
        f_min = np.min(self.pareto_F, axis=0)
        f_max = np.max(self.pareto_F, axis=0)
        f_range = f_max - f_min
        # Handle zero division
        f_range[f_range == 0] = 1.0
        
        normalized_F = (self.pareto_F - f_min) / f_range
        # Distance to origin
        dists = np.linalg.norm(normalized_F, axis=1)
        knee_idx = int(np.argmin(dists))
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Subplot A: Energy vs Roughness
        axes[0].scatter(xs, ys, color='steelblue', alpha=0.7, edgecolors='none', label='Pareto Front')
        axes[0].scatter([baseline["energy_sec"]], [baseline["roughness_ra"]], color='red', marker='*', s=150, label='Baseline')
        axes[0].scatter([xs[knee_idx]], [ys[knee_idx]], color='gold', marker='o', s=100, edgecolors='black', label='Knee Point')
        axes[0].set_xlabel("Energy SEC (J/mm³)", fontsize=11)
        axes[0].set_ylabel("Surface Roughness Ra (μm)", fontsize=11)
        axes[0].set_title("Energy vs Surface Roughness", fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # Subplot B: Energy vs Machining Time
        axes[1].scatter(xs, zs, color='steelblue', alpha=0.7, edgecolors='none', label='Pareto Front')
        axes[1].scatter([baseline["energy_sec"]], [baseline["time_s"]], color='red', marker='*', s=150, label='Baseline')
        axes[1].scatter([xs[knee_idx]], [zs[knee_idx]], color='gold', marker='o', s=100, edgecolors='black', label='Knee Point')
        axes[1].set_xlabel("Energy SEC (J/mm³)", fontsize=11)
        axes[1].set_ylabel("Machining Time (s)", fontsize=11)
        axes[1].set_title("Energy vs Machining Time", fontsize=12, fontweight='bold')
        axes[1].legend()
        
        # Subplot C: Roughness vs Machining Time
        axes[2].scatter(ys, zs, color='steelblue', alpha=0.7, edgecolors='none', label='Pareto Front')
        axes[2].scatter([baseline["roughness_ra"]], [baseline["time_s"]], color='red', marker='*', s=150, label='Baseline')
        axes[2].scatter([ys[knee_idx]], [zs[knee_idx]], color='gold', marker='o', s=100, edgecolors='black', label='Knee Point')
        axes[2].set_xlabel("Surface Roughness Ra (μm)", fontsize=11)
        axes[2].set_ylabel("Machining Time (s)", fontsize=11)
        axes[2].set_title("Surface Roughness vs Machining Time", fontsize=12, fontweight='bold')
        axes[2].legend()
        
        plt.suptitle("Pareto Front 2D Projections", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = output_dir / "pareto_2d_projections.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved 2D projections plot to {save_path}")
        return str(save_path)

    def plot_convergence(self, output_dir: Path) -> str:
        """Generates convergence plot of the Hypervolume indicator over generations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.convergence_history:
            logger.warning("No convergence history found. Skipping convergence plot.")
            return ""
            
        df_conv = pd.DataFrame(self.convergence_history)
        
        # Find convergence generation (improvement < 0.1% per generation)
        conv_gen = self.n_gen
        final_hv = self.hypervolume
        for idx in range(1, len(df_conv)):
            prev_hv = df_conv.iloc[idx-1]["hypervolume"]
            curr_hv = df_conv.iloc[idx]["hypervolume"]
            if prev_hv > 0:
                imp = (curr_hv - prev_hv) / prev_hv
                if imp < 0.001 and conv_gen == self.n_gen:
                    # check if it remains stable for next 5 generations
                    stable = True
                    for k in range(1, min(6, len(df_conv) - idx)):
                        next_hv = df_conv.iloc[idx + k]["hypervolume"]
                        if (next_hv - curr_hv) / curr_hv >= 0.001:
                            stable = False
                            break
                    if stable:
                        conv_gen = int(df_conv.iloc[idx]["generation"])
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))
        
        plt.plot(df_conv["generation"], df_conv["hypervolume"], color="#d35400", linewidth=2.5, label="Hypervolume")
        plt.axhline(final_hv, color='gray', linestyle='--', alpha=0.7, label=f"Final HV ({final_hv:.4f})")
        
        if conv_gen < self.n_gen:
            plt.axvline(conv_gen, color='green', linestyle=':', alpha=0.7, label=f"Convergence Gen ({conv_gen})")
            
        plt.title("NSGA-II Hypervolume Convergence History", fontsize=12, fontweight='bold')
        plt.xlabel("Generation Number", fontsize=11)
        plt.ylabel("Hypervolume Indicator", fontsize=11)
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        save_path = output_dir / "nsga2_convergence.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved convergence plot to {save_path}")
        return str(save_path)

    def plot_objective_distributions(self, output_dir: Path) -> str:
        """Generates subplots showing distribution of objective values across all Pareto solutions."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        xs = self.pareto_F[:, 0]  # Energy SEC
        ys = self.pareto_F[:, 1]  # Roughness
        zs = self.pareto_F[:, 2]  # Time
        
        baseline = self.adapter.get_baseline_prediction()
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Energy distribution
        sns.histplot(xs, kde=True, color="#2ecc71", ax=axes[0])
        axes[0].axvline(baseline["energy_sec"], color="red", linestyle="--", linewidth=2, label="Baseline")
        axes[0].set_xlabel("Energy SEC (J/mm³)", fontsize=11)
        axes[0].set_title("SEC Distribution", fontsize=12, fontweight="bold")
        axes[0].legend()
        
        # Roughness distribution
        sns.histplot(ys, kde=True, color="#34495e", ax=axes[1])
        axes[1].axvline(baseline["roughness_ra"], color="red", linestyle="--", linewidth=2, label="Baseline")
        axes[1].set_xlabel("Roughness Ra (μm)", fontsize=11)
        axes[1].set_title("Roughness Distribution", fontsize=12, fontweight="bold")
        axes[1].legend()
        
        # Time distribution
        sns.histplot(zs, kde=True, color="#e74c3c", ax=axes[2])
        axes[2].axvline(baseline["time_s"], color="red", linestyle="--", linewidth=2, label="Baseline")
        axes[2].set_xlabel("Machining Time (s)", fontsize=11)
        axes[2].set_title("Machining Time Distribution", fontsize=12, fontweight="bold")
        axes[2].legend()
        
        plt.suptitle("Pareto Front Objective Distributions", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        save_path = output_dir / "pareto_objective_distributions.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved objective distributions plot to {save_path}")
        return str(save_path)

    def plot_decision_variable_distributions(self, output_dir: Path) -> str:
        """Generates box plots showing the ranges of each decision variable across Pareto solutions."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_X = pd.DataFrame(self.pareto_X, columns=list(self.decision_space.keys()))
        
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, len(self.decision_space), figsize=(16, 5))
        
        for i, (key, val) in enumerate(self.decision_space.items()):
            sns.boxplot(y=df_X[key], color="lightblue", ax=axes[i], width=0.4)
            axes[i].set_title(f"{key} ({val['unit']})", fontsize=12, fontweight="bold")
            axes[i].set_ylabel("")
            # Set bounds
            axes[i].set_ylim(val["bounds"][0]*0.9, val["bounds"][1]*1.1)
            
        plt.suptitle("Decision Variable Distributions Across Pareto Front", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        
        save_path = output_dir / "pareto_variable_ranges.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved decision variable distributions plot to {save_path}")
        return str(save_path)
