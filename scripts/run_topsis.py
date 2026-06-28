import os
import sys
import argparse
import logging
import json
import joblib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from rich.console import Console

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.optimization.decision_space import DECISION_SPACE
from src.optimization.surrogate_adapter import SurrogateAdapter
from src.optimization.topsis import TOPSIS
from src.optimization.sensitivity import TOPSISSensitivityAnalyzer
from scripts.run_optimization import compute_target_train_stats, compute_target_train_data
from src.models.model_registry import MODEL_REGISTRY

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunTOPSIS")

console = Console()

def main():
    parser = argparse.ArgumentParser(description="EnergyOptAI — TOPSIS MCDM CLI")
    parser.add_argument("--w-energy", type=float, default=0.5, help="Weight for energy SEC")
    parser.add_argument("--w-roughness", type=float, default=0.2, help="Weight for roughness")
    parser.add_argument("--w-time", type=float, default=0.3, help="Weight for machining time")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top solutions to display")
    parser.add_argument("--sensitivity", action="store_true", default=True, help="Run sensitivity analysis")
    
    args = parser.parse_args()
    
    # Validate weights sum to 1.0
    weights_sum = args.w_energy + args.w_roughness + args.w_time
    if abs(weights_sum - 1.0) > 1e-4:
        console.print(f"[bold red]Error: Weights must sum to 1.0 (got {weights_sum:.4f})[/bold red]")
        sys.exit(1)
        
    opt_dir = Path("outputs/results/optimization")
    rec_dir = Path("outputs/results/recommendations")
    fig_dir = Path("outputs/figures/topsis")
    res_dir = Path("outputs/results")
    
    rec_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    pareto_F_path = opt_dir / "pareto_F.csv"
    pareto_X_path = opt_dir / "pareto_X.csv"
    
    if not pareto_F_path.exists() or not pareto_X_path.exists():
        console.print("[bold red]Error: Pareto files not found. Run optimization script first.[/bold red]")
        sys.exit(1)
        
    # STEP 1 - Load Pareto data
    pareto_F = pd.read_csv(pareto_F_path)
    pareto_X = pd.read_csv(pareto_X_path)
    n_solutions = len(pareto_F)
    console.print(f"[bold green]Loaded {n_solutions} Pareto solutions.[/bold green]")
    
    # STEP 2 - Run TOPSIS
    console.print("[bold blue]Running TOPSIS ranking...[/bold blue]")
    weights_arr = np.array([args.w_energy, args.w_roughness, args.w_time])
    topsis = TOPSIS(
        weights=weights_arr,
        benefit_criteria=[False, False, False],
        criteria_names=["energy_sec", "roughness_ra", "time_s"]
    )
    topsis.fit(pareto_F.values)
    ranking = topsis.get_ranking()
    
    # STEP 3 - Load adapter for baseline comparison
    console.print("[bold blue]Re-initializing SurrogateAdapter for baseline comparison...[/bold blue]")
    model_dir = Path("outputs/models/trained")
    scaler_dir = Path("outputs/models/scalers")
    
    models = {}
    scalers = {}
    encoders = {}
    feature_names = {}
    train_stats = {}
    train_data = {}
    
    for target in ["roughness", "time", "energy"]:
        m_key = "catboost" if target != "energy" else "random_forest"
        suffix = "_sec" if target == "energy" else ""
        models[target] = joblib.load(model_dir / f"{target}_{m_key}{suffix}_final.pkl")
        feature_names[target] = models[target].feature_names
        scalers[target] = joblib.load(scaler_dir / f"{target}_{m_key}_scaler.joblib")
        encoders[target] = joblib.load(scaler_dir / f"{target}_{m_key}_encoders.joblib")
        stats, X_train = compute_target_train_data(target)
        train_stats[target] = stats
        train_data[target] = X_train
        
    adapter = SurrogateAdapter(
        models=models,
        scalers=scalers,
        encoders=encoders,
        feature_names=feature_names,
        train_stats=train_stats,
        decision_space=DECISION_SPACE,
        X_train_data=train_data
    )
    
    baseline = adapter.get_baseline_prediction()
    best_sol = topsis.get_best_solution()
    best_idx = best_sol["solution_idx"]
    best_X = pareto_X.values[best_idx]
    
    # Compute improvements
    energy_imp = ((baseline["energy_sec"] - best_sol["energy_sec"]) / baseline["energy_sec"]) * 100
    roughness_imp = ((baseline["roughness_ra"] - best_sol["roughness_ra"]) / baseline["roughness_ra"]) * 100
    time_imp = ((baseline["time_s"] - best_sol["time_s"]) / baseline["time_s"]) * 100
    
    # STEP 4 - Print final recommendation box (standard ASCII boundaries for safety)
    console.print("\n[bold green]+------------------------------------------------------+[/bold green]")
    console.print("[bold green]|     ENERGYOPTAI - FINAL RECOMMENDATION               |[/bold green]")
    console.print("[bold green]+------------------------------------------------------+[/bold green]")
    console.print(f"|  TOPSIS Rank 1 Solution (ID: {best_idx})              ")
    console.print(f"|  Closeness Coefficient: {best_sol['closeness']:.4f}  ")
    console.print("[bold green]+------------------------------------------------------+[/bold green]")
    console.print("|  RECOMMENDED MACHINING PARAMETERS:                   ")
    console.print(f"|    Feed Rate (f):       {best_X[0]:.4f} mm/rev       ")
    console.print(f"|    Depth of Cut (ap):    {best_X[1]:.4f} mm           ")
    console.print(f"|    Spindle Speed (S):   {best_X[2]:.0f} rpm          ")
    console.print(f"|    Tool Wear (TCond):   {best_X[3]:.4f} mm           ")
    console.print("[bold green]+------------------------------------------------------+[/bold green]")
    console.print("|  PREDICTED PERFORMANCE:                              ")
    console.print(f"|    Energy SEC:   {best_sol['energy_sec']:.4f} J/mm3  ({energy_imp:+.1f}% vs baseline)")
    console.print(f"|    Roughness Ra: {best_sol['roughness_ra']:.4f} um   ({roughness_imp:+.1f}% vs baseline)")
    console.print(f"|    Cycle Time:   {best_sol['time_s']:.4f} s    ({time_imp:+.1f}% vs baseline)")
    console.print("[bold green]+------------------------------------------------------+[/bold green]")
    console.print(f"|  TOPSIS WEIGHTS USED:                                ")
    console.print(f"|    Energy: {args.w_energy:.2f} | Roughness: {args.w_roughness:.2f} | Time: {args.w_time:.2f} ")
    console.print("[bold green]+------------------------------------------------------+[/bold green]\n")
    
    # STEP 5 - Generate plots
    topsis.plot_closeness_distribution(fig_dir)
    topsis.plot_ranking_radar(args.top_k, fig_dir)
    topsis.plot_pareto_with_ranking(pareto_F.values, fig_dir, args.top_k)
    
    # STEP 6 - Sensitivity Analysis
    stability_score = 0.0
    if args.sensitivity:
        console.print("[bold blue]Running TOPSIS Rank Sensitivity Analysis...[/bold blue]")
        sensitivity = TOPSISSensitivityAnalyzer(
            pareto_F=pareto_F.values,
            pareto_X=pareto_X.values,
            decision_space=DECISION_SPACE
        )
        sensitivity.run_all_scenarios()
        stability = sensitivity.compute_rank_stability()
        stability_score = stability["stability_score"]
        
        # Generate plots and tables
        sensitivity.plot_sensitivity_heatmap(fig_dir)
        sensitivity.plot_sensitivity_parallel_coords(fig_dir)
        sensitivity.generate_sensitivity_table_latex()
        
        console.print(f"[bold green]Rank stability: solution {stability['most_stable_solution_idx']} appears as rank 1 in {stability['appears_in_rank1_count']}/{stability['total_scenarios']} weight scenarios (stability score: {stability_score:.2f})[/bold green]")
        
    # STEP 7 - Save recommendations
    rec_json = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "topsis_weights": {
            "energy": args.w_energy,
            "roughness": args.w_roughness,
            "time": args.w_time
        },
        "recommended_parameters": {
            "feed_rate": float(best_X[0]),
            "depth_of_cut": float(best_X[1]),
            "spindle_speed": float(best_X[2]),
            "tool_condition": float(best_X[3])
        },
        "predicted_performance": {
            "energy_sec_j_mm3": float(best_sol["energy_sec"]),
            "roughness_ra_um": float(best_sol["roughness_ra"]),
            "time_s": float(best_sol["time_s"])
        },
        "vs_baseline": {
            "energy_improvement_pct": float(energy_imp),
            "roughness_improvement_pct": float(roughness_imp),
            "time_improvement_pct": float(time_imp)
        },
        "topsis_rank": 1,
        "closeness_coefficient": float(best_sol["closeness"]),
        "pareto_front_size": int(n_solutions),
        "sensitivity_stability_score": float(stability_score)
    }
    
    with open(rec_dir / "final_recommendation.json", 'w', encoding='utf-8') as f:
        json.dump(rec_json, f, indent=4)
        
    # Save CSV
    df_rec = pd.DataFrame([{
        "feed_rate": best_X[0],
        "depth_of_cut": best_X[1],
        "spindle_speed": best_X[2],
        "tool_condition": best_X[3],
        "energy_sec": best_sol["energy_sec"],
        "roughness_ra": best_sol["roughness_ra"],
        "time_s": best_sol["time_s"],
        "closeness": best_sol["closeness"]
    }])
    df_rec.to_csv(rec_dir / "final_recommendation.csv", index=False)
    
    # STEP 8 - Print and save LaTeX table
    latex_table = topsis.to_latex()
    with open(res_dir / "topsis_ranking_table.tex", 'w', encoding='utf-8') as f:
        f.write(latex_table)
        
    console.print("\n[bold cyan]LaTeX TOPSIS Ranking Table Saved.[/bold cyan]")

if __name__ == '__main__':
    main()
