import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import MODEL_REGISTRY, TARGET_CONFIG
from src.training.train_pipeline import TrainPipeline
from src.optimization.decision_space import DECISION_SPACE
from src.optimization.surrogate_adapter import SurrogateAdapter
from src.optimization.nsga2_optimizer import EnergyOptNSGA2

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunOptimization")

console = Console()

def compute_target_train_stats(target: str) -> dict:
    """Gets the unscaled training feature stats for a target variable using inverse scaling."""
    logger.info(f"Extracting training stats for target '{target}'...")
    model_cls = MODEL_REGISTRY["catboost"] if target != "energy" else MODEL_REGISTRY["random_forest"]
    save_suffix = "_sec" if target == "energy" else ""
    
    # Configure TARGET_CONFIG for energy to sec
    if target == "energy":
        TARGET_CONFIG["energy"]["target_col"] = "sec"
        TARGET_CONFIG["energy"]["unit"] = "J/mm³"
        
    pipeline = TrainPipeline(model_class=model_cls, target_name=target, run_tuning=False, save_suffix=save_suffix)
    X_train, _, _, _ = pipeline.prepare_data()
    
    # Load scaler to perform inverse transform and obtain original unscaled bounds
    model_name = model_cls(target_name=target).model_name
    scaler_path = Path("outputs/models/scalers") / f"{target}_{model_name}_scaler.joblib"
    scaler = joblib.load(scaler_path)
    
    unscaled_arr = scaler.inverse_transform(X_train)
    df_unscaled = pd.DataFrame(unscaled_arr, columns=X_train.columns)
    
    stats = {}
    for col in df_unscaled.columns:
        stats[col] = {
            "min": float(df_unscaled[col].min()),
            "max": float(df_unscaled[col].max()),
            "median": float(df_unscaled[col].median())
        }
    return stats

def compute_target_train_data(target: str) -> tuple:
    """Gets both unscaled training stats and scaled training features DataFrame."""
    logger.info(f"Extracting training data and stats for target '{target}'...")
    model_cls = MODEL_REGISTRY["catboost"] if target != "energy" else MODEL_REGISTRY["random_forest"]
    save_suffix = "_sec" if target == "energy" else ""
    
    if target == "energy":
        TARGET_CONFIG["energy"]["target_col"] = "sec"
        TARGET_CONFIG["energy"]["unit"] = "J/mm³"
        
    pipeline = TrainPipeline(model_class=model_cls, target_name=target, run_tuning=False, save_suffix=save_suffix)
    X_train, _, _, _ = pipeline.prepare_data()
    
    # Load scaler to perform inverse transform and obtain original unscaled bounds
    model_name = model_cls(target_name=target).model_name
    scaler_path = Path("outputs/models/scalers") / f"{target}_{model_name}_scaler.joblib"
    scaler = joblib.load(scaler_path)
    
    unscaled_arr = scaler.inverse_transform(X_train)
    df_unscaled = pd.DataFrame(unscaled_arr, columns=X_train.columns)
    
    stats = {}
    for col in df_unscaled.columns:
        stats[col] = {
            "min": float(df_unscaled[col].min()),
            "max": float(df_unscaled[col].max()),
            "median": float(df_unscaled[col].median())
        }
    return stats, X_train

def main():
    parser = argparse.ArgumentParser(description="EnergyOptAI — NSGA-II Optimization CLI")
    parser.add_argument("--pop", type=int, default=100, help="Population size")
    parser.add_argument("--gen", type=int, default=200, help="Number of generations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--validate-only", action="store_true", help="Runs validation only")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--tool-condition", type=str, default="mid",
                        help="Tool condition scenario: new, mid, worn or float value")
    
    args = parser.parse_args()
    
    # Map scenario keys to float values
    scenarios = DECISION_SPACE["tool_condition"]["scenarios"]
    tool_condition_val = 0.053
    if args.tool_condition == "new":
        tool_condition_val = scenarios["new_tool"]
        tool_condition_label = "new_tool"
    elif args.tool_condition == "mid":
        tool_condition_val = scenarios["mid_life"]
        tool_condition_label = "mid_life"
    elif args.tool_condition == "worn":
        tool_condition_val = scenarios["worn_tool"]
        tool_condition_label = "worn_tool"
    else:
        try:
            tool_condition_val = float(args.tool_condition)
            tool_condition_label = f"custom_{tool_condition_val}"
        except ValueError:
            console.print(f"[bold red]Invalid tool condition value: {args.tool_condition}. Expected new, mid, worn, or float.[/bold red]")
            sys.exit(1)
            
    console.print(f"Tool Condition fixed at: {tool_condition_val:.4f} mm ({tool_condition_label} state)")
    console.print("Optimizing over: feed_rate, depth_of_cut, spindle_speed")
    
    model_dir = Path("outputs/models/trained")
    scaler_dir = Path("outputs/models/scalers")
    output_dir = Path("outputs/results/optimization")
    fig_dir = Path("outputs/figures/pareto")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold blue]=== Load Champions, Scalers, and Encoders ===[/bold blue]")
    
    models = {}
    scalers = {}
    encoders = {}
    feature_names = {}
    train_stats = {}
    train_data = {}
    
    targets = ["roughness", "time", "energy"]
    
    for target in targets:
        m_key = "catboost" if target != "energy" else "random_forest"
        suffix = "_sec" if target == "energy" else ""
        
        # Load model pickle
        model_path = model_dir / f"{target}_{m_key}{suffix}_final.pkl"
        if not model_path.exists():
            console.print(f"[bold red]Trained champion model not found at {model_path}. Run training first.[/bold red]")
            sys.exit(1)
            
        models[target] = joblib.load(model_path)
        feature_names[target] = models[target].feature_names
        
        # Load scaler
        scaler_path = scaler_dir / f"{target}_{m_key}_scaler.joblib"
        scalers[target] = joblib.load(scaler_path)
        
        # Load encoder
        encoder_path = scaler_dir / f"{target}_{m_key}_encoders.joblib"
        encoders[target] = joblib.load(encoder_path)
        
        # Compute training stats dynamically
        stats, X_train = compute_target_train_data(target)
        train_stats[target] = stats
        train_data[target] = X_train
        
    console.print("[bold green]All champion files loaded successfully.[/bold green]")
    
    # Instantiate SurrogateAdapter
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
    
    console.print("\n[bold cyan]BASELINE prediction (Median Parameters):[/bold cyan]")
    console.print(f"  Energy SEC:   {baseline['energy_sec']:.4f} J/mm3")
    console.print(f"  Roughness Ra: {baseline['roughness_ra']:.4f} um")
    console.print(f"  Machining Time: {baseline['time_s']:.4f} s")
    
    if args.validate_only:
        console.print("[bold green]Validation check complete. SurrogateAdapter works correctly.[/bold green]")
        return
        
    # Run NSGA-II Optimization
    console.print(f"\n[bold blue]=== Starting NSGA-II Multi-Objective Optimization ===[/bold blue]")
    optimizer = EnergyOptNSGA2(
        adapter=adapter,
        decision_space=DECISION_SPACE,
        pop_size=args.pop,
        n_gen=args.gen,
        seed=args.seed,
        tool_condition_fixed=tool_condition_val
    )
    
    res = optimizer.run()
    paths = optimizer.save_results(output_dir)
    
    # Plot results
    if not args.no_plots:
        console.print("\n[bold blue]=== Generating Pareto Front Visualizations ===[/bold blue]")
        optimizer.plot_pareto_3d(fig_dir)
        optimizer.plot_pareto_2d_projections(fig_dir)
        optimizer.plot_convergence(fig_dir)
        optimizer.plot_objective_distributions(fig_dir)
        optimizer.plot_decision_variable_distributions(fig_dir)
        
    # Print summary
    min_f = np.min(res["pareto_F"], axis=0)
    max_f = np.max(res["pareto_F"], axis=0)
    
    # Calculate % improvement
    energy_imp = ((baseline["energy_sec"] - min_f[0]) / baseline["energy_sec"]) * 100
    roughness_imp = ((baseline["roughness_ra"] - min_f[1]) / baseline["roughness_ra"]) * 100
    time_imp = ((baseline["time_s"] - min_f[2]) / baseline["time_s"]) * 100
    
    console.print("\n[bold green]=======================================[/bold green]")
    console.print("[bold green]NSGA-II OPTIMIZATION COMPLETE[/bold green]")
    console.print("[bold green]=======================================[/bold green]")
    console.print(f"Population: {args.pop}  |  Generations: {args.gen}")
    console.print(f"Pareto solutions found: {res['n_solutions']}")
    console.print(f"Hypervolume indicator: {res['hypervolume']:.6f}")
    console.print("")
    console.print("OBJECTIVE RANGES ACROSS PARETO FRONT:")
    console.print(f"  Energy SEC (J/mm3): {min_f[0]:.4f} to {max_f[0]:.4f}")
    console.print(f"  Roughness Ra (um):  {min_f[1]:.4f} to {max_f[1]:.4f}")
    console.print(f"  Time (s):           {min_f[2]:.4f} to {max_f[2]:.4f}")
    console.print("")
    console.print("MAX REDUCTION VS BASELINE (median config):")
    console.print(f"  Energy SEC:     {energy_imp:+.1f}% reduction achievable")
    console.print(f"  Roughness Ra:   {roughness_imp:+.1f}% reduction achievable")
    console.print(f"  Machining Time: {time_imp:+.1f}% reduction achievable")
    console.print("")
    console.print("Pareto front saved to:")
    console.print(f"  {paths['pareto_X']}")
    console.print(f"  {paths['pareto_F']}")
    console.print("[bold green]=======================================[/bold green]")

if __name__ == "__main__":
    main()
