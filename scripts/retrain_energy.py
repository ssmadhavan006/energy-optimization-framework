import os
import sys
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import MODEL_REGISTRY, TARGET_CONFIG
from src.training.train_pipeline import TrainPipeline
from src.evaluation.results_table import build_comparison_df, save_results
from src.data.loaders import load_all

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RetrainEnergy")

console = Console()

def main():
    parser = argparse.ArgumentParser(description="EnergyOptAI — Retrain Energy Model Script")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna tuning trials")
    parser.add_argument("--skip-tuning", action="store_true", help="Uses default parameters without Optuna tuning")
    parser.add_argument("--target-col", type=str, choices=["sec", "spindle_power_w"], default="sec",
                        help="Energy target column to predict")
    
    args = parser.parse_args()
    
    target = "energy"
    
    # Dynamically update the configuration to target the chosen column
    TARGET_CONFIG[target]["target_col"] = args.target_col
    if args.target_col == "sec":
        TARGET_CONFIG[target]["unit"] = "J/mm³"
        TARGET_CONFIG[target]["description"] = "Specific Energy Consumption per unit MRR"
    else:
        TARGET_CONFIG[target]["unit"] = "W"
        TARGET_CONFIG[target]["description"] = "Specific Spindle Power Consumption"
        
    save_suffix = f"_{args.target_col}"
    logger.info(f"Retraining energy target. Target col: {args.target_col}, Suffix: {save_suffix}")
    
    model_keys = list(MODEL_REGISTRY.keys())
    all_results = []
    
    for m_key in model_keys:
        is_baseline = m_key in ["linear_regression", "svr"]
        run_tuning = not args.skip_tuning and not is_baseline
        
        console.print(f"[bold green]Training {m_key} for target {target} ({args.target_col})...[/bold green] (Tuning={run_tuning})")
        
        try:
            model_cls = MODEL_REGISTRY[m_key]
            pipeline = TrainPipeline(
                model_class=model_cls,
                target_name=target,
                run_tuning=run_tuning,
                n_trials=args.trials,
                random_state=42,
                save_suffix=save_suffix
            )
            
            # Run train & evaluate sequence
            res = pipeline.run_all()
            
            # Store test set metrics
            test_metrics = res["test_metrics"]
            all_results.append({
                "target": target,
                "model": m_key,
                "r2": test_metrics["r2"],
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "mape": test_metrics["mape"],
                "n_samples": test_metrics["n_samples"]
            })
            
        except Exception as e:
            console.print(f"[bold red]Error training {m_key} on {target}: {e}[/bold red]")
            logger.exception(f"Pipeline failure: {m_key} on {target}")
            
    if all_results:
        df_results = build_comparison_df(all_results)
        
        # Save results to a target-specific CSV
        metrics_dir = Path("outputs/results/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        comparison_name = f"energy{save_suffix}_model_comparison"
        df_results.to_csv(metrics_dir / f"{comparison_name}.csv", index=False)
        df_results.to_json(metrics_dir / f"{comparison_name}.json", orient="records", indent=4)
        
        # Display rich table
        table = Table(title=f"Retrained Energy Model ({args.target_col}) Performance Summary")
        table.add_column("Model", style="magenta")
        table.add_column("R²", justify="right", style="green")
        table.add_column("RMSE", justify="right")
        table.add_column("MAE", justify="right")
        table.add_column("MAPE (%)", justify="right")
        table.add_column("Samples", justify="right")
        
        for _, row in df_results.iterrows():
            table.add_row(
                str(row['model']),
                f"{row['r2']:.4f}",
                f"{row['rmse']:.4f}",
                f"{row['mae']:.4f}",
                f"{row['mape']:.2f}%",
                str(int(row['n_samples']))
            )
        console.print(table)
        
        # Compute and print descriptive statistics of the target variable
        datasets = load_all()
        from src.data.feature_engineering import engineer_mendeley_energy_target
        df_eng = engineer_mendeley_energy_target(datasets["mendeley"]["parsed"])
        stats = df_eng[args.target_col].describe()
        console.print(f"\n[bold blue]=== Target Variable ({args.target_col}) Descriptive Stats ===[/bold blue]")
        console.print(stats)

if __name__ == '__main__':
    main()
