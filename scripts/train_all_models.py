import os
import sys
import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import MODEL_REGISTRY, TARGET_CONFIG, TARGETS
from src.training.train_pipeline import TrainPipeline
from src.evaluation.results_table import build_comparison_df, save_results
from src.data.loaders import load_all

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainAllModels")

console = Console()

def main():
    parser = argparse.ArgumentParser(description="EnergyOptAI — Model Training Script")
    parser.add_argument("--target", type=str, choices=["energy", "roughness", "time", "all"], default="all",
                        help="Target variable to train models for")
    parser.add_argument("--model", type=str, choices=["all", "xgboost", "catboost", "random_forest", "baselines"], default="all",
                        help="Model family to train")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna tuning trials")
    parser.add_argument("--skip-tuning", action="store_true", help="Uses default parameters without Optuna tuning")
    parser.add_argument("--cv-only", action="store_true", help="Runs K-fold CV only without final training")
    
    args = parser.parse_args()
    
    # Resolve targets and models
    target_list = TARGETS if args.target == "all" else [args.target]
    
    model_keys = []
    if args.model == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    elif args.model == "baselines":
        model_keys = ["linear_regression", "svr"]
    else:
        model_keys = [args.model]
        
    all_results = []
    
    # Directory for metrics
    metrics_dir = Path("outputs/results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each target & model combination
    for target in target_list:
        console.print(f"\n[bold blue]=== Target: {target.upper()} ===[/bold blue]")
        
        for m_key in model_keys:
            # We don't tune baseline models with Optuna
            is_baseline = m_key in ["linear_regression", "svr"]
            run_tuning = not args.skip_tuning and not is_baseline
            
            console.print(f"[bold green]Training {m_key} for target {target}...[/bold green] (Tuning={run_tuning})")
            
            try:
                model_cls = MODEL_REGISTRY[m_key]
                pipeline = TrainPipeline(
                    model_class=model_cls,
                    target_name=target,
                    run_tuning=run_tuning,
                    n_trials=args.trials,
                    random_state=42
                )
                
                # Run full pipeline or K-fold CV only
                X_train, X_test, y_train, y_test = pipeline.prepare_data()
                
                if args.cv_only:
                    logger.info(f"Running CV only for {m_key} on {target}")
                    from src.training.cross_validator import CrossValidator
                    model_inst = model_cls(target_name=target)
                    validator = CrossValidator(n_splits=5)
                    cv_res = validator.run(model_inst, X_train, y_train, target)
                    
                    # Store CV mean score
                    all_results.append({
                        "target": target,
                        "model": m_key,
                        "r2": cv_res["aggregate"]["r2"]["mean"],
                        "rmse": cv_res["aggregate"]["rmse"]["mean"],
                        "mae": cv_res["aggregate"]["mae"]["mean"],
                        "mape": cv_res["aggregate"]["mape"]["mean"],
                        "n_samples": len(y_train)
                    })
                else:
                    # Run full train & evaluate sequence
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
                
    # Build and print results summary table
    if all_results:
        df_results = build_comparison_df(all_results)
        save_results(df_results, "full_model_comparison")
        
        # Display rich table
        table = Table(title="Model Performance Comparison Summary")
        table.add_column("Target", style="cyan")
        table.add_column("Model", style="magenta")
        table.add_column("R²", justify="right", style="green")
        table.add_column("RMSE", justify="right")
        table.add_column("MAE", justify="right")
        table.add_column("MAPE (%)", justify="right")
        table.add_column("Samples", justify="right")
        
        for _, row in df_results.iterrows():
            table.add_row(
                str(row['target']),
                str(row['model']),
                f"{row['r2']:.4f}",
                f"{row['rmse']:.4f}",
                f"{row['mae']:.4f}",
                f"{row['mape']:.2f}%",
                str(int(row['n_samples']))
            )
        console.print(table)
        
        # Print LaTeX table for paper inclusion
        from src.evaluation.results_table import print_latex_table
        print_latex_table(df_results)
        
        # Generate all comparison plots
        try:
            logger.info("Generating evaluation and comparison plots...")
            from src.evaluation.comparison_plots import generate_all_plots
            # Load datasets to plot residuals/actuals
            datasets = load_all()
            generate_all_plots(all_results, MODEL_REGISTRY, datasets)
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
            
    else:
        console.print("[bold red]No results compiled.[/bold red]")

if __name__ == "__main__":
    main()
