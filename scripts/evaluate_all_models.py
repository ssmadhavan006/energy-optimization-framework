import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import MODEL_REGISTRY, TARGETS, TARGET_CONFIG
from src.training.train_pipeline import TrainPipeline
from src.evaluation.results_table import build_comparison_df, save_results, print_latex_table

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EvaluateAllModels")

console = Console()

def main():
    console.print("[bold blue]=== EnergyOptAI — Model Re-Evaluation Script ===[/bold blue]")
    
    trained_dir = Path("outputs/models/trained")
    if not trained_dir.exists():
        console.print("[bold red]Error: outputs/models/trained/ directory not found. Please train models first.[/bold red]")
        sys.exit(1)
        
    all_results = []
    
    for target in TARGETS:
        for m_key, model_cls in MODEL_REGISTRY.items():
            model_path = trained_dir / f"{target}_{m_key}_final.pkl"
            
            if not model_path.exists():
                logger.warning(f"Trained model file not found: {model_path.name}. Skipping.")
                continue
                
            logger.info(f"Loading and evaluating model: {model_path.name}...")
            
            try:
                # 1. Instantiate pipeline to get test data
                pipeline = TrainPipeline(
                    model_class=model_cls,
                    target_name=target,
                    run_tuning=False
                )
                _, X_test, _, y_test = pipeline.prepare_data()
                
                # 2. Instantiate and load model
                model_inst = model_cls(target_name=target)
                model_inst.load(model_path)
                
                # 3. Evaluate
                metrics = model_inst.evaluate(X_test, y_test)
                
                all_results.append({
                    "target": target,
                    "model": m_key,
                    "r2": metrics["r2"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "mape": metrics["mape"],
                    "n_samples": metrics["n_samples"]
                })
                
            except Exception as e:
                logger.error(f"Failed to evaluate {m_key} on {target}: {e}")
                
    if all_results:
        # Build DataFrame and save
        df_results = build_comparison_df(all_results)
        save_results(df_results, "full_model_comparison")
        
        # Display rich table
        table = Table(title="Model Performance Comparison Summary (Re-evaluated)")
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
        
        # Print LaTeX table
        print_latex_table(df_results)
        
        # Generate all comparison plots
        try:
            logger.info("Generating evaluation and comparison plots...")
            from src.evaluation.comparison_plots import generate_all_plots
            from src.data.loaders import load_all
            datasets = load_all()
            generate_all_plots(all_results, MODEL_REGISTRY, datasets)
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
        
    else:
        console.print("[bold red]No trained models were successfully evaluated.[/bold red]")

if __name__ == "__main__":
    main()
