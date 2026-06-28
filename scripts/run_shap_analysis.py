import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.table import Table

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import MODEL_REGISTRY, TARGET_CONFIG, TARGETS
from src.training.train_pipeline import TrainPipeline
from src.explainability.shap_analysis import SHAPAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunSHAP")

console = Console()

def get_best_model_info(target: str) -> tuple:
    """
    Returns (model_name, model_class, pkl_filename, save_suffix).
    """
    target = target.lower()
    if target == "energy":
        # Champion re-engineered model
        return "random_forest", MODEL_REGISTRY["random_forest"], "energy_random_forest_sec_final.pkl", "_sec"
    elif target == "roughness":
        return "catboost", MODEL_REGISTRY["catboost"], "roughness_catboost_final.pkl", ""
    elif target == "time":
        return "catboost", MODEL_REGISTRY["catboost"], "time_catboost_final.pkl", ""
    else:
        raise ValueError(f"Unknown target: {target}")

def run_target_shap(target: str, max_samples: int) -> SHAPAnalyzer:
    console.print(f"\n[bold blue]=== SHAP Analysis: {target.upper()} ===[/bold blue]")
    
    # 1. Resolve model info
    m_key, model_cls, pkl_name, save_suffix = get_best_model_info(target)
    
    # Dynamically select sec target column for energy target
    if target == "energy":
        TARGET_CONFIG["energy"]["target_col"] = "sec"
        TARGET_CONFIG["energy"]["unit"] = "J/mm³"
        
    model_path = Path("outputs/models/trained") / pkl_name
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model pickle not found at {model_path}. Please train it first.")
        
    # 2. Re-create dataset preparation exactly matching training
    pipeline = TrainPipeline(
        model_class=model_cls,
        target_name=target,
        run_tuning=False,
        save_suffix=save_suffix
    )
    
    X_train, X_test, y_train, y_test = pipeline.prepare_data()
    
    # 3. Instantiate model wrapper and load trained weights
    model_inst = model_cls(target_name=target)
    model_inst.load(model_path)
    logger.info(f"Loaded trained model from {model_path}")
    
    # 4. Run SHAP Analyzer
    unit = TARGET_CONFIG[target]["unit"]
    analyzer = SHAPAnalyzer(
        model=model_inst,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        target_name=target,
        target_unit=unit,
        feature_names=X_test.columns.tolist(),
        max_display=10
    )
    
    analyzer.compute()
    analyzer.plot_all()
    analyzer.save_shap_values()
    
    # Print engineering insights
    insights = analyzer.get_engineering_insights()
    console.print("\n[bold green]Generated Engineering Insights:[/bold green]")
    for s in insights:
        console.print(f" - {s}")
        
    return analyzer

def main():
    parser = argparse.ArgumentParser(description="EnergyOptAI — SHAP Analysis CLI")
    parser.add_argument("--target", type=str, choices=["energy", "roughness", "time", "all"], default="all",
                        help="Target variable to run SHAP analysis for")
    parser.add_argument("--samples", type=int, default=500, help="Maximum number of test samples for SHAP")
    parser.add_argument("--comparison-only", action="store_true", help="Only generates cross-target comparison plots")
    
    args = parser.parse_args()
    
    targets_to_run = ["roughness", "time", "energy"] if args.target == "all" else [args.target]
    
    analyzers = {}
    
    if not args.comparison_only:
        for target in targets_to_run:
            try:
                analyzer = run_target_shap(target, args.samples)
                analyzers[target] = analyzer
            except Exception as e:
                logger.exception(f"SHAP Analysis failed for target {target}: {e}")
                console.print(f"[bold red]SHAP Analysis failed for target {target}: {e}[/bold red]")
                
    # Run cross-target comparison if multiple targets are processed
    if args.target == "all" or args.comparison_only:
        console.print("\n[bold blue]=== Cross-Target SHAP Comparison ===[/bold blue]")
        try:
            # Re-load or import comparison
            from src.explainability.shap_comparison import generate_all_comparison_plots
            
            # If in comparison-only mode, we need to load or run the analyzers
            if not analyzers:
                logger.info("Loading models and datasets for comparison plots...")
                for target in ["roughness", "time", "energy"]:
                    m_key, model_cls, pkl_name, save_suffix = get_best_model_info(target)
                    if target == "energy":
                        TARGET_CONFIG["energy"]["target_col"] = "sec"
                        TARGET_CONFIG["energy"]["unit"] = "J/mm³"
                    pipeline = TrainPipeline(
                        model_class=model_cls,
                        target_name=target,
                        run_tuning=False,
                        save_suffix=save_suffix
                    )
                    X_train, X_test, y_train, y_test = pipeline.prepare_data()
                    model_inst = model_cls(target_name=target)
                    model_inst.load(Path("outputs/models/trained") / pkl_name)
                    
                    analyzer = SHAPAnalyzer(
                        model=model_inst,
                        X_train=X_train,
                        X_test=X_test,
                        y_test=y_test,
                        target_name=target,
                        target_unit=TARGET_CONFIG[target]["unit"],
                        feature_names=X_test.columns.tolist()
                    )
                    analyzer.compute()
                    analyzers[target] = analyzer
                    
            saved_paths = generate_all_comparison_plots(analyzers)
            console.print("\n[bold green]Cross-target comparison complete. Saved plots:[/bold green]")
            for p in saved_paths:
                console.print(f" - {p}")
                
        except Exception as e:
            logger.exception(f"Cross-target comparison failed: {e}")
            console.print(f"[bold red]Cross-target comparison failed: {e}[/bold red]")
            
    # Print completion summary
    console.print("\n[bold green]SHAP ANALYSIS COMPLETE[/bold green]")
    table = Table(title="SHAP Analysis Execution Summary")
    table.add_column("Target", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Status", style="green")
    
    for t in ["roughness", "time", "energy"]:
        status = "Success" if t in analyzers else "Skipped/Failed"
        m_key, _, _, _ = get_best_model_info(t)
        table.add_row(t, m_key, status)
        
    console.print(table)

if __name__ == '__main__':
    main()
