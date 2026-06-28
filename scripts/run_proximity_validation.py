import os
import sys
import logging
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import MODEL_REGISTRY, TARGET_CONFIG
from src.optimization.decision_space import DECISION_SPACE
from src.optimization.surrogate_adapter import SurrogateAdapter
from src.optimization.proximity_validator import ProximityValidator
from scripts.run_optimization import compute_target_train_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunProximity")

def main():
    logger.info("Initializing Proximity Validator...")
    
    # 1. Load models, scalers, and stats
    model_dir = Path("outputs/models/trained")
    scaler_dir = Path("outputs/models/scalers")
    output_dir = Path("outputs/results/optimization")
    fig_dir = Path("outputs/figures/validation")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Load Pareto X coordinates (default is the mid-life Pareto X)
    pareto_x_path = output_dir / "pareto_X.csv"
    if not pareto_x_path.exists():
        # Let's search if any scenario pareto file exists, or warn
        logger.warning(f"Default pareto_X.csv not found at {pareto_x_path}. Trying pareto_X_mid_tool.csv...")
        pareto_x_path = output_dir / "pareto_X_mid_tool.csv"
        
    if not pareto_x_path.exists():
        logger.error("No Pareto front coordinates found. Run optimization script first.")
        sys.exit(1)
        
    df_pareto = pd.read_csv(pareto_x_path)
    logger.info(f"Loaded Pareto coordinates with shape {df_pareto.shape}")
    
    # Run validator
    validator = ProximityValidator(adapter=adapter, distance_threshold=0.2)
    augmented_df = validator.flag_out_of_distribution(df_pareto)
    summary = validator.summarize(augmented_df)
    
    # Print results
    print("\nPareto Proximity Validation Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    
    # Save files
    augmented_df.to_csv(output_dir / "pareto_X_validated.csv", index=False)
    logger.info(f"Saved validated pareto features to {output_dir / 'pareto_X_validated.csv'}")
    
    # Save summary as json
    with open(output_dir / "proximity_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    # Plot
    plot_path = validator.plot_distance_distribution(augmented_df, fig_dir)
    logger.info(f"Saved proximity plot to {plot_path}")
    
    # Copy plot to paper figures
    paper_fig_path = Path("paper/figures/Fig13_proximity_distances.png")
    paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(plot_path, paper_fig_path)
    logger.info(f"Copied proximity plot to paper figure folder: {paper_fig_path}")

if __name__ == "__main__":
    main()
