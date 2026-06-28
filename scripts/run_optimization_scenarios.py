import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.model_registry import MODEL_REGISTRY, TARGET_CONFIG
from src.optimization.decision_space import DECISION_SPACE
from src.optimization.surrogate_adapter import SurrogateAdapter
from src.optimization.nsga2_optimizer import EnergyOptNSGA2
from scripts.run_optimization import compute_target_train_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunScenarios")

def main():
    model_dir = Path("outputs/models/trained")
    scaler_dir = Path("outputs/models/scalers")
    output_dir = Path("outputs/results/optimization")
    fig_dir = Path("outputs/figures/pareto")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading champion models, scalers, and encoders...")
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
    
    scenarios = {
        "new_tool": 0.0,
        "mid_tool": 0.053,
        "worn_tool": 0.10
    }
    
    results = {}
    
    # Run NSGA-II for each scenario
    for name, value in scenarios.items():
        logger.info(f"=== Running NSGA-II Optimization for Scenario: {name} (TCond = {value} mm) ===")
        optimizer = EnergyOptNSGA2(
            adapter=adapter,
            decision_space=DECISION_SPACE,
            pop_size=100,
            n_gen=200,
            seed=42,
            verbose=False,
            tool_condition_fixed=value
        )
        res = optimizer.run()
        
        # Save X and F files
        df_X = pd.DataFrame(optimizer.pareto_X, columns=list(DECISION_SPACE.keys()))
        df_F = pd.DataFrame(optimizer.pareto_F, columns=["energy_sec", "roughness_ra", "time_s"])
        
        df_X.to_csv(output_dir / f"pareto_X_{name}.csv", index=False)
        df_F.to_csv(output_dir / f"pareto_F_{name}.csv", index=False)
        
        results[name] = {
            "F": optimizer.pareto_F,
            "X": optimizer.pareto_X
        }
        logger.info(f"Scenario {name} finished. Found {len(optimizer.pareto_F)} solutions.")
        
    # Plot Pareto front comparisons (Roughness vs Time)
    plt.figure(figsize=(10, 6), facecolor='white')
    colors = {"new_tool": "green", "mid_tool": "blue", "worn_tool": "red"}
    markers = {"new_tool": "o", "mid_tool": "^", "worn_tool": "s"}
    labels = {
        "new_tool": "New Tool (TCond = 0.00 mm)",
        "mid_tool": "Mid-Life Tool (TCond = 0.053 mm)",
        "worn_tool": "Worn Tool (TCond = 0.10 mm)"
    }
    
    for name, data in results.items():
        F = data["F"]
        # F contains: energy_sec, roughness_ra, time_s
        plt.scatter(
            F[:, 2], F[:, 1], 
            color=colors[name], marker=markers[name], 
            label=labels[name], alpha=0.7, edgecolors='k', s=50
        )
        
    plt.xlabel("Machining Cycle Time (seconds)", fontsize=11)
    plt.ylabel("Surface Roughness Ra (microns)", fontsize=11)
    plt.title("Pareto Front Shift under Tool Wear Progression", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    
    plot_path = fig_dir / "tool_wear_pareto_shift.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot to {plot_path}")
    
    # Also save to paper/figures/
    paper_fig_path = Path("paper/figures/Fig12_tool_wear_pareto_shift.png")
    paper_fig_path.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(plot_path, paper_fig_path)
    logger.info(f"Copied comparison plot to paper figure folder: {paper_fig_path}")

if __name__ == "__main__":
    main()
