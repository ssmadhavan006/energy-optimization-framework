import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from typing import Dict, Any, List

logger = logging.getLogger("ComparisonPlots")

# Output directory for results plots
OUT_DIR = Path("outputs/figures/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_model_comparison_bar(df: pd.DataFrame, target_name: str, metric: str = 'r2') -> None:
    """
    Grouped bar chart comparing all models on one target.
    Best model is annotated with its value.
    """
    df_t = df[df["target"] == target_name].copy()
    if df_t.empty:
        return
        
    df_t["model_label"] = df_t["model"].apply(lambda x: x.replace('_', ' ').title())
    df_t["family"] = df_t["model"].apply(
        lambda x: "Baseline" if x in ["linear_regression", "svr"] else "Ensemble"
    )
    
    # Sort for plotting
    df_t = df_t.sort_values(by=metric, ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = df_t["family"].map({"Baseline": "lightcoral", "Ensemble": "dodgerblue"})
    
    ax = sns.barplot(data=df_t, x="model_label", y=metric, palette=list(colors))
    plt.title(f"Model Comparison on {target_name.capitalize()} ({metric.upper()})", fontsize=14, fontweight='bold')
    plt.xlabel("Model Family")
    plt.ylabel(metric.upper())
    plt.xticks(rotation=15)
    
    # Annotate best model
    best_row = df_t.iloc[0]
    best_val = best_row[metric]
    plt.annotate(
        f"Best: {best_val:.4f}",
        xy=(0, best_val),
        xytext=(0.3, best_val * 0.95 if best_val > 0 else best_val + 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        fontweight='bold',
        fontsize=10
    )
    
    # Add values on top of all bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9)
                    
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target_name}_model_comparison_{metric}.png", dpi=300)
    plt.close()

def plot_cv_score_distribution(cv_results: Dict[str, Any], target_name: str) -> None:
    """
    Box plot showing 5-fold CV score distribution per model.
    """
    # cv_results should map model_name -> list of fold score metrics
    plot_data = []
    for model_name, res in cv_results.items():
        if "fold_metrics" in res:
            for fold in res["fold_metrics"]:
                plot_data.append({
                    "Model": model_name.replace('_', ' ').title(),
                    "RMSE": fold["rmse"]
                })
                
    if not plot_data:
        return
        
    df_plot = pd.DataFrame(plot_data)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x="Model", y="RMSE", palette="Set3")
    sns.stripplot(data=df_plot, x="Model", y="RMSE", color="black", alpha=0.5, size=6, jitter=0.1)
    plt.title(f"5-Fold CV RMSE Distribution — {target_name.capitalize()}", fontsize=14, fontweight='bold')
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target_name}_cv_distribution.png", dpi=300)
    plt.close()

def plot_actual_vs_predicted(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series, target_name: str) -> None:
    """
    Scatter plot: actual vs predicted values on test set.
    Generates one subplot per model, all in one figure.
    """
    n_models = len(models_dict)
    if n_models == 0:
        return
        
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharex=False, sharey=False)
    if n_models == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
        
    y_test_arr = y_test.values.flatten() if hasattr(y_test, 'values') else np.asarray(y_test).flatten()
    
    for idx, (m_key, model) in enumerate(models_dict.items()):
        try:
            y_pred = model.predict(X_test).flatten()
            r2_val = model.evaluate(X_test, y_test)["r2"]
            
            ax = axs[idx]
            ax.scatter(y_test_arr, y_pred, alpha=0.5, color='teal')
            
            # Draw y=x line
            min_val = min(y_test_arr.min(), y_pred.min())
            max_val = max(y_test_arr.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_title(f"{m_key.replace('_', ' ').title()}\n$R^2$ = {r2_val:.4f}", fontsize=11)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        except Exception as e:
            logger.warning(f"Failed to plot actual vs predicted for {m_key}: {e}")
            
    # Hide empty subplots
    for i in range(idx + 1, len(axs)):
        fig.delaxes(axs[i])
        
    plt.suptitle(f"Actual vs. Predicted Values — {target_name.capitalize()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target_name}_actual_vs_predicted.png", dpi=300)
    plt.close()

def plot_residuals(model: Any, X_test: pd.DataFrame, y_test: pd.Series, target_name: str) -> None:
    """
    Residual plot: predicted values vs residuals.
    Includes horizontal zero line and shaded ±1 RMSE band.
    """
    try:
        y_pred = model.predict(X_test).flatten()
        y_test_arr = y_test.values.flatten() if hasattr(y_test, 'values') else np.asarray(y_test).flatten()
        residuals = y_test_arr - y_pred
        
        rmse = np.sqrt(np.mean(residuals**2))
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
        plt.axhline(0, color='red', linestyle='--', lw=2)
        
        # Shaded RMSE band
        plt.fill_between(
            [y_pred.min(), y_pred.max()],
            [-rmse, -rmse],
            [rmse, rmse],
            color='gray',
            alpha=0.15,
            label='$\pm$1 RMSE Band'
        )
        
        plt.title(f"Residual Plot: {model.model_name.replace('_', ' ').title()} ({target_name.capitalize()})", fontsize=12, fontweight='bold')
        plt.xlabel("Predicted Value")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{target_name}_{model.model_name}_residuals.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to plot residuals for {model.model_name}: {e}")

def plot_learning_curves(model: Any, X_train: pd.DataFrame, y_train: pd.Series, target_name: str) -> None:
    """
    Plots training and validation learning curves as sample size grows.
    """
    try:
        # Re-build fresh estimator on a copy to avoid mutating the fitted model in-place
        import copy
        model_copy = copy.deepcopy(model)
        model_copy.build(model_copy.params)
        estimator = model_copy.model
        
        # Scaling is handled internally by train_pipeline for clean data.
        # We can just fit learning curve directly on X_train (already scaled).
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1, random_state=42
        )
        
        # Convert neg RMSE back to positive RMSE
        train_rmse = -np.mean(train_scores, axis=1)
        test_rmse = -np.mean(test_scores, axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_rmse, 'o-', color="r", label="Training Score")
        plt.plot(train_sizes, test_rmse, 'o-', color="g", label="Cross-validation Score")
        
        plt.title(f"Learning Curves: {model.model_name.replace('_', ' ').title()} ({target_name.capitalize()})", fontsize=12, fontweight='bold')
        plt.xlabel("Training Examples")
        plt.ylabel("RMSE")
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{target_name}_{model.model_name}_learning_curve.png", dpi=300)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to plot learning curves for {model.model_name}: {e}")

def generate_all_plots(all_results: list, models_registry: dict, datasets: dict) -> None:
    """
    Generates all comparative plots across targets and models.
    """
    logger.info("Generating comparative and model evaluation plots...")
    df_results = pd.DataFrame(all_results)
    
    # 1. Bar charts per target
    for target in df_results["target"].unique():
        plot_model_comparison_bar(df_results, target, 'r2')
        plot_model_comparison_bar(df_results, target, 'rmse')
        
    # 2. Residuals, learning curves and actual vs predicted plots
    # For actual/predicted, we need to load test data and model instances
    trained_dir = Path("outputs/models/trained")
    if not trained_dir.exists():
        return
        
    for target in df_results["target"].unique():
        models_dict = {}
        X_train, X_test, y_train, y_test = None, None, None, None
        
        for m_key, model_cls in models_registry.items():
            model_path = trained_dir / f"{target}_{m_key}_final.pkl"
            if model_path.exists():
                try:
                    # Prepare test data once for the target
                    if X_test is None:
                        from src.training.train_pipeline import TrainPipeline
                        pipeline = TrainPipeline(model_class=model_cls, target_name=target, run_tuning=False)
                        X_train, X_test, y_train, y_test = pipeline.prepare_data()
                        
                    # Load model
                    model_inst = model_cls(target_name=target)
                    model_inst.load(model_path)
                    
                    models_dict[m_key] = model_inst
                    
                    # Generate individual model curves
                    plot_residuals(model_inst, X_test, y_test, target)
                    plot_learning_curves(model_inst, X_train, y_train, target)
                    
                except Exception as e:
                    logger.error(f"Error preparing plot data for {m_key} on {target}: {e}")
                    
        if models_dict and X_test is not None:
            plot_actual_vs_predicted(models_dict, X_test, y_test, target)
            
    logger.info("All plots generated successfully.")
