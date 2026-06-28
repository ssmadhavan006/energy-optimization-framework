import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.models.base_model import BaseModel
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SHAPAnalysis")

class SHAPAnalyzer:
    """
    SHAP TreeExplainer wrapper for all EnergyOptAI models.
    Supports TreeExplainer for efficiency on tree-based models (XGBoost, CatBoost, RandomForest).
    """

    def __init__(
        self,
        model: BaseModel,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        target_name: str,
        target_unit: str,
        feature_names: List[str],
        output_dir: Path = Path("outputs/figures/shap"),
        max_display: int = 15,
        background_samples: int = 100
    ):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.target_name = target_name
        self.target_unit = target_unit
        self.feature_names = feature_names
        self.output_dir = Path(output_dir)
        self.max_display = max_display
        self.background_samples = background_samples
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subsample test set if larger than 500 rows to prevent memory crashes
        if len(X_test) > 500:
            logger.info(f"Subsampling test set from {len(X_test)} to 500 rows for SHAP.")
            self.X_test_subsampled = X_test.sample(500, random_state=42).copy()
            if y_test is not None:
                self.y_test_subsampled = y_test.loc[self.X_test_subsampled.index].copy()
        else:
            self.X_test_subsampled = X_test.copy()
            if y_test is not None:
                self.y_test_subsampled = y_test.copy()
                
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

    def compute(self) -> None:
        """Computes SHAP values using TreeExplainer or Explainer fallback."""
        logger.info(f"Computing SHAP values for target {self.target_name}...")
        
        # Extract underlying estimator from the BaseModel wrapper
        estimator = self.model.model
        
        try:
            # Check if model has a get_booster or similar method, or use estimator directly
            # check_additivity=False is used for CatBoost / XGBoost compatibility
            self.explainer = shap.TreeExplainer(estimator, check_additivity=False)
            self.shap_values = self.explainer(self.X_test_subsampled)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Falling back to shap.Explainer...")
            try:
                # Fallback to standard Explainer
                self.explainer = shap.Explainer(estimator)
                self.shap_values = self.explainer(self.X_test_subsampled)
                logger.info("shap.Explainer fallback initialized successfully.")
            except Exception as ex:
                logger.error(f"Fallback shap.Explainer also failed: {ex}")
                raise
                
        # Resolve CatBoost shape mismatch (squeeze 3D arrays if necessary)
        if hasattr(self.shap_values, "values") and len(self.shap_values.values.shape) == 3:
            logger.info(f"Detected 3D SHAP values array of shape {self.shap_values.values.shape}. Squeezing...")
            self.shap_values.values = self.shap_values.values[:, :, 0]
            if len(self.shap_values.base_values.shape) > 1:
                self.shap_values.base_values = self.shap_values.base_values[:, 0]
                
        # Extract expected value
        if hasattr(self.explainer, "expected_value"):
            ev = self.explainer.expected_value
            if isinstance(ev, (list, np.ndarray)) and len(ev) > 0:
                self.expected_value = ev[0]
            else:
                self.expected_value = ev
        else:
            self.expected_value = np.mean(self.model.predict(self.X_test_subsampled))
            
        logger.info(f"SHAP values computed successfully. Expected value: {self.expected_value:.4f}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Returns DataFrame of mean absolute SHAP values per feature, sorted descending."""
        if self.shap_values is None:
            raise ValueError("SHAP values have not been computed. Run compute() first.")
            
        mean_abs_shaps = np.abs(self.shap_values.values).mean(axis=0)
        df_imp = pd.DataFrame({
            "feature_name": self.feature_names,
            "mean_abs_shap": mean_abs_shaps
        })
        df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df_imp["rank"] = df_imp.index + 1
        return df_imp

    def plot_global_importance_bar(self, figsize=(10, 7)) -> str:
        """Generates horizontal bar chart of mean |SHAP| per feature."""
        df_imp = self.get_feature_importance().head(self.max_display)
        
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.barplot(
            x="mean_abs_shap", 
            y="feature_name", 
            data=df_imp, 
            color="steelblue", 
            ax=ax
        )
        
        # Add labels to the bars
        for i, p in enumerate(ax.patches):
            val = df_imp.iloc[i]["mean_abs_shap"]
            ax.annotate(f"{val:.4f}", (val, p.get_y() + p.get_height() / 2),
                        ha='left', va='center', xytext=(4, 0), textcoords='offset points', fontsize=9)
            
        ax.set_title(f"Global Feature Importance — {self.target_name.capitalize()} ({self.target_unit})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=11)
        ax.set_ylabel("Features", fontsize=11)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{self.target_name}_shap_global_importance.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved global importance bar plot to {save_path}")
        return str(save_path)

    def plot_beeswarm_summary(self, figsize=(12, 8)) -> str:
        """Generates SHAP beeswarm summary plot."""
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        
        # Call SHAP's beeswarm summary plot
        shap.plots.beeswarm(self.shap_values, max_display=self.max_display, show=False)
        
        # Set title and labels
        fig = plt.gcf()
        ax = fig.gca()
        ax.set_title(f"SHAP Summary — {self.target_name.capitalize()} ({self.target_unit})", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{self.target_name}_shap_beeswarm.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved beeswarm summary plot to {save_path}")
        return str(save_path)

    def plot_dependence(self, feature_name: str, interaction_feature: str = "auto", figsize=(9, 6)) -> str:
        """Generates a scatter-based SHAP dependence plot for the specified feature."""
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in feature list.")
            
        feat_idx = self.feature_names.index(feature_name)
        x_vals = self.X_test_subsampled[feature_name]
        shap_vals = self.shap_values.values[:, feat_idx]
        
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        
        # Automatically choose interaction feature if set to "auto"
        if interaction_feature == "auto":
            # Select the top feature (different from feature_name)
            df_imp = self.get_feature_importance()
            top_other_feats = [f for f in df_imp["feature_name"] if f != feature_name]
            interaction_feature = top_other_feats[0] if top_other_feats else None
            
        if interaction_feature and interaction_feature in self.X_test_subsampled.columns:
            color_vals = self.X_test_subsampled[interaction_feature]
            
            # Map categories to codes if categorical
            if color_vals.dtype == object or isinstance(color_vals.dtype, pd.CategoricalDtype):
                color_vals = color_vals.astype('category').cat.codes
                
            sc = plt.scatter(x_vals, shap_vals, c=color_vals, cmap='plasma', alpha=0.8, edgecolors='none', s=40)
            cbar = plt.colorbar(sc)
            cbar.set_label(f"{interaction_feature} values", fontsize=10)
        else:
            plt.scatter(x_vals, shap_vals, color='steelblue', alpha=0.8, edgecolors='none', s=40)
            
        plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
        plt.title(f"SHAP Dependence — {feature_name} effect on {self.target_name.capitalize()}", fontsize=12, fontweight='bold')
        plt.xlabel(f"{feature_name} values", fontsize=11)
        plt.ylabel(f"SHAP Value ({self.target_unit})", fontsize=11)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{self.target_name}_shap_dependence_{feature_name}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved SHAP dependence plot for {feature_name} to {save_path}")
        return str(save_path)

    def plot_top_dependence_plots(self, n_top: int = 5) -> List[str]:
        """Generates dependence plots for the top n_top features by SHAP importance."""
        df_imp = self.get_feature_importance().head(n_top)
        paths = []
        for feat in df_imp["feature_name"]:
            paths.append(self.plot_dependence(feat))
        return paths

    def plot_waterfall_local(self, sample_idx: int = 0, figsize=(10, 7)) -> str:
        """Generates local explanation waterfall plot for a single sample."""
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=figsize)
        
        # Call waterfall plot
        shap.plots.waterfall(self.shap_values[sample_idx], max_display=10, show=False)
        
        # Add prediction vs true values annotation
        fig = plt.gcf()
        ax = fig.gca()
        
        pred_val = self.model.predict(self.X_test_subsampled.iloc[[sample_idx]])[0]
        true_val = self.y_test_subsampled.iloc[sample_idx]
        
        ax.set_title(f"Local SHAP Explanation — Sample {sample_idx} ({self.target_name.capitalize()})", fontsize=12, fontweight='bold')
        ax.text(0.95, 0.05, f"Prediction: {pred_val:.4f} {self.target_unit}\nTrue: {true_val:.4f} {self.target_unit}",
                transform=ax.transAxes, ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.output_dir / f"{self.target_name}_shap_waterfall_sample_{sample_idx}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved waterfall local explanation plot for sample {sample_idx} to {save_path}")
        return str(save_path)

    def plot_waterfall_best_and_worst(self) -> List[str]:
        """Generates waterfall plots for typical, best-performing, and worst-performing samples."""
        pred_vals = self.model.predict(self.X_test_subsampled)
        true_vals = self.y_test_subsampled.values
        errors = np.abs(pred_vals - true_vals)
        
        # 1. Typical case: closest to median prediction
        median_pred = np.median(pred_vals)
        typical_idx = int(np.argmin(np.abs(pred_vals - median_pred)))
        
        # 2. Worst case: highest predicted value
        worst_idx = int(np.argmax(pred_vals))
        
        # 3. Best case: lowest predicted value
        best_idx = int(np.argmin(pred_vals))
        
        paths = []
        for name, idx in [("typical", typical_idx), ("worst", worst_idx), ("best", best_idx)]:
            path = self.plot_waterfall_local(idx)
            # Rename to match case name
            new_path = self.output_dir / f"{self.target_name}_shap_waterfall_{name}_case.png"
            if Path(path).exists():
                Path(path).replace(new_path)
            paths.append(str(new_path))
            
        return paths

    def plot_force_plot_html(self, n_samples: int = 50) -> str:
        """Generates interactive force plot saved as an HTML file."""
        save_path = self.output_dir / f"{self.target_name}_shap_force_plot.html"
        
        # Slice for n_samples
        n_samples = min(n_samples, len(self.X_test_subsampled))
        
        # shap.force_plot outputs an HTML object
        html_obj = shap.force_plot(
            self.expected_value,
            self.shap_values.values[:n_samples],
            self.X_test_subsampled.iloc[:n_samples]
        )
        
        shap.save_html(str(save_path), html_obj)
        logger.info(f"Saved force plot HTML to {save_path}")
        return str(save_path)

    def save_shap_values(self) -> str:
        """Saves raw SHAP values as CSV for scientific reproducibility."""
        df_shap = pd.DataFrame(
            self.shap_values.values,
            columns=self.feature_names,
            index=self.X_test_subsampled.index
        )
        results_dir = Path("outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / f"{self.target_name}_shap_values.csv"
        df_shap.to_csv(save_path)
        logger.info(f"Saved raw SHAP values CSV to {save_path}")
        return str(save_path)

    def get_engineering_insights(self) -> List[str]:
        """Derives human-readable engineering insights based on feature correlation and SHAP impact."""
        df_imp = self.get_feature_importance().head(5)
        insights = []
        
        for idx, row in df_imp.iterrows():
            feat = row["feature_name"]
            rank = row["rank"]
            impact = row["mean_abs_shap"]
            
            x_vals = self.X_test_subsampled[feat]
            # Convert categories to codes to calculate correlation
            if x_vals.dtype == object or isinstance(x_vals.dtype, pd.CategoricalDtype):
                x_vals = x_vals.astype('category').cat.codes
                
            shap_vals = self.shap_values.values[:, self.feature_names.index(feat)]
            
            # Compute correlation to determine direction of effect
            if x_vals.nunique() > 1:
                corr = np.corrcoef(x_vals, shap_vals)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
            else:
                corr = 0.0
                
            direction = "increases" if corr > 0.0 else "reduces"
            effect = "positive" if corr > 0.0 else "negative"
            
            insight_str = (
                f"{feat} (rank {rank}): Higher values of {feat} have a {effect} impact on {self.target_name} SHAP values "
                f"(correlation: {corr:.4f}), indicating that increasing {feat} generally {direction} predicted {self.target_name}."
            )
            insights.append(insight_str)
            
        results_dir = Path("outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        insights_path = results_dir / f"{self.target_name}_shap_insights.txt"
        with open(insights_path, 'w', encoding='utf-8') as f:
            for s in insights:
                f.write(s + "\n")
                
        logger.info(f"Saved SHAP insights text file to {insights_path}")
        return insights

    def plot_all(self) -> Dict[str, str]:
        """Sequentially runs all plotting pipelines and returns dictionary of paths."""
        logger.info(f"Running full SHAP visualization pipeline for {self.target_name}...")
        plots = {
            "global_importance": self.plot_global_importance_bar(),
            "beeswarm": self.plot_beeswarm_summary(),
            "force_plot_html": self.plot_force_plot_html()
        }
        
        # Dependence plots
        dep_paths = self.plot_top_dependence_plots(n_top=5)
        for i, path in enumerate(dep_paths):
            plots[f"dependence_{i}"] = path
            
        # Waterfall local explanation cases
        wf_paths = self.plot_waterfall_best_and_worst()
        for i, name in enumerate(["typical", "worst", "best"]):
            plots[f"waterfall_{name}"] = wf_paths[i]
            
        return plots
