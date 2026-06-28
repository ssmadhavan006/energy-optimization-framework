import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from typing import Dict, Any, Tuple
from src.optimization.surrogate_adapter import SurrogateAdapter

logger = logging.getLogger("ProximityValidator")

class ProximityValidator:
    """
    Validates that NSGA-II Pareto solutions lie within or near
    the training distribution of each surrogate model.
    Uses nearest-neighbor distance in normalized feature space.
    """
    
    def __init__(
        self,
        adapter: SurrogateAdapter,
        n_neighbors: int = 5,
        distance_threshold: float = 0.2
    ):
        self.adapter = adapter
        self.n_neighbors = n_neighbors
        self.distance_threshold = distance_threshold
        
    def compute_nn_distances(
        self,
        pareto_df: pd.DataFrame,
        target: str
    ) -> np.ndarray:
        """
        For each Pareto solution, computes the distance to its
        nearest neighbor in the training set for the specified target's surrogate.
        """
        if self.adapter.X_train_data is None or target not in self.adapter.X_train_data:
            logger.warning(f"Training data not found for target '{target}'. Returning zeros.")
            return np.zeros(len(pareto_df))
            
        X_train_scaled = self.adapter.X_train_data[target].values
        
        # Fit NearestNeighbors on training data
        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        nn.fit(X_train_scaled)
        
        distances = []
        for _, row in pareto_df.iterrows():
            # row is feed_rate, depth_of_cut, spindle_speed, tool_condition
            sol_vec = row.values
            
            # Build feature vector
            df_feat = self.adapter.build_feature_vector(sol_vec, target)
            
            # Apply encoders and scaling
            if target in self.adapter.encoders and self.adapter.encoders[target]:
                from src.data.preprocessors import encode_categoricals
                df_feat, _ = encode_categoricals(df_feat, encoders=self.adapter.encoders[target])
                
            scaled_feat = self.adapter.scalers[target].transform(df_feat)[0]
            
            # Find nearest neighbor
            dist, _ = nn.kneighbors([scaled_feat], n_neighbors=1)
            distances.append(float(dist[0][0]))
            
        return np.array(distances)

    def flag_out_of_distribution(self, pareto_df: pd.DataFrame) -> pd.DataFrame:
        """Adds nn distance and OOD flag columns to pareto_df."""
        augmented = pareto_df.copy()
        
        dist_energy = self.compute_nn_distances(pareto_df, "energy")
        dist_roughness = self.compute_nn_distances(pareto_df, "roughness")
        
        augmented["nn_dist_energy"] = dist_energy
        augmented["nn_dist_roughness"] = dist_roughness
        
        augmented["is_ood_energy"] = dist_energy > self.distance_threshold
        augmented["is_ood_roughness"] = dist_roughness > self.distance_threshold
        augmented["is_ood_any"] = augmented["is_ood_energy"] | augmented["is_ood_roughness"]
        
        return augmented

    def summarize(self, augmented_df: pd.DataFrame) -> dict:
        """Computes summary statistics of the proximity analysis."""
        n_total = len(augmented_df)
        n_ood_energy = int(augmented_df["is_ood_energy"].sum())
        n_ood_roughness = int(augmented_df["is_ood_roughness"].sum())
        n_ood_any = int(augmented_df["is_ood_any"].sum())
        
        pct_within = float((1 - n_ood_any / n_total) * 100.0) if n_total > 0 else 100.0
        
        return {
            "n_total": n_total,
            "n_ood_energy": n_ood_energy,
            "n_ood_roughness": n_ood_roughness,
            "n_ood_any": n_ood_any,
            "pct_within_distribution": pct_within,
            "mean_nn_dist_energy": float(augmented_df["nn_dist_energy"].mean()),
            "mean_nn_dist_roughness": float(augmented_df["nn_dist_roughness"].mean()),
            "max_nn_dist_energy": float(augmented_df["nn_dist_energy"].max()),
            "max_nn_dist_roughness": float(augmented_df["nn_dist_roughness"].max()),
            "threshold_used": self.distance_threshold
        }

    def plot_distance_distribution(self, augmented_df: pd.DataFrame, output_dir: Path) -> Path:
        """Plots distribution of nearest neighbor distances for Pareto solutions."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 5), facecolor='white')
        
        # Plot energy distances
        plt.subplot(1, 2, 1)
        plt.hist(augmented_df["nn_dist_energy"], bins=15, color="orange", alpha=0.7, edgecolor='k')
        plt.axvline(self.distance_threshold, color="red", linestyle="--", label=f"OOD Thresh ({self.distance_threshold})")
        plt.title("Proximity to Energy Train Data", fontsize=10, fontweight='bold')
        plt.xlabel("Euclidean Distance (Normalized)", fontsize=9)
        plt.ylabel("Frequency", fontsize=9)
        plt.legend(fontsize=8)
        
        # Plot roughness distances
        plt.subplot(1, 2, 2)
        plt.hist(augmented_df["nn_dist_roughness"], bins=15, color="green", alpha=0.7, edgecolor='k')
        plt.axvline(self.distance_threshold, color="red", linestyle="--", label=f"OOD Thresh ({self.distance_threshold})")
        plt.title("Proximity to Roughness Train Data", fontsize=10, fontweight='bold')
        plt.xlabel("Euclidean Distance (Normalized)", fontsize=9)
        plt.ylabel("Frequency", fontsize=9)
        plt.legend(fontsize=8)
        
        plt.suptitle("Pareto Solution Proximity to Training Data", fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        plot_path = output_dir / "proximity_distances.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
