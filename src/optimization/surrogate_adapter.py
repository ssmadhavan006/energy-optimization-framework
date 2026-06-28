import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from src.models.base_model import BaseModel
from src.data.preprocessors import encode_categoricals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SurrogateAdapter")

class SurrogateAdapter:
    """
    Maps a unified machining parameter vector to the specific
    feature format required by each trained prediction model,
    then returns the model's prediction.
    """

    def __init__(
        self,
        models: Dict[str, BaseModel],
        scalers: Dict[str, Any],
        encoders: Dict[str, Dict[str, Any]],
        feature_names: Dict[str, List[str]],
        train_stats: Dict[str, Dict[str, float]],
        decision_space: Dict[str, Any],
        X_train_data: Dict[str, pd.DataFrame] = None,
        clip_to_bounds: bool = True
    ):
        self.models = models
        self.scalers = scalers
        self.encoders = encoders
        self.feature_names = feature_names
        self.train_stats = train_stats
        self.decision_space = decision_space
        self.X_train_data = X_train_data
        self.clip_to_bounds = clip_to_bounds

    def build_feature_vector(
        self,
        decision_vector: np.ndarray,
        target: str
    ) -> pd.DataFrame:
        """
        Constructs the feature DataFrame required by the named
        target's model from the decision vector.
        """
        # 1. Unpack decision_vector
        # Order: feed_rate (f), depth_of_cut (ap), spindle_speed (S), tool_condition (TCond)
        f_val = float(decision_vector[0])
        ap_val = float(decision_vector[1])
        S_val = float(decision_vector[2])
        TCond_val = float(decision_vector[3])

        # 2. Build feature values dict
        feat_dict = {}
        target_features = self.feature_names[target]

        for feat in target_features:
            # Check mappings
            if feat == "f":
                feat_dict[feat] = f_val
            elif feat == "ap":
                feat_dict[feat] = ap_val
            elif feat == "TCond":
                feat_dict[feat] = TCond_val
            elif feat == "S":
                feat_dict[feat] = S_val
            elif feat == "vc":
                # For Kaggle models, vc is cutting speed. If the model saw constant vc (350.0),
                # we should use the train median to avoid extrapolation!
                feat_dict[feat] = self.train_stats[target].get("vc", {}).get("median", 350.0)
            elif feat == "F_val":
                # Mendeley feed rate: f (mm/rev) * S (rpm)
                feat_dict[feat] = f_val * S_val
            elif feat == "cutting_speed_vc":
                # Mendeley cutting speed: pi * D_W * S / 1000. D_W median is 16.0
                dw = self.train_stats["energy"].get("D_W", {}).get("median", 16.0)
                feat_dict[feat] = (np.pi * dw * S_val) / 1000.0
            elif feat == "time_s" and target == "energy":
                # Use block duration median from Mendeley training data
                feat_dict[feat] = self.train_stats["energy"].get("time_s", {}).get("median", 4.62)
            else:
                # If the feature is not in the decision space, use train_stats median as proxy
                if feat in self.train_stats[target]:
                    feat_dict[feat] = self.train_stats[target][feat]["median"]
                else:
                    logger.debug(f"Feature '{feat}' not found in train_stats for target '{target}'. Using 0.0.")
                    feat_dict[feat] = 0.0
                logger.debug(f"Using median proxy for feature {feat} in {target} model")

        # 3. Return as single-row DataFrame with correct column order matching training
        df_feat = pd.DataFrame([feat_dict], columns=target_features)
        
        # Apply categorical encoders if present
        if target in self.encoders and self.encoders[target]:
            df_feat, _ = encode_categoricals(df_feat, encoders=self.encoders[target])
            
        return df_feat

    def predict_all(
        self,
        decision_vector: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Returns (energy_sec, roughness_ra, time_s) predictions
        for a given decision vector.
        """
        # Optional clipping to bounds before predicting
        if self.clip_to_bounds:
            decision_vector = self.clip_decision_vector(decision_vector)

        predictions = {}
        for target in ["energy", "roughness", "time"]:
            try:
                # Build raw feature row
                df_feat = self.build_feature_vector(decision_vector, target)
                
                # Apply scaler
                scaler = self.scalers[target]
                scaled_feat_arr = scaler.transform(df_feat)
                df_scaled = pd.DataFrame(scaled_feat_arr, columns=df_feat.columns)
                
                # Query model
                pred_val = self.models[target].predict(df_scaled)[0]
                
                # Clip negative predictions to 0 (unphysical)
                predictions[target] = max(0.0, float(pred_val))
            except Exception as e:
                logger.error(f"Prediction failed for target '{target}' with decision vector {decision_vector}: {e}")
                predictions[target] = float('inf')

        return predictions["energy"], predictions["roughness"], predictions["time"]

    def clip_decision_vector(self, decision_vector: np.ndarray) -> np.ndarray:
        """Clips decision vector parameters to the defined decision space bounds."""
        clipped = np.copy(decision_vector)
        for i, (key, val) in enumerate(self.decision_space.items()):
            bounds = val["bounds"]
            clipped[i] = np.clip(clipped[i], bounds[0], bounds[1])
        return clipped

    def validate_decision_bounds(
        self,
        decision_vector: np.ndarray
    ) -> bool:
        """
        Checks all decision variables are within training bounds.
        Returns True if valid, False if any variable is out of bounds.
        """
        for i, (key, val) in enumerate(self.decision_space.items()):
            bounds = val["bounds"]
            if decision_vector[i] < bounds[0] - 1e-6 or decision_vector[i] > bounds[1] + 1e-6:
                return False
        return True

    def get_baseline_prediction(self) -> Dict[str, Any]:
        """
        Predicts all three targets using the median values of
        all decision variables from the training set.
        """
        # Order: feed_rate (f), depth_of_cut (ap), spindle_speed (S), tool_condition (TCond)
        baseline_vec = np.array([
            self.train_stats["roughness"]["f"]["median"],
            self.train_stats["roughness"]["ap"]["median"],
            self.train_stats["energy"]["S"]["median"],
            self.train_stats["roughness"]["TCond"]["median"]
        ])
        
        energy_pred, roughness_pred, time_pred = self.predict_all(baseline_vec)
        
        return {
            "decision_vector": baseline_vec.tolist(),
            "energy_sec": energy_pred,
            "roughness_ra": roughness_pred,
            "time_s": time_pred,
            "description": "Median baseline (dataset average)"
        }

    def check_prediction_confidence(self, decision_vector: np.ndarray, threshold: float = 0.5) -> Tuple[float, bool]:
        """
        Computes the minimum Euclidean distance of the scaled feature vector
        to the training data samples for the energy model.
        Returns (min_distance, is_confident).
        """
        if self.X_train_data is None or "energy" not in self.X_train_data:
            return 0.0, True
            
        df_feat = self.build_feature_vector(decision_vector, "energy")
        
        # Apply categorical encoders if present
        if "energy" in self.encoders and self.encoders["energy"]:
            df_feat, _ = encode_categoricals(df_feat, encoders=self.encoders["energy"])
            
        scaled_feat = self.scalers["energy"].transform(df_feat)[0]
        
        # Compute distance to all training samples
        # self.X_train_data["energy"] is already scaled (preprocessed X_train)
        X_train_scaled = self.X_train_data["energy"].values
        dists = np.linalg.norm(X_train_scaled - scaled_feat, axis=1)
        min_dist = float(np.min(dists))
        
        # If min_dist is below threshold, we have high confidence
        is_confident = min_dist <= threshold
        return min_dist, is_confident

    def predict_batch(
        self,
        x: np.ndarray,
        t_cond_val: float
    ) -> np.ndarray:
        """
        Predicts objectives for the entire batch of decision vectors.
        x: np.ndarray of shape (N, 3), columns: feed_rate, depth_of_cut, spindle_speed
        Returns: np.ndarray of shape (N, 3), columns: energy, roughness, time
        """
        N = len(x)
        predictions = {}
        
        for target in ["energy", "roughness", "time"]:
            target_features = self.feature_names[target]
            feat_dict = {}
            
            for feat in target_features:
                if feat == "f":
                    feat_dict[feat] = x[:, 0]
                elif feat == "ap":
                    feat_dict[feat] = x[:, 1]
                elif feat == "TCond":
                    feat_dict[feat] = np.full(N, t_cond_val)
                elif feat == "S":
                    feat_dict[feat] = x[:, 2]
                elif feat == "vc":
                    feat_dict[feat] = np.full(N, self.train_stats[target].get("vc", {}).get("median", 350.0))
                elif feat == "F_val":
                    feat_dict[feat] = x[:, 0] * x[:, 2]
                elif feat == "cutting_speed_vc":
                    dw = self.train_stats["energy"].get("D_W", {}).get("median", 16.0)
                    feat_dict[feat] = (np.pi * dw * x[:, 2]) / 1000.0
                elif feat == "time_s" and target == "energy":
                    feat_dict[feat] = np.full(N, self.train_stats["energy"].get("time_s", {}).get("median", 4.62))
                else:
                    if feat in self.train_stats[target]:
                        # Handle array assignment properly
                        feat_dict[feat] = [self.train_stats[target][feat]["median"]] * N
                    else:
                        feat_dict[feat] = np.zeros(N)
                        
            df_feat = pd.DataFrame(feat_dict, columns=target_features)
            
            # Apply encoders if present
            if target in self.encoders and self.encoders[target]:
                from src.data.preprocessors import encode_categoricals
                df_feat, _ = encode_categoricals(df_feat, encoders=self.encoders[target])
                
            # Apply scaler
            scaled_feat_arr = self.scalers[target].transform(df_feat)
            df_scaled = pd.DataFrame(scaled_feat_arr, columns=target_features)
            
            # Predict
            pred_val = self.models[target].predict(df_scaled)
            predictions[target] = np.maximum(0.0, pred_val)
            
        return np.column_stack([predictions["energy"], predictions["roughness"], predictions["time"]])
