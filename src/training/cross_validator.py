import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from src.models.base_model import BaseModel
from src.evaluation.metrics import compute_all_metrics, compute_cv_metrics
from typing import Dict, Any, List

logger = logging.getLogger("CrossValidator")

class CrossValidator:
    """
    5-fold cross-validation wrapper for all BaseModel subclasses.
    Fits scalers fold-by-fold to prevent data leakage.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        
    def run(self, model: BaseModel, X: pd.DataFrame, y: pd.Series, target_name: str) -> Dict[str, Any]:
        """
        Runs K-fold CV, returns per-fold and aggregated metrics.
        """
        logger.info(f"Starting {self.n_splits}-fold CV for {model.model_name} on {target_name}...")
        
        fold_metrics = []
        cv_scores: Dict[str, List[float]] = {
            "r2": [],
            "rmse": [],
            "mae": [],
            "mape": []
        }
        
        X_arr = X.values
        # Flatten target
        y_arr = y.values.flatten() if hasattr(y, 'values') else np.asarray(y).flatten()
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kf.split(X_arr)):
            # Split data
            X_train_f, X_val_f = X_arr[train_idx], X_arr[val_idx]
            y_train_f, y_val_f = y_arr[train_idx], y_arr[val_idx]
            
            # 1. Scaling: Fit scaler on training fold ONLY and transform both
            scaler = MinMaxScaler()
            X_train_f_scaled = scaler.fit_transform(X_train_f)
            X_val_f_scaled = scaler.transform(X_val_f)
            
            # Re-convert to DataFrames to preserve interface compatibility
            df_X_train_f = pd.DataFrame(X_train_f_scaled, columns=X.columns)
            df_X_val_f = pd.DataFrame(X_val_f_scaled, columns=X.columns)
            
            ser_y_train_f = pd.Series(y_train_f)
            ser_y_val_f = pd.Series(y_val_f)
            
            # 2. Build model instance fresh with parameters
            model.build(model.params)
            
            # 3. Fit
            # For gradient boosting we can pass eval_set to prevent overfitting
            if model.model_name in ['xgboost', 'catboost']:
                model.train(df_X_train_f, ser_y_train_f, eval_set=(df_X_val_f, ser_y_val_f))
            else:
                model.train(df_X_train_f, ser_y_train_f)
                
            # 4. Predict & Evaluate
            y_pred_f = model.predict(df_X_val_f)
            metrics = compute_all_metrics(y_val_f, y_pred_f, target_name)
            
            fold_metrics.append(metrics)
            cv_scores["r2"].append(metrics["r2"])
            cv_scores["rmse"].append(metrics["rmse"])
            cv_scores["mae"].append(metrics["mae"])
            cv_scores["mape"].append(metrics["mape"])
            
            logger.debug(f"Fold {fold_idx+1} RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
            
        aggregate = compute_cv_metrics(cv_scores)
        
        # Save fold metrics to outputs/results/metrics/
        import json
        out_dir = Path("outputs/results/metrics")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv_path = out_dir / f"{target_name}_{model.model_name}_cv_folds.json"
        with open(cv_path, "w", encoding="utf-8") as f:
            json.dump(fold_metrics, f, indent=4)
        logger.info(f"Saved CV fold metrics to {cv_path}")
        
        return {
            "model_name": model.model_name,
            "target_name": target_name,
            "fold_metrics": fold_metrics,
            "aggregate": aggregate
        }
