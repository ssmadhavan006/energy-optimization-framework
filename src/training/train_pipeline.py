import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from src.models.model_registry import TARGET_CONFIG, MODEL_REGISTRY
from src.data.loaders import load_all
from src.data.feature_engineering import engineer_all_features
from src.data.preprocessors import (
    impute_missing,
    detect_outliers_iqr,
    encode_categoricals,
    scale_features,
    split_data
)
from src.training.cross_validator import CrossValidator
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.models.base_model import BaseModel
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainPipeline")

class TrainPipeline:
    """
    Orchestration class that runs the full train-tune-evaluate sequence
    for one model family on one target.
    """
    
    def __init__(
        self,
        model_class,
        target_name: str,
        run_tuning: bool = True,
        n_trials: int = 50,
        test_size: float = 0.2,
        random_state: int = 42,
        save_suffix: str = ""
    ):
        self.model_class = model_class
        self.target_name = target_name.lower()
        self.run_tuning_flag = run_tuning
        self.n_trials = n_trials
        self.test_size = test_size
        self.random_state = random_state
        self.save_suffix = save_suffix
        
        # Load config
        if self.target_name not in TARGET_CONFIG:
            raise ValueError(f"Unknown target: {self.target_name}")
        self.config = TARGET_CONFIG[self.target_name]
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Loads the dataset, engineers features, splits into train/test,
        preprocesses (imputation, encoding, outlier removal, scaling).
        """
        logger.info(f"Preparing data for target: {self.target_name}...")
        
        # 1. Load correct dataset
        datasets = load_all()
        dataset_name = self.config["dataset"]
        
        if dataset_name == "mendeley":
            df = datasets["mendeley"]["parsed"]
            target_col = self.config["target_col"]
            if target_col in ["sec", "spindle_power_w"]:
                from src.data.feature_engineering import engineer_mendeley_energy_target
                original_n = len(df)
                df = engineer_mendeley_energy_target(df)
                logger.info(f"Energy target re-engineered to {target_col}. Dataset reduced from {original_n} to {len(df)} rows.")
            else:
                # Target engineering: compute total energy in kWh (raw is Ws/Joules)
                # Power sum * 0.002s is in Ws. kWh = Ws / 3.6e6
                energy_sum_ws = (df["ENERGY|x"] + df["ENERGY|y"] + df["ENERGY|z"] + 
                                 df["ENERGY|S"] + df["ENERGY|T"])
                df["total_energy_kwh"] = energy_sum_ws / 3.6e6
            
        elif dataset_name == "kaggle":
            df = datasets["kaggle"]
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        # 2. Engineer features
        df = engineer_all_features(df, dataset_name)
        
        # 3. Drop rows where target column is null
        target_col = self.config["target_col"]
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in {dataset_name} columns.")
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
        
        # 4. Identify features and target, and drop descriptive columns
        if dataset_name == "mendeley":
            drop_cols = [
                'ENERGY|x', 'ENERGY|y', 'ENERGY|z', 'ENERGY|S', 'ENERGY|T', 
                'Workpiece', 'Filename', 'total_energy_kwh', 'sec', 'spindle_power_w',
                'operation_group', 'is_cutting', 'block_change', 'file_change',
                'n_blocks_aggregated', 'mrr'
            ]
        else:
            drop_cols = [
                'Run_ID', 'Experiment', 'Replica', 'Group', 'Subgroup', 
                'Source', 'Ra', 'Rz', 'Rsk', 'Rku', 'RSm', 'Rt', 'CTime'
            ]
            
        # 5. Split train/test first (crucial to prevent leak)
        X_train, X_test, y_train, y_test = split_data(
            df, target_cols=[target_col], test_size=self.test_size, seed=self.random_state
        )
        
        # Drop columns that are 100% NaN in training features (like Fx, Fy, Condition for CTime)
        nan_cols = [c for c in X_train.columns if X_train[c].isnull().all()]
        if nan_cols:
            logger.info(f"Dropping columns that are 100% NaN in training features: {nan_cols}")
            X_train = X_train.drop(columns=nan_cols)
            X_test = X_test.drop(columns=nan_cols)
            
        # 6. Imputation: fit on train, transform test
        # In preprocessors, impute_missing fills based on stats. Let's do it train then test
        X_train = impute_missing(X_train)
        X_test = impute_missing(X_test)
        
        # 7. Outlier removal on training set ONLY
        # Identify numeric columns for outlier check
        numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and c not in drop_cols]
        outlier_mask = detect_outliers_iqr(X_train, numeric_cols)
        logger.info(f"Removing {outlier_mask.sum()} outliers from training set using IQR method.")
        X_train = X_train[~outlier_mask].reset_index(drop=True)
        y_train = y_train[~outlier_mask].reset_index(drop=True)
        
        # 8. Encode categoricals: fit on train, transform test
        X_train, encoders = encode_categoricals(X_train)
        X_test, _ = encode_categoricals(X_test, encoders=encoders)
        
        # Drop columns after encoding and cleaning
        feat_cols = [c for c in X_train.columns if c not in drop_cols]
        X_train = X_train[feat_cols]
        X_test = X_test[feat_cols]
        
        # 9. Scaling: fit on train, transform test
        model_name = self.model_class(target_name=self.target_name).model_name
        scaler_name = f"{self.target_name}_{model_name}_scaler.joblib"
        X_train, X_test, scaler = scale_features(
            X_train, X_test, feat_cols, method='minmax', save_name=scaler_name
        )
        
        # Save encoders
        encoder_dir = Path("outputs/models/scalers")
        encoder_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, encoder_dir / f"{self.target_name}_{model_name}_encoders.joblib")
        
        logger.info(f"Data prepared: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, X_test, y_train.iloc[:, 0], y_test.iloc[:, 0]
        
    def run_baseline_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Runs K-fold CV on baseline models and returns metrics."""
        from src.models.baseline_models import LinearRegressionModel, SVRModel
        
        results = {}
        validator = CrossValidator(n_splits=5)
        
        for base_cls in [LinearRegressionModel, SVRModel]:
            base_model = base_cls(target_name=self.target_name)
            res = validator.run(base_model, X_train, y_train, self.target_name)
            results[base_model.model_name] = res
            
        return results
        
    def run_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Runs Optuna hyperparameter tuning study."""
        tuner = HyperparameterTuner(
            model_class=self.model_class,
            X_train=X_train,
            y_train=y_train,
            target_name=self.target_name,
            n_trials=self.n_trials
        )
        best_params = tuner.optimize()
        return best_params
        
    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series, params: Dict[str, Any]) -> BaseModel:
        """Trains final model on all training data with best params."""
        model = self.model_class(target_name=self.target_name, params=params)
        model.build(params)
        
        logger.info(f"Training final {model.model_name} model for {self.target_name}...")
        model.train(X_train, y_train)
        
        # Save final model
        trained_dir = Path("outputs/models/trained")
        trained_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = trained_dir / f"{self.target_name}_{model.model_name}{self.save_suffix}_final.pkl"
        model.save(model_save_path)
        logger.info(f"Saved trained model to {model_save_path}")
        
        return model
        
    def evaluate(self, model: BaseModel, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluates model on held-out test set and saves metrics."""
        logger.info(f"Evaluating final {model.model_name} on held-out test set...")
        metrics = model.evaluate(X_test, y_test)
        
        # Save metrics to JSON
        metrics_dir = Path("outputs/results/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_json_path = metrics_dir / f"{self.target_name}_{model.model_name}{self.save_suffix}_test_metrics.json"
        
        import json
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved test set metrics to {metrics_json_path}")
        
        # Save X_test and y_test to outputs/results/ for downstream use (like bootstrap intervals)
        results_dir = Path("outputs/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        X_test.to_csv(results_dir / f"{self.target_name}_X_test.csv", index=False)
        y_test.to_csv(results_dir / f"{self.target_name}_y_test.csv", index=False)
        logger.info(f"Saved test features and targets to {results_dir}")
        
        return metrics
        
    def run_all(self) -> Dict[str, Any]:
        """Runs the entire pipeline (prepare -> baseline CV -> tuning -> train -> evaluate)."""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # 1. Baseline CV
        logger.info("Evaluating baseline models (Linear Reg & SVR)...")
        baseline_results = self.run_baseline_cv(X_train, y_train)
        
        # 2. Hyperparameter Tuning
        best_params = {}
        if self.run_tuning_flag:
            logger.info(f"Tuning ensemble model {self.model_class(target_name=self.target_name).model_name}...")
            best_params = self.run_tuning(X_train, y_train)
            
        # 3. Train final model
        model = self.train_final_model(X_train, y_train, best_params)
        
        # 4. Evaluate
        test_metrics = self.evaluate(model, X_test, y_test)
        
        return {
            "baseline_metrics": baseline_results,
            "best_params": best_params,
            "test_metrics": test_metrics
        }
