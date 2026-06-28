import logging
from pathlib import Path
import pandas as pd
import optuna
from src.training.cross_validator import CrossValidator
from typing import Dict, Any, Optional

# Disable Optuna spam logs by default (keep warning and error)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger("HyperparameterTuner")

class HyperparameterTuner:
    """
    Optuna-based Bayesian optimization for model hyperparameters.
    """
    
    def __init__(
        self,
        model_class,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_name: str,
        n_trials: int = 50,
        cv_folds: int = 5,
        direction: str = "minimize",
        study_name: Optional[str] = None,
        storage: Optional[str] = None
    ):
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.target_name = target_name
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.direction = direction
        
        # Configure standard naming and storage paths
        self.study_name = study_name or f"{target_name}_{model_class(target_name=target_name).model_name}_study"
        self.storage = storage or "sqlite:///outputs/results/optuna_studies.db"
        self.best_params: Optional[Dict[str, Any]] = None
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function targeting mean CV RMSE minimization."""
        # 1. Sample parameters from the model search space
        params = self.model_class.optuna_search_space(trial)
        
        # 2. Instantiate wrapper model
        model = self.model_class(target_name=self.target_name, params=params)
        
        # 3. K-fold CV
        validator = CrossValidator(n_splits=self.cv_folds)
        results = validator.run(model, self.X_train, self.y_train, self.target_name)
        
        # 4. Return mean RMSE score for optimization target
        mean_rmse = results["aggregate"]["rmse"]["mean"]
        return mean_rmse
        
    def optimize(self) -> Dict[str, Any]:
        """Runs the optimization trials and returns the best params dict."""
        logger.info(f"Starting Optuna study '{self.study_name}' with {self.n_trials} trials...")
        
        # Ensure outputs directories exist
        Path("outputs/results").mkdir(parents=True, exist_ok=True)
        
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Create or load the SQLite persistent study
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            pruner=pruner,
            load_if_exists=True
        )
        
        study.optimize(self.objective, n_trials=self.n_trials)
        
        self.best_params = study.best_params
        logger.info(f"Study completed. Best trial RMSE: {study.best_value:.4f}")
        
        # Save study log to CSV
        study_df = study.trials_dataframe()
        csv_path = Path("outputs/results/metrics") / f"{self.target_name}_{self.model_class(target_name=self.target_name).model_name}_optuna_study.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        study_df.to_csv(csv_path, index=False)
        logger.info(f"Saved study trials DataFrame to {csv_path}")
        
        return self.best_params
        
    def get_best_params(self) -> Dict[str, Any]:
        """Returns the best params found. Raises error if optimize has not run."""
        if self.best_params is None:
            raise ValueError("Optimization has not been run. Call optimize() first.")
        return self.best_params
