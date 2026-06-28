from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

class BaseModel(ABC):
    """
    Abstract base for all EnergyOptAI prediction models.
    
    Each model wraps a scikit-learn compatible estimator and
    provides a unified interface for training, prediction,
    saving, and loading.
    """
    
    def __init__(self, target_name: str, params: Optional[Dict[str, Any]] = None):
        """
        Initializes the base model wrapper.
        
        Args:
            target_name: The name of the target variable (e.g., 'energy', 'roughness', 'time').
            params: Dictionary of hyperparameters to override defaults.
        """
        self.target_name = target_name
        self.is_fitted = False
        self.params = params if params is not None else {}
        self.feature_names: List[str] = []
        self.model: Any = None
        
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the unique identifier of the model class."""
        pass
        
    @abstractmethod
    def build(self, params: Dict[str, Any]) -> None:
        """
        Instantiates the underlying estimator with the given params.
        Must set self.model to the instantiated estimator.
        
        Args:
            params: Hyperparameters for the estimator.
        """
        pass
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fits the underlying estimator. Must set self.is_fitted = True.
        
        Args:
            X_train: Training features DataFrame.
            y_train: Training target Series.
        """
        if self.model is None:
            self.build(self.params)
            
        self.feature_names = list(X_train.columns)
        # Handle 1D targets cleanly
        if isinstance(y_train, pd.DataFrame):
            if y_train.shape[1] == 1:
                y_train_fit = y_train.iloc[:, 0]
            else:
                y_train_fit = y_train
        else:
            y_train_fit = y_train
            
        self.X_train = X_train.copy()
        self.y_train = y_train_fit.copy()
        
        self.model.fit(X_train, y_train_fit)
        self.is_fitted = True
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the fitted estimator.
        
        Args:
            X: Input features DataFrame.
            
        Returns:
            np.ndarray: Predicted values.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError(f"Model {self.model_name} is not fitted. Call train() first.")
            
        # Ensure column ordering matches training
        X_ordered = X[self.feature_names]
        return self.model.predict(X_ordered)
        
    def predict_with_interval(
        self,
        X: pd.DataFrame,
        n_bootstrap: int = 200,
        confidence: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (point_pred, lower_bound, upper_bound).
        Uses bootstrap: resample training data n_bootstrap times,
        train a fresh model each time, collect predictions,
        compute percentile intervals.
        Only works if training data is stored: self.X_train, self.y_train.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError(f"Model {self.model_name} is not fitted. Call train() first.")
            
        point_pred = self.predict(X)
        
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train') or self.X_train is None:
            return point_pred, point_pred, point_pred
            
        preds = []
        n_samples = len(self.X_train)
        model_cls = self.__class__
        
        for i in range(n_bootstrap):
            # Resample training data with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_resampled = self.X_train.iloc[indices].reset_index(drop=True)
            y_resampled = self.y_train.iloc[indices].reset_index(drop=True)
            
            # Create a new instance and train
            bootstrap_model = model_cls(target_name=self.target_name, params=self.params)
            bootstrap_model.build(self.params)
            bootstrap_model.train(X_resampled, y_resampled)
            
            # Predict
            pred = bootstrap_model.predict(X)
            preds.append(pred)
            
        preds = np.array(preds)
        
        alpha = 1.0 - confidence
        lower_pct = 100 * (alpha / 2.0)
        upper_pct = 100 * (1.0 - alpha / 2.0)
        
        lower_bound = np.percentile(preds, lower_pct, axis=0)
        upper_bound = np.percentile(preds, upper_pct, axis=0)
        
        return point_pred, lower_bound, upper_bound
        
    def get_params(self) -> Dict[str, Any]:
        """Returns current hyperparameters of the model wrapper."""
        return self.params
        
    def save(self, path: Path) -> None:
        """
        Saves the model to a joblib file. Creates parent directories if needed.
        
        Args:
            path: Destination file path or directory.
        """
        if path.is_dir():
            # Construct standard name if a directory is passed
            filename = f"{self.model_name}_{self.target_name}_final.pkl"
            full_path = path / filename
        else:
            full_path = path
            
        full_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, full_path)
        
    def load(self, path: Path) -> None:
        """
        Loads the model from a joblib file and copies attributes.
        
        Args:
            path: Source file path.
        """
        loaded_wrapper = joblib.load(path)
        self.model = loaded_wrapper.model
        self.is_fitted = loaded_wrapper.is_fitted
        self.params = loaded_wrapper.params
        self.feature_names = loaded_wrapper.feature_names
        self.target_name = loaded_wrapper.target_name
        
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluates the model on test set using project metrics.
        
        Args:
            X_test: Test features.
            y_test: Test targets.
            
        Returns:
            Dict[str, float]: Performance metrics (R2, RMSE, MAE, MAPE).
        """
        # We import compute_all_metrics dynamically to avoid circular import issues
        from src.evaluation.metrics import compute_all_metrics
        
        y_pred = self.predict(X_test)
        
        # Flatten target Series/DataFrame if necessary
        if isinstance(y_test, (pd.Series, pd.DataFrame)):
            y_true_arr = y_test.values.flatten()
        else:
            y_true_arr = np.asarray(y_test).flatten()
            
        return compute_all_metrics(y_true_arr, y_pred.flatten(), self.target_name)
