import logging
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
from src.models.base_model import BaseModel

logger = logging.getLogger("RandomForestModel")

class RandomForestModel(BaseModel):
    """
    Model wrapper for Random Forest Regressor.
    """
    
    @property
    def model_name(self) -> str:
        return "random_forest"
        
    def build(self, params: Dict[str, Any]) -> None:
        """
        Builds the Random Forest regressor model.
        
        Args:
            params: Random Forest parameters.
        """
        default_params = {
            'n_estimators': 300,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        self.params = {**default_params, **params}
        self.model = RandomForestRegressor(**self.params)
        
    def build_for_tuning(self, params: Dict[str, Any]) -> None:
        """Helper to build for Optuna where max_depth is integer only."""
        self.build(params)

    @classmethod
    def optuna_search_space(cls, trial) -> Dict[str, Any]:
        """
        Defines the Optuna search space for Random Forest.
        """
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7])
        }
