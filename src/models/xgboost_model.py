import logging
from typing import Dict, Any, Optional
from xgboost import XGBRegressor
from src.models.base_model import BaseModel

logger = logging.getLogger("XGBoostModel")

class XGBoostModel(BaseModel):
    """
    Model wrapper for XGBoost Regressor.
    """
    
    @property
    def model_name(self) -> str:
        return "xgboost"
        
    def build(self, params: Dict[str, Any]) -> None:
        """
        Builds the XGBoost regressor model.
        
        Args:
            params: XGBoost parameters.
        """
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'rmse'
        }
        self.params = {**default_params, **params}
        self.model = XGBRegressor(**self.params)
        
    def train(self, X_train, y_train, eval_set=None) -> None:
        """
        Fits XGBoost regressor. Handles early stopping if eval_set is provided.
        """
        if self.model is None:
            self.build(self.params)
            
        self.feature_names = list(X_train.columns)
        
        # Flatten target
        y_train_fit = y_train.values.flatten() if hasattr(y_train, 'values') else y_train
        
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_fit = y_val.values.flatten() if hasattr(y_val, 'values') else y_val
            # Fit with early stopping
            self.model.fit(
                X_train, y_train_fit, 
                eval_set=[(X_val, y_val_fit)], 
                verbose=False
            )
        else:
            # If no eval_set, rebuild without early stopping/eval_metric to avoid warnings/crashes
            clean_params = {k: v for k, v in self.params.items() if k not in ['early_stopping_rounds', 'eval_metric']}
            temp_model = XGBRegressor(**clean_params)
            temp_model.fit(X_train, y_train_fit)
            self.model = temp_model
            
        self.is_fitted = True

    @classmethod
    def optuna_search_space(cls, trial) -> Dict[str, Any]:
        """
        Defines the Optuna search space for XGBoost.
        """
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
