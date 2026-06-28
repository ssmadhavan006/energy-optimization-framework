import logging
from typing import Dict, Any, Optional
from catboost import CatBoostRegressor
from src.models.base_model import BaseModel

logger = logging.getLogger("CatBoostModel")

class CatBoostModel(BaseModel):
    """
    Model wrapper for CatBoost Regressor.
    """
    
    @property
    def model_name(self) -> str:
        return "catboost"
        
    def build(self, params: Dict[str, Any]) -> None:
        """
        Builds the CatBoost regressor model.
        
        Args:
            params: CatBoost parameters.
        """
        default_params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.05,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': 0,
            'allow_writing_files': False
        }
        self.params = {**default_params, **params}
        self.model = CatBoostRegressor(**self.params)
        
    def train(self, X_train, y_train, eval_set=None) -> None:
        """
        Fits CatBoost regressor. Handles early stopping if eval_set is provided.
        """
        if self.model is None:
            self.build(self.params)
            
        self.feature_names = list(X_train.columns)
        
        # Flatten target
        y_train_fit = y_train.values.flatten() if hasattr(y_train, 'values') else y_train
        
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_fit = y_val.values.flatten() if hasattr(y_val, 'values') else y_val
            # Fit with early stopping (CatBoost calls it early_stopping_rounds)
            self.model.fit(
                X_train, y_train_fit,
                eval_set=(X_val, y_val_fit),
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train_fit, verbose=False)
            
        self.is_fitted = True

    @classmethod
    def optuna_search_space(cls, trial) -> Dict[str, Any]:
        """
        Defines the Optuna search space for CatBoost.
        """
        return {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=50),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255)
        }
