import logging
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from src.models.base_model import BaseModel

logger = logging.getLogger("BaselineModels")

class LinearRegressionModel(BaseModel):
    """
    Baseline model wrapping scikit-learn Linear Regression.
    """
    
    @property
    def model_name(self) -> str:
        return "linear_regression"
        
    def build(self, params: Dict[str, Any]) -> None:
        """
        Builds the Linear Regression model.
        
        Args:
            params: Dictionary of parameters (ignored for Linear Regression).
        """
        # LinearRegression does not have tuning params for our project baseline
        self.model = LinearRegression()
        self.params = {}

class SVRModel(BaseModel):
    """
    Baseline model wrapping scikit-learn Support Vector Regression (SVR).
    """
    
    @property
    def model_name(self) -> str:
        return "svr"
        
    def build(self, params: Dict[str, Any]) -> None:
        """
        Builds the SVR model.
        
        Args:
            params: SVR parameters.
        """
        default_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'epsilon': 0.1
        }
        # Update default params with overrides
        self.params = {**default_params, **params}
        self.model = SVR(**self.params)
