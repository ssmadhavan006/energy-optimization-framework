from src.models.baseline_models import LinearRegressionModel, SVRModel
from src.models.xgboost_model import XGBoostModel
from src.models.catboost_model import CatBoostModel
from src.models.random_forest_model import RandomForestModel

# Model registry mapping keys to their corresponding wrapper classes
MODEL_REGISTRY = {
    "linear_regression": LinearRegressionModel,
    "svr": SVRModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "random_forest": RandomForestModel,
}

# Supported optimization/prediction targets
TARGETS = ["energy", "roughness", "time"]

# Target configuration metadata
TARGET_CONFIG = {
    "energy": {
        "dataset": "mendeley",
        "target_col": "sec",
        "unit": "J/mm³",
        "minimize": True,
        "description": "Specific Energy Consumption per unit MRR",
        "engineering_fn": "engineer_mendeley_energy_target"
    },
    "roughness": {
        "dataset": "kaggle",
        "target_col": "Ra",
        "unit": "μm",
        "minimize": True,
    },
    "time": {
        "dataset": "kaggle",
        "target_col": "CTime",
        "unit": "s",
        "minimize": True,
    }
}
