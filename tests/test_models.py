import pytest
import numpy as np
from pathlib import Path
import pandas as pd
from src.models.model_registry import MODEL_REGISTRY

@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(20, 4), columns=["feat1", "feat2", "feat3", "feat4"])
    # y target
    y = pd.Series(X["feat1"] * 2.0 - X["feat2"] * 1.5 + np.random.normal(0, 0.05, 20), name="target")
    return X, y

@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_model_lifecycle(model_key, synthetic_data, tmp_path):
    X, y = synthetic_data
    model_cls = MODEL_REGISTRY[model_key]
    
    # 1. Instantiation
    model_inst = model_cls(target_name="energy")
    assert model_inst.target_name == "energy"
    assert model_inst.model_name == model_key
    assert not model_inst.is_fitted
    
    # 2. Build model with empty params (uses defaults)
    model_inst.build({})
    assert model_inst.model is not None
    
    # 3. Fit model on training data
    model_inst.train(X, y)
    assert model_inst.is_fitted
    assert len(model_inst.feature_names) == 4
    
    # 4. Predict
    preds = model_inst.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (20,)
    
    # 5. Evaluate
    evals = model_inst.evaluate(X, y)
    assert isinstance(evals, dict)
    assert evals["target"] == "energy"
    assert "r2" in evals
    assert "rmse" in evals
    assert "mae" in evals
    assert "mape" in evals
    
    # 6. Save and load
    save_file = tmp_path / f"test_model_{model_key}.pkl"
    model_inst.save(save_file)
    assert save_file.exists()
    
    # Load back into a new wrapper instance
    loaded_inst = model_cls(target_name="energy")
    loaded_inst.load(save_file)
    assert loaded_inst.is_fitted
    assert loaded_inst.feature_names == model_inst.feature_names
    assert loaded_inst.target_name == "energy"
    
    # Predict again
    loaded_preds = loaded_inst.predict(X)
    assert np.allclose(preds, loaded_preds)
