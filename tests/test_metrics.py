import pytest
import numpy as np
from src.evaluation.metrics import (
    compute_r2,
    compute_rmse,
    compute_mae,
    compute_mape,
    compute_all_metrics,
    compute_cv_metrics
)

def test_compute_r2():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    # Basic correctness check
    r2 = compute_r2(y_true, y_pred)
    assert isinstance(r2, float)
    assert 0.0 < r2 <= 1.0
    
    # Check exception handling with invalid inputs
    assert compute_r2(None, y_pred) == 0.0

def test_compute_rmse():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    # Expected squared errors: 0.25, 0.25, 0.0, 1.0. Mean: 1.5 / 4 = 0.375. Sqrt: sqrt(0.375) approx 0.61237
    rmse = compute_rmse(y_true, y_pred)
    assert isinstance(rmse, float)
    assert np.isclose(rmse, np.sqrt(0.375))
    
    # Check exception handling
    assert compute_rmse(None, y_pred) == 0.0

def test_compute_mae():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    # Absolute errors: 0.5, 0.5, 0.0, 1.0. Mean: 2.0 / 4 = 0.5
    mae = compute_mae(y_true, y_pred)
    assert isinstance(mae, float)
    assert np.isclose(mae, 0.5)
    
    # Check exception handling
    assert compute_mae(None, y_pred) == 0.0

def test_compute_mape():
    # 0 values should be ignored by the mask
    y_true = np.array([100.0, 0.0, 200.0])
    y_pred = np.array([90.0, 50.0, 220.0])
    # Relative abs errors: |100-90|/100 = 10%, |200-220|/200 = 10%. Mean percentage = 10%
    mape = compute_mape(y_true, y_pred)
    assert isinstance(mape, float)
    assert np.isclose(mape, 10.0)
    
    # All zeros
    assert compute_mape(np.array([0.0, 0.0]), np.array([1.0, 2.0])) == 0.0
    
    # Check exception handling
    assert compute_mape(None, y_pred) == 0.0

def test_compute_all_metrics():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([9.0, 21.0])
    res = compute_all_metrics(y_true, y_pred, "energy")
    
    assert isinstance(res, dict)
    assert res["target"] == "energy"
    assert "r2" in res
    assert "rmse" in res
    assert "mae" in res
    assert "mape" in res
    assert res["n_samples"] == 2

def test_compute_cv_metrics():
    cv_scores = {
        "rmse": [0.1, 0.2, 0.3],
        "r2": [0.8, 0.9, 0.7]
    }
    agg = compute_cv_metrics(cv_scores)
    
    assert isinstance(agg, dict)
    assert "rmse" in agg
    assert "r2" in agg
    
    rmse_stats = agg["rmse"]
    assert np.isclose(rmse_stats["mean"], 0.2)
    assert np.isclose(rmse_stats["min"], 0.1)
    assert np.isclose(rmse_stats["max"], 0.3)
    assert rmse_stats["std"] > 0.0

def test_is_mape_reliable():
    from src.evaluation.metrics import is_mape_reliable
    # All reliable (no near-zeros)
    y_true_ok = np.array([10.0, 15.0, 20.0])
    assert is_mape_reliable(y_true_ok) is True
    
    # Unreliable (too many near-zeros, e.g. 50% under 0.01)
    y_true_bad = np.array([0.001, 10.0])
    assert is_mape_reliable(y_true_bad) is False
