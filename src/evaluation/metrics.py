import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, List, Any

def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes R-squared score."""
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return 0.0

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error (RMSE)."""
    try:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception:
        return 0.0

def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error (MAE)."""
    try:
        return float(mean_absolute_error(y_true, y_pred))
    except Exception:
        return 0.0

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Mean Absolute Percentage Error (MAPE).
    Skips samples where y_true is 0.
    """
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        mask = y_true != 0.0
        if not np.any(mask):
            return 0.0
            
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    except Exception:
        return 0.0

def is_mape_reliable(y_true: np.ndarray, threshold: float = 0.01) -> bool:
    """
    Returns False if > 5% of samples have |y_true| < threshold.
    MAPE is unreliable for near-zero targets.
    """
    y_true_abs = np.abs(y_true)
    return float((y_true_abs < threshold).mean()) < 0.05

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_name: str) -> Dict[str, Any]:
    """
    Computes R2, RMSE, MAE, and MAPE metrics and returns them as a dictionary.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    metrics = {
        "target": target_name,
        "r2": compute_r2(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mae": compute_mae(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "n_samples": len(y_true)
    }
    
    if not is_mape_reliable(y_true):
        metrics["mape"] = None
        metrics["mape_note"] = "MAPE suppressed: near-zero targets"
        
    return metrics

def compute_cv_metrics(cv_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregates per-fold scores into mean, std, min, and max.
    
    Args:
        cv_scores: Dict mapping metric name to a list of score floats across folds.
        
    Returns:
        Dict: mapping metric name to {mean, std, min, max} dict.
    """
    results = {}
    for metric, scores in cv_scores.items():
        arr = np.asarray(scores)
        results[metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
    return results
