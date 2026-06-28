import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from src.optimization.decision_space import DECISION_SPACE
from src.optimization.surrogate_adapter import SurrogateAdapter
from src.optimization.nsga2_optimizer import EnergyOptNSGA2
from scripts.run_optimization import compute_target_train_stats, compute_target_train_data

@pytest.fixture
def loaded_adapter():
    model_dir = Path("outputs/models/trained")
    scaler_dir = Path("outputs/models/scalers")
    
    models = {}
    scalers = {}
    encoders = {}
    feature_names = {}
    train_stats = {}
    train_data = {}
    
    for target in ["roughness", "time", "energy"]:
        m_key = "catboost" if target != "energy" else "random_forest"
        suffix = "_sec" if target == "energy" else ""
        
        models[target] = joblib.load(model_dir / f"{target}_{m_key}{suffix}_final.pkl")
        feature_names[target] = models[target].feature_names
        scalers[target] = joblib.load(scaler_dir / f"{target}_{m_key}_scaler.joblib")
        encoders[target] = joblib.load(scaler_dir / f"{target}_{m_key}_encoders.joblib")
        stats, X_train = compute_target_train_data(target)
        train_stats[target] = stats
        train_data[target] = X_train
        
    adapter = SurrogateAdapter(
        models=models,
        scalers=scalers,
        encoders=encoders,
        feature_names=feature_names,
        train_stats=train_stats,
        decision_space=DECISION_SPACE,
        X_train_data=train_data
    )
    return adapter

def test_surrogate_adapter_initialization(loaded_adapter):
    assert len(loaded_adapter.models) == 3
    assert len(loaded_adapter.scalers) == 3
    assert len(loaded_adapter.feature_names) == 3

def test_surrogate_adapter_predict_all_returns_three_floats(loaded_adapter):
    test_vec = np.array([0.1, 0.5, 8000.0, 0.1])
    energy, roughness, time_val = loaded_adapter.predict_all(test_vec)
    assert isinstance(energy, float)
    assert isinstance(roughness, float)
    assert isinstance(time_val, float)
    assert energy >= 0
    assert roughness >= 0
    assert time_val >= 0

def test_validate_decision_bounds(loaded_adapter):
    # Valid vector
    valid_vec = np.array([0.1, 0.5, 8000.0, 0.1])
    assert loaded_adapter.validate_decision_bounds(valid_vec) is True
    
    # Out of bounds
    invalid_vec1 = np.array([0.05, 0.5, 8000.0, 0.1])
    assert loaded_adapter.validate_decision_bounds(invalid_vec1) is False

def test_prediction_confidence_proximity(loaded_adapter):
    # Test valid/centered vector (should be close to some training samples, i.e., high confidence)
    valid_vec = np.array([0.1, 0.5, 8000.0, 0.1])
    min_dist, is_confident = loaded_adapter.check_prediction_confidence(valid_vec, threshold=2.0)
    assert isinstance(min_dist, float)
    assert isinstance(is_confident, bool)
    assert min_dist >= 0.0

def test_nsga2_problem_bounds(loaded_adapter):
    optimizer = EnergyOptNSGA2(
        adapter=loaded_adapter,
        decision_space=DECISION_SPACE,
        pop_size=10,
        n_gen=5
    )
    prob = optimizer._build_pymoo_problem()
    assert prob.n_var == 3
    assert prob.n_obj == 3
    assert len(prob.xl) == 3
    assert len(prob.xu) == 3

def test_nsga2_optimization_runs_and_saves(loaded_adapter, tmp_path):
    optimizer = EnergyOptNSGA2(
        adapter=loaded_adapter,
        decision_space=DECISION_SPACE,
        pop_size=10,
        n_gen=5,
        save_history=False
    )
    res = optimizer.run()
    assert "pareto_X" in res
    assert "pareto_F" in res
    assert res["n_solutions"] > 0
    assert res["hypervolume"] >= 0
    
    paths = optimizer.save_results(tmp_path)
    assert Path(paths["pareto_X"]).exists()
    assert Path(paths["pareto_F"]).exists()
    assert Path(paths["summary"]).exists()

def test_optimization_plots_generation(loaded_adapter, tmp_path):
    optimizer = EnergyOptNSGA2(
        adapter=loaded_adapter,
        decision_space=DECISION_SPACE,
        pop_size=10,
        n_gen=2,
        save_history=False
    )
    optimizer.run()
    
    path_3d = optimizer.plot_pareto_3d(tmp_path)
    path_2d = optimizer.plot_pareto_2d_projections(tmp_path)
    path_dist = optimizer.plot_objective_distributions(tmp_path)
    path_ranges = optimizer.plot_decision_variable_distributions(tmp_path)
    
    assert Path(path_3d).exists()
    assert Path(path_2d).exists()
    assert Path(path_dist).exists()
    assert Path(path_ranges).exists()
