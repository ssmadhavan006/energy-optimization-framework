import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.explainability.shap_analysis import SHAPAnalyzer
from src.models.random_forest_model import RandomForestModel

@pytest.fixture
def simple_analyzer(tmp_path):
    # Set random seed for reproducibility
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(50, 5),
                     columns=[f'f{i}' for i in range(5)])
    y = pd.Series(X['f0'] * 2 + X['f1'] + np.random.rand(50) * 0.1)
    
    m = RandomForestModel("test", "test_target")
    m.build({})
    m.train(X, y)
    
    analyzer = SHAPAnalyzer(
        model=m,
        X_train=X,
        X_test=X,
        y_test=y,
        target_name="test",
        target_unit="units",
        feature_names=X.columns.tolist(),
        output_dir=tmp_path
    )
    return analyzer

def test_shap_compute_runs_without_error(simple_analyzer):
    simple_analyzer.compute()
    assert simple_analyzer.shap_values is not None

def test_shap_values_shape_matches_data(simple_analyzer):
    simple_analyzer.compute()
    assert simple_analyzer.shap_values.values.shape[1] == 5

def test_feature_importance_sorted_descending(simple_analyzer):
    simple_analyzer.compute()
    imp = simple_analyzer.get_feature_importance()
    assert imp['mean_abs_shap'].is_monotonic_decreasing

def test_feature_importance_top_feature_is_f0(simple_analyzer):
    simple_analyzer.compute()
    imp = simple_analyzer.get_feature_importance()
    assert imp.iloc[0]['feature_name'] == 'f0'

def test_global_importance_plot_saves_file(simple_analyzer):
    simple_analyzer.compute()
    path = simple_analyzer.plot_global_importance_bar()
    assert Path(path).exists()
    assert Path(path).suffix == '.png'

def test_shap_values_save_creates_csv(simple_analyzer):
    simple_analyzer.compute()
    path = simple_analyzer.save_shap_values()
    assert Path(path).exists()
    df = pd.read_csv(path)
    assert df.shape[1] == 6  # 5 features + index column

def test_engineering_insights_returns_list(simple_analyzer):
    simple_analyzer.compute()
    insights = simple_analyzer.get_engineering_insights()
    assert isinstance(insights, list)
    assert len(insights) >= 1
    assert all(isinstance(s, str) for s in insights)

def test_waterfall_saves_file(simple_analyzer):
    simple_analyzer.compute()
    path = simple_analyzer.plot_waterfall_local(sample_idx=0)
    assert Path(path).exists()
