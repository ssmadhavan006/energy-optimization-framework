import pytest
import pandas as pd
from src.data.loaders import load_all

def test_load_all():
    """
    Tests that load_all() runs successfully, returns a dict with expected keys,
    and each DataFrame has > 0 rows and no columns that are 100% null.
    """
    datasets = load_all()
    
    # Check return type
    assert isinstance(datasets, dict)
    assert 'mendeley' in datasets
    assert 'kaggle' in datasets
    assert 'bosch' in datasets
    
    # Validate Mendeley
    mendeley_data = datasets['mendeley']
    assert isinstance(mendeley_data, dict)
    assert 'parsed' in mendeley_data
    df_mendeley = mendeley_data['parsed']
    assert len(df_mendeley) > 0
    # Check that no columns are 100% null
    for col in df_mendeley.columns:
        assert df_mendeley[col].isnull().sum() < len(df_mendeley), f"Column {col} is 100% null"
        
    # Validate Kaggle
    df_kaggle = datasets['kaggle']
    assert isinstance(df_kaggle, pd.DataFrame)
    assert len(df_kaggle) > 0
    for col in df_kaggle.columns:
        assert df_kaggle[col].isnull().sum() < len(df_kaggle), f"Column {col} is 100% null"
        
    # Validate Bosch
    df_bosch = datasets['bosch']
    assert isinstance(df_bosch, pd.DataFrame)
    assert len(df_bosch) > 0
    for col in df_bosch.columns:
        assert df_bosch[col].isnull().sum() < len(df_bosch), f"Column {col} is 100% null"
