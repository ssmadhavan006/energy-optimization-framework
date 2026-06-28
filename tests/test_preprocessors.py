import pytest
import pandas as pd
import numpy as np
from src.data.preprocessors import (
    detect_outliers_iqr,
    remove_outliers,
    impute_missing,
    encode_categoricals,
    scale_features,
    split_data
)

def test_outlier_detection_dtype():
    df = pd.DataFrame({'val': [1.0, 2.0, 1.5, 100.0, 2.5]})
    mask = detect_outliers_iqr(df, ['val'])
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
    assert mask.iloc[3] == True  # 100.0 is outlier
    assert mask.iloc[0] == False

def test_remove_outliers():
    df = pd.DataFrame({'val': [1.0, 2.0, 1.5, 100.0, 2.5]})
    df_clean = remove_outliers(df, ['val'])
    assert len(df_clean) == 4
    assert 100.0 not in df_clean['val'].values

def test_impute_missing_leaves_no_nulls():
    df = pd.DataFrame({'val': [1.0, np.nan, 1.5, np.nan, 2.5]})
    df_imp = impute_missing(df)
    assert df_imp['val'].isnull().sum() == 0
    assert df_imp['val'].iloc[1] == 1.5  # median of [1.0, 1.5, 2.5] is 1.5

def test_scale_features_minmax_range():
    df_train = pd.DataFrame({'val': [1.0, 2.0, 3.0]})
    df_test = pd.DataFrame({'val': [1.5, 4.0, 0.0]})
    
    df_train_sc, df_test_sc, scaler = scale_features(
        df_train, df_test, ['val'], method='minmax', save_name='test_minmax.joblib'
    )
    
    # Train scaled should be in [0, 1]
    assert df_train_sc['val'].min() == 0.0
    assert df_train_sc['val'].max() == 1.0
    
    # Test scaled can be outside [0, 1] if values are outside train bounds
    assert df_test_sc['val'].iloc[1] > 1.0  # 4.0 is > 3.0
    assert df_test_sc['val'].iloc[2] < 0.0  # 0.0 is < 1.0

def test_split_data_row_count():
    df = pd.DataFrame({
        'feat1': range(10),
        'feat2': range(10),
        'target': range(10)
    })
    
    X_train, X_test, y_train, y_test = split_data(df, 'target', test_size=0.3, seed=42)
    
    assert len(X_train) == 7
    assert len(X_test) == 3
    assert len(y_train) == 7
    assert len(y_test) == 3
    assert len(X_train) + len(X_test) == len(df)
