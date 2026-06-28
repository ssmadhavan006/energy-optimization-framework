import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from typing import List, Tuple, Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Preprocessors")

# Constants
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
SCALER_DIR = Path("outputs/models/scalers")

def detect_outliers_iqr(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    """
    Detects outliers using the Interquartile Range (IQR) method.
    
    Args:
        df: Input DataFrame.
        cols: List of numerical columns to check.
        
    Returns:
        pd.Series: Boolean mask where True represents an outlier in any of the specified columns.
    """
    mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found for outlier detection.")
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        col_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        mask = mask | col_mask
    return mask

def remove_outliers(df: pd.DataFrame, cols: List[str], method: str = 'iqr') -> pd.DataFrame:
    """
    Removes outliers from the DataFrame.
    
    Args:
        df: Input DataFrame.
        cols: Columns to check for outliers.
        method: Method to use ('iqr' is supported).
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    df_copy = df.copy()
    if method.lower() == 'iqr':
        outlier_mask = detect_outliers_iqr(df_copy, cols)
        num_outliers = outlier_mask.sum()
        logger.info(f"Removing {num_outliers} outliers using IQR method.")
        return df_copy[~outlier_mask].reset_index(drop=True)
    else:
        logger.warning(f"Unknown outlier detection method: {method}. Returning original DataFrame.")
        return df_copy

def impute_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Imputes missing values in the DataFrame.
    
    Args:
        df: Input DataFrame.
        strategy: Imputation strategy ('median', 'mean', or 'mode').
        
    Returns:
        pd.DataFrame: Imputed DataFrame.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            null_count = df_copy[col].isnull().sum()
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                if strategy == 'median':
                    fill_value = df_copy[col].median()
                elif strategy == 'mean':
                    fill_value = df_copy[col].mean()
                else:
                    fill_value = df_copy[col].mode().iloc[0] if not df_copy[col].mode().empty else 0
            else:
                fill_value = df_copy[col].mode().iloc[0] if not df_copy[col].mode().empty else "Unknown"
            
            df_copy[col] = df_copy[col].fillna(fill_value)
            logger.info(f"Imputed {null_count} missing values in '{col}' with {strategy} value: {fill_value}")
    return df_copy

def encode_categoricals(
    df: pd.DataFrame,
    encoders: Optional[Dict[str, LabelEncoder]] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encodes categorical features using LabelEncoder.
    If encoders dict is provided, uses them to transform.
    Otherwise fits new encoders.
    
    Args:
        df: Input DataFrame.
        encoders: Optional dictionary of fitted LabelEncoders.
        
    Returns:
        Tuple: (Encoded DataFrame, Dictionary of fitted LabelEncoders)
    """
    df_copy = df.copy()
    out_encoders = {} if encoders is None else encoders.copy()
    
    for col in df_copy.columns:
        dtype_str = str(df_copy[col].dtype).lower()
        is_cat = (df_copy[col].dtype == 'object' or 
                  'string' in dtype_str or 
                  'str' in dtype_str or 
                  isinstance(df_copy[col].dtype, pd.CategoricalDtype))
                  
        if is_cat:
            if encoders is not None and col in encoders:
                le = encoders[col]
                # Map unseen classes to the first class to handle cleanly
                vals = df_copy[col].astype(str).values
                unseen = ~np.isin(vals, le.classes_)
                if np.any(unseen):
                    logger.debug(f"Unseen categories in test column '{col}': {np.unique(vals[unseen])}. Mapping to first class '{le.classes_[0]}'.")
                    vals[unseen] = le.classes_[0]
                df_copy[col] = le.transform(vals)
            else:
                logger.info(f"Encoding categorical column: '{col}'")
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                out_encoders[col] = le
                
    return df_copy, out_encoders

def scale_features(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame, 
    cols: List[str], 
    method: str = 'minmax',
    save_name: str = 'scaler.joblib'
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Scales numerical features. Fits scaler on training data ONLY, and transforms test data.
    Saves the fitted scaler to outputs/models/scalers/.
    
    Args:
        df_train: Training DataFrame.
        df_test: Testing DataFrame.
        cols: Columns to scale.
        method: Scaling method ('minmax' or 'standard').
        save_name: Filename to save the fitted scaler.
        
    Returns:
        Tuple: (Scaled training DataFrame, Scaled testing DataFrame, Fitted Scaler object)
    """
    df_tr_copy = df_train.copy()
    df_te_copy = df_test.copy()
    
    if not cols:
        return df_tr_copy, df_te_copy, None
        
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")
        
    logger.info(f"Fitting {method} scaler on training data for columns {cols}...")
    df_tr_copy[cols] = scaler.fit_transform(df_tr_copy[cols])
    df_te_copy[cols] = scaler.transform(df_te_copy[cols])
    
    # Save the scaler
    SCALER_DIR.mkdir(parents=True, exist_ok=True)
    save_path = SCALER_DIR / save_name
    joblib.dump(scaler, save_path)
    logger.info(f"Saved fitted scaler to {save_path}")
    
    return df_tr_copy, df_te_copy, scaler

def split_data(
    df: pd.DataFrame, 
    target_cols: Union[str, List[str]], 
    test_size: float = DEFAULT_TEST_SIZE, 
    seed: int = DEFAULT_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame into features (X) and targets (y) for train/test datasets.
    
    Args:
        df: Input DataFrame.
        target_cols: Name(s) of target column(s).
        test_size: Ratio of test dataset.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    if isinstance(target_cols, str):
        target_cols = [target_cols]
        
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    X = df[feature_cols]
    y = df[target_cols]
    
    logger.info(f"Splitting data with test_size={test_size}, random_state={seed}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    
    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

if __name__ == '__main__':
    print("Testing preprocessors module...")
    # Create simple dummy dataframe
    data = {
        'val': [1.0, 2.0, 1.5, 100.0, 2.5, np.nan], # 100.0 is an outlier
        'cat': ['A', 'B', 'A', 'B', 'A', 'B']
    }
    df = pd.DataFrame(data)
    
    # Impute
    df_imp = impute_missing(df)
    print("Imputed:\n", df_imp)
    
    # Encode
    df_enc, encs = encode_categoricals(df_imp)
    print("Encoded:\n", df_enc)
    
    # Remove outliers
    df_clean = remove_outliers(df_enc, ['val'])
    print("Cleaned:\n", df_clean)
    
    # Split
    X_train, X_test, y_train, y_test = split_data(df_clean, 'cat')
    print(f"X_train rows: {len(X_train)}, X_test rows: {len(X_test)}")
    
    # Scale
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, ['val'], 'minmax', 'test_scaler.joblib')
    print("Scaled X_train:\n", X_train_sc)
    print("Scaled X_test:\n", X_test_sc)
