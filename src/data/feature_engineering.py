import logging
import numpy as np
import pandas as pd
from typing import Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FeatureEngineering")

def compute_cutting_speed(diameter: Union[float, pd.Series], spindle_speed: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """
    Computes cutting speed (v_c) in m/min.
    Formula: v_c = pi * D * N / 1000
    
    Args:
        diameter: Tool diameter in mm.
        spindle_speed: Spindle speed in rpm.
        
    Returns:
        Cutting speed in m/min.
    """
    return np.pi * diameter * spindle_speed / 1000.0

def compute_mrr(
    feed_rate: Union[float, pd.Series], 
    depth_of_cut: Union[float, pd.Series], 
    width_of_cut: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Computes Material Removal Rate (MRR) in mm^3/min.
    Formula: MRR = f * ap * ae
    
    Args:
        feed_rate: Feed rate in mm/min.
        depth_of_cut: Depth of cut (ap) in mm.
        width_of_cut: Width of cut / stepover (ae) in mm.
        
    Returns:
        Material Removal Rate in mm^3/min.
    """
    return feed_rate * depth_of_cut * width_of_cut

def compute_specific_energy(
    energy: Union[float, pd.Series], 
    mrr: Union[float, pd.Series], 
    time: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Computes Specific Energy Consumption (SEC) in Wh/mm^3 or J/mm^3.
    Formula: SEC = Energy / (MRR * time)
    Note: Handle division by zero using np.where or replacing zero/inf.
    
    Args:
        energy: Energy consumed (in Joules or Wh).
        mrr: Material Removal Rate (in mm^3/min or mm^3/sec).
        time: Machining time (in seconds or minutes, matching the MRR time unit).
        
    Returns:
        Specific Energy Consumption.
    """
    denominator = mrr * time
    if isinstance(denominator, pd.Series):
        return np.where(denominator > 0.0, energy / denominator, 0.0)
    else:
        return energy / denominator if denominator > 0.0 else 0.0

def compute_load_factor(
    spindle_load_pct: Union[float, pd.Series], 
    time: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Computes spindle load factor.
    Formula: load_factor = spindle_load * time
    
    Args:
        spindle_load_pct: Spindle load in percent (%).
        time: Duration in seconds.
        
    Returns:
        Spindle load factor (load * time).
    """
    return spindle_load_pct * time

def compute_tool_wear_ratio(
    wear: Union[float, pd.Series], 
    max_wear: float
) -> Union[float, pd.Series]:
    """
    Computes tool wear ratio.
    Formula: tool_wear_ratio = wear / max_wear
    
    Args:
        wear: Current tool wear value.
        max_wear: Maximum tool wear limit.
        
    Returns:
        Tool wear ratio.
    """
    if max_wear <= 0.0:
        return 0.0 if not isinstance(wear, pd.Series) else pd.Series(0.0, index=wear.index)
    return wear / max_wear

def engineer_all_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Applies relevant engineered features to the DataFrame based on the dataset.
    Only computes features where the source columns exist, otherwise skips with a warning.
    
    Args:
        df: Input DataFrame.
        dataset_name: Name of the dataset ('mendeley', 'kaggle', or 'bosch').
        
    Returns:
        pd.DataFrame: DataFrame with engineered features added.
    """
    df_copy = df.copy()
    dataset_name = dataset_name.lower()
    
    if 'mendeley' in dataset_name:
        # Mendeley features: D_W is tool diameter, S is spindle speed, F_val is feed rate
        # Energy columns: ENERGY|x, ENERGY|y, ENERGY|z, ENERGY|S, ENERGY|T
        # We can compute cutting speed (v_c)
        if 'D_W' in df_copy.columns and 'S' in df_copy.columns:
            logger.info("Computing cutting speed for Mendeley...")
            df_copy['cutting_speed_vc'] = compute_cutting_speed(df_copy['D_W'], df_copy['S'])
        else:
            logger.warning("Missing D_W or S columns to compute cutting speed in Mendeley.")
            
        # We can compute MRR. We need depth of cut (ap) and width of cut (ae).
        # Since these are G-code instructions, we can assume a standard depth/width of cut
        # or skip if columns are not present. In Mendeley, let's look if they exist.
        # If not, let's skip.
        
    elif 'kaggle' in dataset_name:
        # Kaggle features:
        # ap is depth of cut (mm), vc is cutting speed (m/min), f is feed rate (mm/rev).
        # Tool_ID or Tool wear columns? TCond could represent tool wear.
        # Let's see: we have ap, vc, f.
        # We can derive Spindle Speed: N = 1000 * vc / (pi * D).
        # We can compute MRR if we know width of cut (ae) and feed speed (mm/min).
        # Feed speed: F = f * N (mm/min).
        # Let's derive Spindle Speed N first. We need tool diameter D.
        # If tool diameter is not available, we can assume a standard diameter (e.g. 50mm or 100mm) or skip.
        # Let's write robust conditional checks.
        pass
        
    elif 'bosch' in dataset_name:
        # Bosch features are vibration channel RMS, std, mean, peak.
        # We can compute triaxial magnitude features: mag = sqrt(x^2 + y^2 + z^2)
        logger.info("Computing triaxial magnitude features for Bosch vibration data...")
        for stat in ['mean', 'std', 'rms', 'max', 'peak']:
            cols = [f"ch_{ch}_{stat}" for ch in range(3)]
            if all(c in df_copy.columns for c in cols):
                df_copy[f"vibration_mag_{stat}"] = np.sqrt(
                    df_copy[cols[0]]**2 + df_copy[cols[1]]**2 + df_copy[cols[2]]**2
                )
                
    return df_copy

def engineer_mendeley_energy_target(
    df: pd.DataFrame,
    spindle_energy_col: str = 'ENERGY|S',
    mrr_col: str = None,
    feed_rate_col: str = None,
    depth_of_cut_col: str = None,
    width_of_cut_col: str = None,
    time_col: str = 'time_s',
    min_spindle_energy_threshold: float = 1e-6
) -> pd.DataFrame:
    """
    Re-engineers the Mendeley dataset to produce a valid energy
    prediction target: Specific Energy Consumption (SEC) or Spindle Power (Watts).

    Strategy:
    1. FILTER: Identify rows with active spindle cutting (spindle energy > threshold, no G0 moves).
    2. AGGREGATE: Group consecutive G1/G2/G3 cutting blocks into unified operation units.
    3. COMPUTE MRR & SEC: Compute Material Removal Rate and SEC (J/mm^3).
    4. FALLBACK: Compute Average Spindle Power (Watts).
    5. VALIDATE: Filter out invalid/extreme values.

    Args:
        df: Raw parsed Mendeley DataFrame.
        spindle_energy_col: Column with spindle energy in Joules.
        mrr_col: Column name of pre-computed MRR (if any).
        feed_rate_col: Column name of feed rate (default: F_val).
        depth_of_cut_col: Column name of depth of cut (ap).
        width_of_cut_col: Column name of width of cut (ae).
        time_col: Column name of block duration in seconds (default: time_s).
        min_spindle_energy_threshold: Spindle energy threshold to classify cutting.

    Returns:
        pd.DataFrame with 'sec', 'spindle_power_w', 'mrr' added.
    """
    df_copy = df.copy()
    
    # 1. FILTER: Identify active cutting blocks
    is_cutting = (df_copy[spindle_energy_col] > min_spindle_energy_threshold)
    if 'Commands' in df_copy.columns:
        is_cutting = is_cutting & (df_copy['Commands'] != 'G0')
    if 'S' in df_copy.columns:
        is_cutting = is_cutting & (df_copy['S'] > 0.0)
        
    df_copy['is_cutting'] = is_cutting
    
    # 2. AGGREGATE: Group consecutive cutting blocks within the same filename
    if 'Filename' in df_copy.columns:
        block_change = df_copy['is_cutting'] != df_copy['is_cutting'].shift()
        file_change = df_copy['Filename'] != df_copy['Filename'].shift()
        df_copy['operation_group'] = (block_change | file_change).cumsum()
    else:
        block_change = df_copy['is_cutting'] != df_copy['is_cutting'].shift()
        df_copy['operation_group'] = block_change.cumsum()
        
    # Aggregate only cutting rows
    df_cutting = df_copy[df_copy['is_cutting']].copy()
    if df_cutting.empty:
        logger.warning("No active cutting rows found in Mendeley dataset.")
        return pd.DataFrame()
        
    group_cols = ['operation_group']
    if 'Filename' in df_cutting.columns:
        group_cols.append('Filename')
        
    # Define aggregation mapping
    agg_dict = {}
    for col in df_cutting.columns:
        if col in group_cols or col in ['is_cutting', 'block_change', 'file_change']:
            continue
        # Energy and time are summed
        if 'ENERGY|' in col or col == time_col or col.startswith('delta_'):
            agg_dict[col] = 'sum'
        # Parameters are averaged
        else:
            if pd.api.types.is_numeric_dtype(df_cutting[col]):
                agg_dict[col] = 'mean'
            else:
                agg_dict[col] = 'first'
                
    agg_dict[spindle_energy_col] = 'sum'
    if time_col in df_cutting.columns:
        agg_dict[time_col] = 'sum'
        
    df_agg = df_cutting.groupby(group_cols).agg(agg_dict).reset_index()
    df_agg['n_blocks_aggregated'] = df_cutting.groupby(group_cols).size().values
    
    # 3. COMPUTE MRR & SEC
    feed_rate_val = df_agg[feed_rate_col] if (feed_rate_col and feed_rate_col in df_agg.columns) else df_agg.get('F_val', None)
    
    # Width of cut ae proxy
    ae = None
    if width_of_cut_col and width_of_cut_col in df_agg.columns:
        ae = df_agg[width_of_cut_col]
    elif 'D_W' in df_agg.columns:
        ae = df_agg['D_W'] / 2.0
        logger.warning("Width of cut column missing. Using D_W / 2 as proxy.")
        
    # Depth of cut ap proxy
    ap = None
    if depth_of_cut_col and depth_of_cut_col in df_agg.columns:
        ap = df_agg[depth_of_cut_col]
    else:
        ap = 1.0
        logger.warning("Depth of cut column missing. Using 1.0 mm as proxy.")
        
    # MRR (mm^3/min)
    if feed_rate_val is not None and ae is not None and ap is not None:
        df_agg['mrr'] = feed_rate_val * ap * ae
        # Volume machined (mm^3) = (feed_rate / 60) * ap * ae * time_s
        volume_mm3 = (feed_rate_val / 60.0) * ap * ae * df_agg[time_col]
        # SEC = ENERGY|S / Volume (J/mm^3)
        df_agg['sec'] = np.where(volume_mm3 > 0.0, df_agg[spindle_energy_col] / volume_mm3, np.nan)
    else:
        logger.warning("Missing feed_rate, depth_of_cut or width_of_cut. Cannot compute SEC.")
        df_agg['mrr'] = 0.0
        df_agg['sec'] = np.nan
        
    # 4. COMPUTE Spindle Power fallback (Watts)
    if time_col in df_agg.columns:
        df_agg['spindle_power_w'] = df_agg[spindle_energy_col] / df_agg[time_col]
    else:
        df_agg['spindle_power_w'] = 0.0
        
    # 5. VALIDATE
    original_n = len(df)
    filtered_n = len(df_cutting)
    
    # Remove rows where time_s is 0 or negative
    if time_col in df_agg.columns:
        df_agg = df_agg[df_agg[time_col] > 0.0]
        
    # Validate SEC and drop nan/inf values if SEC was computed
    df_final = df_agg[df_agg['sec'].notna() & (df_agg['sec'] > 0.0)].copy()
    if not df_final.empty:
        # Filter extreme outliers (>99th percentile)
        sec_99 = df_final['sec'].quantile(0.99)
        df_final = df_final[df_final['sec'] <= sec_99]
        
    final_n = len(df_final)
    logger.info(f"Target re-engineering completed. Original rows: {original_n}, filtered cutting: {filtered_n}, final aggregated: {final_n}")
    
    return df_final


if __name__ == '__main__':
    print("Testing feature engineering module...")
    # Test cutting speed
    vc = compute_cutting_speed(16.0, 10000.0)
    print(f"Cutting speed for D=16mm, N=10000rpm: {vc:.2f} m/min")
    assert np.isclose(vc, 502.6548245743669)
    
    # Test MRR
    mrr = compute_mrr(500.0, 3.0, 10.0)
    print(f"MRR for f=500mm/min, ap=3mm, ae=10mm: {mrr:.2f} mm^3/min")
    assert mrr == 15000.0
