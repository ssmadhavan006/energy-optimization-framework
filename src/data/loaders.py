import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataLoaders")

def find_dataset_dir(keywords: list) -> Path:
    """
    Dynamically discovers a dataset directory in the dataset/ folder using keywords.
    
    Args:
        keywords: List of string keywords to search for in folder names.
        
    Returns:
        Path to the discovered directory.
        
    Raises:
        FileNotFoundError: If no matching directory is found.
    """
    dataset_base = Path("dataset")
    if not dataset_base.exists():
        raise FileNotFoundError(f"Base dataset directory '{dataset_base}' not found.")
        
    for item in dataset_base.iterdir():
        if item.is_dir():
            if any(kw.lower() in item.name.lower() for kw in keywords):
                return item
                
    raise FileNotFoundError(f"Could not find dataset directory with keywords {keywords} in '{dataset_base}'")

def load_mendeley() -> Dict[str, pd.DataFrame]:
    """
    Loads the Mendeley CNC High-Frequency Energy Repository.
    If a parsed CSV cache exists at outputs/results/mendeley_parsed.csv, it loads that.
    Otherwise, it recursively parses the raw JSON files and caches the result.
    
    Returns:
        Dict of DataFrames: {'parsed': df_parsed, 'metadata': df_metadata}
    """
    logger.info("Loading Mendeley CNC High-Frequency Energy Repository...")
    
    try:
        mendeley_dir = find_dataset_dir(["mendeley", "high-frequency"])
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
        
    cache_path = Path("outputs/results/mendeley_parsed.csv")
    metadata_path = mendeley_dir / "CNC Machining Data Respository.xlsx"
    
    # Load metadata sheet
    df_metadata = pd.DataFrame()
    if metadata_path.exists():
        try:
            df_metadata = pd.read_excel(metadata_path)
            logger.info(f"Loaded metadata from {metadata_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load metadata excel: {e}")
            
    if cache_path.exists():
        logger.info(f"Loading cached parsed Mendeley data from {cache_path}")
        df_parsed = pd.read_csv(cache_path)
        return {'parsed': df_parsed.copy(), 'metadata': df_metadata.copy()}
        
    # No cache: parse raw json files
    logger.info("Cache not found. Parsing raw JSON files (this may take a few seconds)...")
    json_dir = mendeley_dir / "Raw Datasets (.json)"
    if not json_dir.exists():
        raise FileNotFoundError(f"Raw JSON directory '{json_dir}' not found in Mendeley dataset.")
        
    json_files = list(json_dir.glob("**/*.json"))
    json_files = [f for f in json_files if "config" not in f.name]
    
    all_parsed_dfs = []
    
    for file_path in json_files:
        logger.info(f"Parsing file: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            payload = data.get("Payload", [])
            
            # Extract HFData
            hf_rows = []
            for block in payload:
                if 'HFData' in block:
                    hf_rows.extend(block['HFData'])
                    
            if not hf_rows:
                logger.warning(f"No HFData found in {file_path.name}")
                continue
                
            df_hf = pd.DataFrame(hf_rows)
            df_hf.set_index(0, inplace=True)
            
            # Extract GCode block events
            block_events = []
            for block in payload:
                if 'HFBlockEvent' in block:
                    evt = block['HFBlockEvent']
                    block_events.append({
                        'cycle': evt['HFProbeCounter'],
                        'gcode': evt['GCode'],
                        'ipo_gc': evt['IpoGC'],
                        'tool': evt.get('ActiveTool', 0)
                    })
                    
            if not block_events:
                logger.warning(f"No G-code blocks found in {file_path.name}")
                continue
                
            df_events = pd.DataFrame(block_events)
            df_events.sort_values('cycle', inplace=True)
            df_events.reset_index(drop=True, inplace=True)
            
            parsed_records = []
            
            for i in range(len(df_events)):
                start_cycle = df_events.loc[i, 'cycle']
                end_cycle = df_events.loc[i+1, 'cycle'] if i+1 < len(df_events) else df_hf.index.max() + 1
                
                interval_data = df_hf.loc[start_cycle:end_cycle - 1]
                if interval_data.empty:
                    continue
                    
                pos_x = interval_data[1]
                pos_y = interval_data[2]
                pos_z = interval_data[3]
                
                delta_x = abs(pos_x.iloc[-1] - pos_x.iloc[0])
                delta_y = abs(pos_y.iloc[-1] - pos_y.iloc[0])
                delta_z = abs(pos_z.iloc[-1] - pos_z.iloc[0])
                delta_xy = np.sqrt(delta_x**2 + delta_y**2)
                
                energy_x = interval_data[71].astype(float).sum() * 0.002
                energy_y = interval_data[72].astype(float).sum() * 0.002
                energy_z = interval_data[73].astype(float).sum() * 0.002
                energy_s = interval_data[75].astype(float).sum() * 0.002
                energy_t = (interval_data[74].astype(float).sum() + 
                            interval_data[76].astype(float).sum() + 
                            interval_data[77].astype(float).sum()) * 0.002
                
                sp_speeds = interval_data[54].astype(float)
                s_val = abs(sp_speeds.mean()) / 6.0
                delta_s = (abs(sp_speeds.iloc[-1]) - abs(sp_speeds.iloc[0])) / 6.0
                
                feed_x = interval_data[50].astype(float)
                feed_y = interval_data[51].astype(float)
                feed_z = interval_data[52].astype(float)
                feed_vector = np.sqrt(feed_x**2 + feed_y**2 + feed_z**2)
                f_val = feed_vector.mean() * 60.0
                
                gcode_str = df_events.loc[i, 'gcode']
                d_w = 16.0
                if 'DM8' in gcode_str or 'DM8' in str(df_events.loc[i, 'tool']):
                    d_w = 8.0
                    
                toolchange = 1 if 'M6' in gcode_str else 0
                turn_op = 0
                
                # Material type: derived from directory name
                material_str = file_path.parent.name.lower()
                material = 'Aluminum' if 'alu' in material_str else 'Plastic'
                
                parsed_records.append({
                    'Commands': df_events.loc[i, 'ipo_gc'],
                    'delta_X': delta_x,
                    'delta_Y': delta_y,
                    'delta_Z': delta_z,
                    'delta_xy': delta_xy,
                    'delta_S': delta_s,
                    'F_val': f_val,
                    'S': s_val,
                    'D_W': d_w,
                    'Toolchange': toolchange,
                    'TurnOp': turn_op,
                    'Material': material,
                    'Workpiece': file_path.parent.name,
                    'Filename': file_path.name,
                    'ENERGY|x': energy_x,
                    'ENERGY|y': energy_y,
                    'ENERGY|z': energy_z,
                    'ENERGY|S': energy_s,
                    'ENERGY|T': energy_t,
                    'time_s': len(interval_data) * 0.002,
                })
                
            df_file_parsed = pd.DataFrame(parsed_records)
            all_parsed_dfs.append(df_file_parsed)
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path.name}: {e}")
            
    if not all_parsed_dfs:
        raise ValueError("Failed to parse any Mendeley JSON files.")
        
    df_parsed = pd.concat(all_parsed_dfs, ignore_index=True)
    
    # Save parsed CSV to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_parsed.to_csv(cache_path, index=False)
    logger.info(f"Saved parsed Mendeley data cache to {cache_path}")
    
    return {'parsed': df_parsed.copy(), 'metadata': df_metadata.copy()}

def load_kaggle_roughness() -> pd.DataFrame:
    """
    Loads and combines all CSV files in the Kaggle CNC Turning Roughness folder.
    
    Returns:
        DataFrame containing turning roughness data.
    """
    logger.info("Loading Kaggle CNC Turning Roughness Dataset...")
    
    try:
        kaggle_dir = find_dataset_dir(["kaggle", "roughness"])
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
        
    csv_files = list(kaggle_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in Kaggle directory '{kaggle_dir}'")
        
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add file source indicator
            df['Source'] = csv_file.name
            dfs.append(df)
            logger.info(f"Loaded Kaggle file: {csv_file.name} with shape {df.shape}")
        except Exception as e:
            logger.error(f"Error reading {csv_file.name}: {e}")
            
    if not dfs:
        raise ValueError("Failed to load any Kaggle CSV files.")
        
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Coerce numeric columns that might contain 'na' or other strings to numeric type
    numeric_cols = [
        'ap', 'vc', 'f', 'Ra', 'Rz', 'Rsk', 'Rku', 'RSm', 'Rt', 
        'Fx', 'Fy', 'Fz', 'F', 'CTime', 'TCond', 'Machined_length', 
        'Init_diameter', 'Final_diameter', 'Replica', 'Condition'
    ]
    for col in df_combined.columns:
        if col in numeric_cols:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
            
    logger.info(f"Combined Kaggle dataset shape: {df_combined.shape}")
    return df_combined.copy()

def load_bosch() -> pd.DataFrame:
    """
    Loads the Bosch CNC Machining (UCI) vibration dataset.
    Extracts statistical features from the triaxial accelerometer signals in each .h5 file.
    If a parsed CSV cache exists at outputs/results/bosch_parsed.csv, it loads that.
    
    Returns:
        DataFrame containing structured statistical features from Bosch vibration data.
    """
    logger.info("Loading Bosch CNC Machining (UCI) Dataset...")
    
    try:
        bosch_dir = find_dataset_dir(["bosch", "uci"])
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
        
    cache_path = Path("outputs/results/bosch_parsed.csv")
    if cache_path.exists():
        logger.info(f"Loading cached parsed Bosch data from {cache_path}")
        df_parsed = pd.read_csv(cache_path)
        return df_parsed.copy()
        
    logger.info("Cache not found. Processing Bosch H5 files (this may take a few seconds)...")
    data_dir = bosch_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found in Bosch dataset.")
        
    h5_files = list(data_dir.glob("**/*.h5"))
    logger.info(f"Found {len(h5_files)} H5 files.")
    
    processed_records = []
    
    for file_path in h5_files:
        try:
            # Extract metadata from file path
            # Structure: data / machine / process / label / file_name
            parts = file_path.parts
            # We locate 'data' index in path parts
            data_idx = parts.index("data")
            
            machine = parts[data_idx + 1]
            process = parts[data_idx + 2]
            label = parts[data_idx + 3]
            filename = file_path.name
            
            # Read vibration data
            with h5py.File(file_path, "r") as f:
                vibration = f["vibration_data"][:]
                
            # Extract features (RMS, Std, Mean, Peak) for X (0), Y (1), Z (2) channels
            features = {
                "Machine": machine,
                "Process": process,
                "Quality_Label": label,
                "Filename": filename
            }
            
            for ch in range(3):
                ch_name = f"ch_{ch}"
                data_ch = vibration[:, ch]
                
                features[f"{ch_name}_mean"] = np.mean(data_ch)
                features[f"{ch_name}_std"] = np.std(data_ch)
                features[f"{ch_name}_rms"] = np.sqrt(np.mean(data_ch**2))
                features[f"{ch_name}_max"] = np.max(data_ch)
                features[f"{ch_name}_min"] = np.min(data_ch)
                features[f"{ch_name}_peak"] = np.max(np.abs(data_ch))
                
            processed_records.append(features)
            
        except Exception as e:
            logger.warning(f"Error parsing Bosch file {file_path.name}: {e}")
            
    if not processed_records:
        raise ValueError("Failed to parse any Bosch H5 files.")
        
    df_parsed = pd.DataFrame(processed_records)
    
    # Save parsed CSV to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_parsed.to_csv(cache_path, index=False)
    logger.info(f"Saved parsed Bosch data cache to {cache_path} with shape {df_parsed.shape}")
    
    return df_parsed.copy()

def load_all() -> Dict[str, Any]:
    """
    Loads all three datasets and returns them in a dictionary.
    
    Returns:
        Dict: {'mendeley': dict, 'kaggle': DataFrame, 'bosch': DataFrame}
    """
    logger.info("Loading all datasets...")
    mendeley_data = load_mendeley()
    kaggle_data = load_kaggle_roughness()
    bosch_data = load_bosch()
    
    return {
        'mendeley': mendeley_data,
        'kaggle': kaggle_data,
        'bosch': bosch_data
    }

if __name__ == '__main__':
    # Test block
    print("Testing loaders...")
    d = load_all()
    print("Mendeley parsed shape:", d['mendeley']['parsed'].shape)
    print("Kaggle combined shape:", d['kaggle'].shape)
    print("Bosch parsed shape:", d['bosch'].shape)
