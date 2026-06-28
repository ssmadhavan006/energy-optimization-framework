import logging
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loaders import load_all
from src.data.feature_engineering import engineer_all_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EDARunner")

# Setup plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# Output directory
EDA_OUT_DIR = Path("outputs/figures/eda")
EDA_OUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_missing_heatmap(df: pd.DataFrame, dataset_name: str):
    """Generates and saves a missing values heatmap."""
    logger.info(f"Generating missing values heatmap for {dataset_name}...")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title(f"Missing Values Heatmap — {dataset_name.capitalize()}")
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / f"{dataset_name}_missing_heatmap.png")
    plt.close()

def generate_distributions(df: pd.DataFrame, cols: list, dataset_name: str):
    """Generates distribution plots (histogram + KDE) for numerical columns."""
    logger.info(f"Generating distribution plots for {dataset_name}...")
    num_cols = len(cols)
    if num_cols == 0:
        return
    rows = (num_cols + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=(15, rows * 4))
    axs = axs.flatten()
    
    for idx, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axs[idx], color='dodgerblue')
        axs[idx].set_title(f"Dist of {col}")
        axs[idx].set_xlabel("")
        
    # Hide unused axes
    for i in range(idx + 1, len(axs)):
        fig.delaxes(axs[i])
        
    plt.suptitle(f"Numerical Column Distributions — {dataset_name.capitalize()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / f"{dataset_name}_distributions.png")
    plt.close()

def generate_correlation_heatmap(df: pd.DataFrame, cols: list, dataset_name: str):
    """Generates correlation heatmap for numerical columns."""
    logger.info(f"Generating correlation heatmap for {dataset_name}...")
    if not cols:
        return
    # Select only existing cols and drop non-numeric
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 2:
        return
    corr = df[valid_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation Heatmap — {dataset_name.capitalize()}")
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / f"{dataset_name}_correlation_heatmap.png")
    plt.close()

def generate_target_distributions(df: pd.DataFrame, target_cols: list, dataset_name: str):
    """Generates distribution plots for target variables."""
    logger.info(f"Generating target distribution plots for {dataset_name}...")
    valid_targets = [t for t in target_cols if t in df.columns]
    if not valid_targets:
        return
        
    fig, axs = plt.subplots(1, len(valid_targets), figsize=(5 * len(valid_targets), 4))
    if len(valid_targets) == 1:
        axs = [axs]
        
    for idx, target in enumerate(valid_targets):
        sns.histplot(df[target].dropna(), kde=True, ax=axs[idx], color='crimson')
        axs[idx].set_title(f"Target: {target}")
        
    plt.suptitle(f"Target Variable Distributions — {dataset_name.capitalize()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / f"{dataset_name}_target_distributions.png")
    plt.close()

def generate_scatter_matrix(df: pd.DataFrame, features: list, target: str, dataset_name: str):
    """Generates scatter plots of top features vs target."""
    logger.info(f"Generating scatter matrix for {dataset_name} vs {target}...")
    valid_features = [f for f in features if f in df.columns]
    if not valid_features or target not in df.columns:
        return
        
    rows = (len(valid_features) + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=(15, rows * 4))
    axs = axs.flatten()
    
    for idx, feat in enumerate(valid_features):
        sns.scatterplot(data=df, x=feat, y=target, ax=axs[idx], alpha=0.5, color='forestgreen')
        axs[idx].set_title(f"{feat} vs {target}")
        
    # Hide unused axes
    for i in range(idx + 1, len(axs)):
        fig.delaxes(axs[i])
        
    plt.suptitle(f"Features vs Target ({target}) — {dataset_name.capitalize()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / f"{dataset_name}_scatter_matrix.png")
    plt.close()

def generate_boxplots_by_group(df: pd.DataFrame, features: list, group_col: str, dataset_name: str):
    """Generates box plots of features grouped by a categorical column."""
    logger.info(f"Generating boxplots grouped by {group_col} for {dataset_name}...")
    valid_features = [f for f in features if f in df.columns]
    if not valid_features or group_col not in df.columns:
        return
        
    rows = (len(valid_features) + 2) // 3
    fig, axs = plt.subplots(rows, 3, figsize=(15, rows * 4))
    axs = axs.flatten()
    
    for idx, feat in enumerate(valid_features):
        sns.boxplot(data=df, x=group_col, y=feat, ax=axs[idx], palette="Set2")
        axs[idx].set_title(f"{feat} by {group_col}")
        
    # Hide unused axes
    for i in range(idx + 1, len(axs)):
        fig.delaxes(axs[i])
        
    plt.suptitle(f"Feature Boxplots Grouped by {group_col} — {dataset_name.capitalize()}", fontsize=16)
    plt.tight_layout()
    
    save_filename = f"{dataset_name}_boxplots_by_{group_col.lower()}.png"
    plt.savefig(EDA_OUT_DIR / save_filename)
    plt.close()

def generate_mendeley_time_series(df: pd.DataFrame):
    """Generates time-series energy consumption plot for Mendeley data."""
    logger.info("Generating time-series plot of energy consumption for Mendeley...")
    # Plot first few hundred rows of SP1 spindle speed and SP1 power to show dynamic behavior
    df_sample = df.iloc[:1000].copy()
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot power
    if 'ENERGY|S' in df_sample.columns:
        axs[0].plot(df_sample.index, df_sample['ENERGY|S'], color='coral', label='Spindle Energy (Ws)')
        axs[0].set_ylabel("Spindle Energy (Ws)")
        axs[0].legend()
        axs[0].set_title("Time-series Profile of Spindle Energy per Instruction Step")
        
    # Plot Spindle Speed
    if 'S' in df_sample.columns:
        axs[1].plot(df_sample.index, df_sample['S'], color='teal', label='Commanded Speed (RPM)')
        axs[1].set_ylabel("Spindle Speed (RPM)")
        axs[1].legend()
        axs[1].set_title("Time-series Profile of Spindle Speed")
        
    plt.xlabel("Sequence of NC Instructions")
    plt.tight_layout()
    plt.savefig(EDA_OUT_DIR / "mendeley_energy_timeseries.png")
    plt.close()

def run_eda():
    """Runs the full EDA pipeline on all datasets."""
    logger.info("Loading datasets for EDA...")
    datasets = load_all()
    
    # 1. Mendeley EDA
    mendeley_df = datasets['mendeley']['parsed']
    # Engineer features
    mendeley_df = engineer_all_features(mendeley_df, 'mendeley')
    
    mendeley_numeric = ['delta_X', 'delta_Y', 'delta_Z', 'delta_xy', 'delta_S', 'F_val', 'S', 'D_W']
    mendeley_targets = ['ENERGY|x', 'ENERGY|y', 'ENERGY|z', 'ENERGY|S', 'ENERGY|T']
    
    generate_missing_heatmap(mendeley_df, "mendeley")
    generate_distributions(mendeley_df, mendeley_numeric, "mendeley")
    generate_correlation_heatmap(mendeley_df, mendeley_numeric + mendeley_targets, "mendeley")
    generate_target_distributions(mendeley_df, mendeley_targets, "mendeley")
    generate_scatter_matrix(mendeley_df, mendeley_numeric, 'ENERGY|S', "mendeley")
    
    if 'Material' in mendeley_df.columns:
        generate_boxplots_by_group(mendeley_df, ['ENERGY|S', 'S', 'F_val'], 'Material', "mendeley")
    generate_mendeley_time_series(mendeley_df)
    
    # 2. Kaggle EDA
    kaggle_df = datasets['kaggle']
    # We clean up object columns for correlation plotting
    kaggle_numeric = ['ap', 'vc', 'f', 'Fx', 'Fy', 'Fz', 'F', 'CTime', 'TCond']
    
    generate_missing_heatmap(kaggle_df, "kaggle")
    generate_distributions(kaggle_df, ['ap', 'vc', 'f', 'Ra', 'CTime', 'F'], "kaggle")
    generate_correlation_heatmap(kaggle_df, kaggle_numeric + ['Ra'], "kaggle")
    generate_target_distributions(kaggle_df, ['Ra'], "kaggle")
    generate_scatter_matrix(kaggle_df, ['ap', 'vc', 'f', 'F'], 'Ra', "kaggle")
    
    if 'Material' in kaggle_df.columns:
        generate_boxplots_by_group(kaggle_df, ['Ra', 'F'], 'Material', "kaggle")
    elif 'Position' in kaggle_df.columns:
        # Group by Position (front/back) as alternative grouping
        generate_boxplots_by_group(kaggle_df, ['Ra', 'F'], 'Position', "kaggle")
        
    # 3. Bosch EDA
    bosch_df = datasets['bosch']
    bosch_df = engineer_all_features(bosch_df, 'bosch')
    
    bosch_numeric = [col for col in bosch_df.columns if bosch_df[col].dtype in [np.float64, np.int64]]
    
    generate_missing_heatmap(bosch_df, "bosch")
    # Take a subset of numeric columns for distributions to avoid crowded plots
    dist_cols = [c for c in bosch_numeric if 'rms' in c or 'peak' in c][:9]
    generate_distributions(bosch_df, dist_cols, "bosch")
    generate_correlation_heatmap(bosch_df, dist_cols, "bosch")
    generate_target_distributions(bosch_df, ['ch_0_rms'], "bosch") # Use ch_0_rms as dummy target variable
    generate_scatter_matrix(bosch_df, [c for c in dist_cols if 'rms' in c], 'ch_0_rms', "bosch")
    
    if 'Quality_Label' in bosch_df.columns:
        generate_boxplots_by_group(bosch_df, ['ch_0_rms', 'ch_1_rms', 'ch_2_rms'], 'Quality_Label', "bosch")
        
    logger.info("EDA pipeline completed successfully.")

if __name__ == '__main__':
    run_eda()
