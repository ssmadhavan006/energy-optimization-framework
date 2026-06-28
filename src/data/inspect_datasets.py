import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def inspect_datasets():
    """
    Recursively inspects the dataset/ directory and generates a comprehensive report
    saved to outputs/results/dataset_inspection_report.txt and a clean summary in the terminal.
    """
    dataset_dir = Path("dataset")
    report_path = Path("outputs/results/dataset_inspection_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory '{dataset_dir}' does not exist.", file=sys.stderr)
        return

    # Store file details for summary table
    all_files = []
    
    # We will write to a list and join at the end to write to report file
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ENERGYOPTAI — DATASET INSPECTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(root) / file
            # Relative path to dataset directory for cleaner printing
            rel_path = file_path.relative_to(dataset_dir.parent)
            ext = file_path.suffix.lower()
            size_bytes = file_path.stat().st_size
            size_kb = size_bytes / 1024.0
            
            all_files.append({
                "path": str(rel_path),
                "extension": ext,
                "size_kb": size_kb
            })
            
            report_lines.append(f"FILE: {rel_path}")
            report_lines.append(f"Extension: {ext}")
            report_lines.append(f"Size: {size_kb:.2f} KB ({size_bytes} bytes)")
            report_lines.append("-" * 40)
            
            # Read tabular data if matches extensions
            df = None
            if ext == '.csv':
                try:
                    df = pd.read_csv(file_path, nrows=100) # Read head first to verify encoding/shape
                    # Re-read fully for stats
                    df = pd.read_csv(file_path)
                except Exception as e:
                    report_lines.append(f"Error reading CSV: {e}")
            elif ext in ['.xlsx', '.xls']:
                try:
                    df = pd.read_excel(file_path)
                except Exception as e:
                    report_lines.append(f"Error reading Excel: {e}")
            elif ext == '.parquet':
                try:
                    df = pd.read_parquet(file_path)
                except Exception as e:
                    report_lines.append(f"Error reading Parquet: {e}")
            
            if df is not None:
                report_lines.append(f"Shape: {df.shape}")
                report_lines.append("Columns & Data Types:")
                for col in df.columns:
                    null_count = df[col].isnull().sum()
                    null_pct = (null_count / len(df)) * 100
                    report_lines.append(f"  - {col} ({df[col].dtype}): {null_count} nulls ({null_pct:.2f}%)")
                
                report_lines.append("\nBasic Statistics:")
                # We format describe() output nicely
                stats_str = df.describe(include='all').to_string()
                report_lines.append(stats_str)
                
                report_lines.append("\nFirst 3 Rows:")
                rows_str = df.head(3).to_string()
                report_lines.append(rows_str)
            else:
                report_lines.append("Non-tabular or unreadable file.")
            
            report_lines.append("\n" + "=" * 80 + "\n")

    # Write report to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    # Print clean summary table to terminal
    print("\n" + "=" * 80)
    print(f"{'DATASET FILE SUMMARY TABLE':^80}")
    print("=" * 80)
    print(f"{'File Path':<60} | {'Ext':<6} | {'Size (KB)':<10}")
    print("-" * 80)
    for f in all_files:
        path_str = f['path']
        if len(path_str) > 57:
            path_str = "..." + path_str[-54:]
        print(f"{path_str:<60} | {f['extension']:<6} | {f['size_kb']:>9.2f}")
    print("=" * 80)
    print(f"Full detailed inspection report saved to: {report_path.resolve()}\n")

if __name__ == "__main__":
    inspect_datasets()
