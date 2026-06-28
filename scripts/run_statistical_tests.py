import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Ensure PYTHONPATH includes project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluation.statistical_tests import StatisticalTester
from src.models.model_registry import MODEL_REGISTRY, TARGETS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RunStats")

def main():
    tester = StatisticalTester()
    models = list(MODEL_REGISTRY.keys())
    
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    paper_table_dir = Path("paper/tables")
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    
    # We will generate statistical significance tests for each of the three targets
    for target in TARGETS:
        logger.info(f"=== Running Pairwise Wilcoxon Signed-Rank Tests for target '{target}' ===")
        
        try:
            df = tester.run_all_pairwise(target, models)
            if df.empty:
                logger.warning(f"No CV scores found or compiled for target {target}")
                continue
                
            csv_path = out_dir / f"statistical_tests_{target}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved stats CSV to {csv_path}")
            
            # Print df
            print(f"\nSignificance tests for {target.upper()}:")
            print(df[["model_a", "model_b", "mean_a", "mean_b", "p_value", "significant"]])
            
            # Generate LaTeX
            latex_table = tester.generate_significance_table_latex(df, target)
            
            tex_path = out_dir / f"statistical_tests_{target}.tex"
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
                
            # Save to paper table folder
            paper_tex_path = paper_table_dir / f"table8_stats_{target}.tex"
            with open(paper_tex_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
                
            logger.info(f"Saved LaTeX tables for target {target}")
        except Exception as e:
            logger.exception(f"Failed to process statistical tests for target {target}: {e}")

if __name__ == "__main__":
    main()
