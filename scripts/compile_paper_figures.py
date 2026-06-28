import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
matplotlib.use('Agg')

def combine_two_images(img1_path, img2_path, output_path, title, panel_labels):
    """Combines two images side-by-side in a 1x2 grid with panel labels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')
    paths = [img1_path, img2_path]
    for idx, (ax, path, label) in enumerate(zip(axes, paths, panel_labels)):
        if Path(path).exists():
            img = Image.open(path)
            ax.imshow(np.array(img))
            ax.axis('off')
            ax.text(0.01, 0.99, label, transform=ax.transAxes, fontsize=12, 
                    fontweight='bold', va='top', ha='left', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.text(0.5, 0.5, f"Missing: {Path(path).name}", 
                    ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Combined 2-panel figure saved to {output_path}")

def combine_three_images(img1_path, img2_path, img3_path, output_path, title, panel_labels):
    """Combines three images side-by-side in a 1x3 grid with panel labels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    paths = [img1_path, img2_path, img3_path]
    for idx, (ax, path, label) in enumerate(zip(axes, paths, panel_labels)):
        if Path(path).exists():
            img = Image.open(path)
            ax.imshow(np.array(img))
            ax.axis('off')
            ax.text(0.01, 0.99, label, transform=ax.transAxes, fontsize=12, 
                    fontweight='bold', va='top', ha='left', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.text(0.5, 0.5, f"Missing: {Path(path).name}", 
                    ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            
    plt.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Combined 3-panel figure saved to {output_path}")

def main():
    fig_dir = Path("paper/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Combine Fig 2: target distributions
    combine_two_images(
        img1_path="outputs/figures/eda/mendeley_target_distributions.png",
        img2_path="outputs/figures/eda/kaggle_target_distributions.png",
        output_path=fig_dir / "Fig2_target_distributions.png",
        title="Exploratory Data Analysis: Target Variable Distributions",
        panel_labels=["(a) Mendeley Energy SEC Dataset", "(b) Kaggle Turning Dataset"]
    )
    
    # 2. Combine Fig 3: model comparison R2
    combine_three_images(
        img1_path="outputs/figures/results/energy_model_comparison_r2.png",
        img2_path="outputs/figures/results/roughness_model_comparison_r2.png",
        img3_path="outputs/figures/results/time_model_comparison_r2.png",
        output_path=fig_dir / "Fig3_model_comparison.png",
        title="Model Performance R² Comparison Across Targets",
        panel_labels=["(a) Energy SEC models", "(b) Surface Roughness models", "(c) Machining Time models"]
    )
    
    # 3. Copy other single figures
    copies = [
        ("outputs/figures/shap/roughness_shap_beeswarm.png", "Fig6_shap_beeswarm_roughness.png"),
        ("outputs/figures/shap/energy_roughness_conflict.png", "Fig7_feature_conflict.png"),
        ("outputs/figures/pareto/nsga2_convergence.png", "Fig8_nsga2_convergence.png"),
        ("outputs/figures/pareto/pareto_2d_projections.png", "Fig9_pareto_projections.png"),
        ("outputs/figures/topsis/ranking_radar.png", "Fig10_topsis_radar.png"),
        ("outputs/figures/topsis/sensitivity_heatmap.png", "Fig11_sensitivity_heatmap.png")
    ]
    
    manifest_entries = [
        "| Figure | File Name | Source Path | Section | Description |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| Figure 1 | Fig1_framework_overview.png | Generated programmatically | Section 3.1 | Framework flow architecture. |",
        "| Figure 2 | Fig2_target_distributions.png | Combined EDA distributions | Section 4.1-4.2 | Target variable distribution panels. |",
        "| Figure 3 | Fig3_model_comparison.png | Combined model comparisons | Section 5.1 | Model R² bar charts. |",
        "| Figure 4 | Fig4_actual_vs_predicted.png | Combined results scatter plots | Section 5.1 | Actual vs predicted fit comparisons. |",
        "| Figure 5 | Fig5_shap_importance.png | Combined SHAP importance | Section 5.2 | Global SHAP feature importances. |"
    ]
    
    for src, dest in copies:
        src_path = Path(src)
        dest_path = fig_dir / dest
        if src_path.exists():
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
            status = "Copied"
        else:
            print(f"Warning: Source not found {src_path}")
            status = "MISSING"
            
        fig_num = dest.split("_")[0].replace("Fig", "Figure ")
        desc = dest.split("_", 2)[2].replace(".png", "").replace("_", " ").capitalize()
        sec = "Section 5" if "shap" in dest or "conflict" in dest else "Section 5.3" if "convergence" in dest or "projections" in dest else "Section 5.4"
        manifest_entries.append(f"| {fig_num} | {dest} | {src} | {sec} | {desc} ({status}). |")
        
    # Write Manifest
    manifest_path = fig_dir / "figure_manifest.md"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("# EnergyOptAI — Manuscript Figures Manifest\n\n")
        f.write("\n".join(manifest_entries))
        f.write("\n")
    print(f"Saved figure manifest to {manifest_path}")

if __name__ == '__main__':
    main()
