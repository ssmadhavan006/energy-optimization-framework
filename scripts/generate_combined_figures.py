import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from PIL import Image
matplotlib.use('Agg')

def combine_three_images(img1_path, img2_path, img3_path, output_path, title, panel_labels):
    """Combines three images side-by-side in a 1x3 grid with panel labels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    
    paths = [img1_path, img2_path, img3_path]
    for idx, (ax, path, label) in enumerate(zip(axes, paths, panel_labels)):
        if Path(path).exists():
            img = Image.open(path)
            ax.imshow(np.array(img))
            ax.axis('off')
            # Add panel label (a), (b), (c)
            ax.text(0.01, 0.99, label, transform=ax.transAxes, fontsize=14, 
                    fontweight='bold', va='top', ha='left', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            ax.text(0.5, 0.5, f"Missing: {Path(path).name}", 
                    ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Combined figure saved to {output_path}")

def main():
    # Fig 4: Actual vs Predicted
    combine_three_images(
        img1_path="outputs/figures/results/energy_actual_vs_predicted.png",
        img2_path="outputs/figures/results/roughness_actual_vs_predicted.png",
        img3_path="outputs/figures/results/time_actual_vs_predicted.png",
        output_path="paper/figures/Fig4_actual_vs_predicted.png",
        title="Actual vs. Predicted Values for Energy SEC, Surface Roughness, and Machining Time",
        panel_labels=["(a) Energy SEC (Random Forest)", "(b) Surface Roughness (CatBoost)", "(c) Machining Time (CatBoost)"]
    )
    
    # Fig 5: SHAP Global Importance
    combine_three_images(
        img1_path="outputs/figures/shap/energy_shap_global_importance.png",
        img2_path="outputs/figures/shap/roughness_shap_global_importance.png",
        img3_path="outputs/figures/shap/time_shap_global_importance.png",
        output_path="paper/figures/Fig5_shap_importance.png",
        title="Global Feature Importance (Mean Absolute SHAP Value) Comparison",
        panel_labels=["(a) Energy SEC Model", "(b) Surface Roughness Model", "(c) Machining Time Model"]
    )

if __name__ == '__main__':
    main()
