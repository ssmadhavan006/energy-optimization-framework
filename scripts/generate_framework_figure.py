import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib
from pathlib import Path
matplotlib.use('Agg')

def main():
    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 14), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15)
    ax.axis('off')
    
    # Title
    ax.text(5, 14.5, "EnergyOptAI: Integrated Framework Architecture", 
            fontsize=14, fontweight='bold', ha='center', va='center', color='#1A1A1A')
    ax.text(5, 14.1, "CNC Energy-Efficient Machining Parameter Selection Pipeline", 
            fontsize=10, ha='center', va='center', color='#555555')
    
    # Helper to draw a box
    def draw_stage_box(ax, x, y, w, h, title, details, bg_color, text_color='white'):
        # Draw fancy box
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                             facecolor=bg_color, edgecolor='none', mutation_scale=1)
        ax.add_patch(box)
        # Title text
        ax.text(x + w/2, y + h - 0.4, title, fontsize=11, fontweight='bold', 
                ha='center', va='center', color=text_color)
        # Detail items
        dy = (h - 0.8) / (len(details) + 1)
        for idx, detail in enumerate(details):
            ax.text(x + w/2, y + h - 0.9 - idx*dy, detail, fontsize=9, 
                    ha='center', va='center', color=text_color, style='italic')
            
    # Stage 1 - Data Layer
    draw_stage_box(ax, 1.5, 11.2, 7.0, 2.0, 
                   "Stage 1: Heterogeneous Data Layer", 
                   ["Mendeley High-Frequency Spindle Power Dataset",
                    "Kaggle CNC Turning Surface Roughness Dataset",
                    "Bosch UCI Industrial Machining Dataset"], 
                   "#4472C4")
    
    # Stage 2 - Preprocessing
    draw_stage_box(ax, 1.5, 8.6, 7.0, 2.0, 
                   "Stage 2: Preprocessing & Target Engineering", 
                   ["IQR Outlier Filter & Median Imputation",
                    "Specific Energy Consumption (SEC) Target Engineering",
                    "Feature Scaling & 80/20 Train-Test Splitting"], 
                   "#ED7D31")
    
    # Stage 3 - ML Prediction Surrogates
    draw_stage_box(ax, 1.5, 6.0, 7.0, 2.0, 
                   "Stage 3: Machine Learning Surrogate Models", 
                   ["Spindle Specific Energy model (Random Forest, R2 = 0.5002)",
                    "Surface Roughness model (CatBoost, R2 = 0.8061)",
                    "Machining Time model (CatBoost, R2 = 0.9947)"], 
                   "#70AD47")
    
    # Stage 4 - Explainability Layer
    draw_stage_box(ax, 1.5, 3.4, 7.0, 2.0, 
                   "Stage 4: SHAP Explainability & Interaction Analysis", 
                   ["TreeExplainer Feature Impact Rankings",
                    "Directional beeswarms and dependency curves",
                    "Cross-target feature conflict analysis (Roughness vs. Energy)"], 
                   "#FFC000", text_color='#1A1A1A')
    
    # Stage 5 - Optimization + MCDM
    draw_stage_box(ax, 1.5, 0.8, 7.0, 2.0, 
                   "Stage 5: NSGA-II Optimization & TOPSIS MCDM", 
                   ["NSGA-II Pareto-optimal search (42 non-dominated solutions)",
                    "TOPSIS vector normalization compromise ranking",
                    "Sensitivity analysis across 6 weight scenarios"], 
                   "#7030A0")
    
    # Final Recommendation Box
    rec_box = FancyBboxPatch((1.5, -0.6), 7.0, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='#404040', edgecolor='none')
    ax.add_patch(rec_box)
    ax.text(5, -0.2, "FINAL COMPROMISE RECOMMENDATION", fontsize=9, fontweight='bold', 
            ha='center', va='center', color='white')
    ax.text(5, -0.5, "Feed: 0.1037 mm/rev | Cut Depth: 0.3357 mm | Spindle: 10201 rpm | Wear: 0.0380 mm", 
            fontsize=8, ha='center', va='center', color='#D9D9D9')

    # Draw Connecting Arrows
    def draw_arrow(ax, y_start, y_end, label):
        arrow = mpatches.FancyArrowPatch((5.0, y_start), (5.0, y_end), 
                                         arrowstyle='-|>', mutation_scale=15, 
                                         color='#555555', linewidth=1.5)
        ax.add_patch(arrow)
        ax.text(5.2, (y_start + y_end)/2, label, fontsize=8, color='#555555', 
                ha='left', va='center', fontweight='semibold')
        
    draw_arrow(ax, 11.2, 10.6, "Raw Data Streams")
    draw_arrow(ax, 8.6, 8.0, "SEC & Cleaned Parameters")
    draw_arrow(ax, 6.0, 5.4, "Prediction Models (Surrogates)")
    draw_arrow(ax, 3.4, 2.8, "Explainability and Constraints")
    draw_arrow(ax, 0.8, 0.2, "Pareto Front")
    
    # Save the figure
    output_path = Path("paper/figures/Fig1_framework_overview.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Framework Overview figure to {output_path}")

if __name__ == '__main__':
    main()
