import os
import sys
import re
from pathlib import Path

def main():
    manuscript_path = Path("paper/manuscript.md")
    bib_path = Path("paper/references.bib")
    fig_dir = Path("paper/figures")
    
    if not manuscript_path.exists():
        print(f"Error: Manuscript not found at {manuscript_path}")
        sys.exit(1)
        
    text = manuscript_path.read_text(encoding="utf-8")
    
    # -------------------------------------------------------------
    # CHECK 1 & 6: Word counts per section
    # -------------------------------------------------------------
    print("=== CHECK 1 & 6: Word Counts per Section ===")
    
    # Define markers for sections
    section_markers = [
        ("Abstract", r"# Abstract"),
        ("Introduction", r"# 1. Introduction"),
        ("Related Work", r"# 2. Related Work"),
        ("Methodology", r"# 3. Methodology"),
        ("Datasets", r"# 4. Datasets and Preprocessing"),
        ("Results", r"# 5. Experimental Results"),
        ("Discussion", r"# 6. Discussion and Limitations"),
        ("Conclusion", r"# 7. Conclusion")
    ]
    
    # Partition text
    section_texts = {}
    for i in range(len(section_markers)):
        name, pattern = section_markers[i]
        start_match = re.search(pattern, text)
        if not start_match:
            print(f"[FAIL] Missing section header: {name} (pattern: {pattern})")
            section_texts[name] = ""
            continue
            
        start_idx = start_match.end()
        # End index is start of next section or end of file
        if i + 1 < len(section_markers):
            next_name, next_pattern = section_markers[i+1]
            next_match = re.search(next_pattern, text)
            end_idx = next_match.start() if next_match else len(text)
        else:
            end_idx = len(text)
            
        section_texts[name] = text[start_idx:end_idx].strip()
        
    # Word count limits
    limits = {
        "Abstract": (0, 250),
        "Introduction": (600, 900),
        "Related Work": (700, 1000),
        "Methodology": (1000, 1400),
        "Datasets": (500, 700),
        "Results": (1000, 1400),
        "Discussion": (800, 1100),
        "Conclusion": (350, 500)
    }
    
    total_words = 0
    all_sections_ok = True
    
    for name, s_text in section_texts.items():
        words = len(s_text.split())
        total_words += words
        low, high = limits[name]
        if low <= words <= high:
            print(f"[PASS] {name:15s}: {words:4d} words (Limit: {low}-{high})")
        else:
            print(f"[FAIL] {name:15s}: {words:4d} words (Limit: {low}-{high})")
            all_sections_ok = False
            
    if 5500 <= total_words <= 8000:
        print(f"[PASS] Total Word Count: {total_words} words (Limit: 5500-8000)")
    else:
        print(f"[FAIL] Total Word Count: {total_words} words (Limit: 5500-8000)")
        all_sections_ok = False
        
    print()
    
    # -------------------------------------------------------------
    # CHECK 2: Number consistency
    # -------------------------------------------------------------
    print("=== CHECK 2: Number Consistency ===")
    required_numbers = [
        "0.9947", "0.8061", "0.5002",
        "1.1359", "0.1237", "13.1220",
        "100", "0.0366", "0.8870",
        "4.2%", "44.2%", "0.50",
        "0.00", "0.053", "0.10"
    ]
    missing_numbers = []
    for num in required_numbers:
        if num not in text:
            missing_numbers.append(num)
            
    if not missing_numbers:
        print("[PASS] All key result numbers found in manuscript.")
    else:
        print("[FAIL] Missing key result numbers from manuscript:")
        for num in missing_numbers:
            print(f"  - {num}")
            
    print()
    
    # -------------------------------------------------------------
    # CHECK 3: Reference citation audit
    # -------------------------------------------------------------
    print("=== CHECK 3: Reference Citation Audit ===")
    # Find citations matching \cite{key} or \citep{key} or \citet{key} or [Author, Year]
    citations_tex = re.findall(r'\\cite[t|p]?\{([^}]+)\}', text)
    flat_citations = []
    for cit in citations_tex:
        for k in cit.split(','):
            flat_citations.append(k.strip())
            
    unique_citations = set(flat_citations)
    print(f"Found {len(unique_citations)} unique LaTeX/BibTeX citation keys.")
    
    # Load references.bib keys
    bib_keys = set()
    if bib_path.exists():
        bib_text = bib_path.read_text(encoding="utf-8")
        bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_text))
        print(f"Found {len(bib_keys)} reference keys in references.bib")
    else:
        print("[FAIL] references.bib not found!")
        
    missing_keys = unique_citations - bib_keys
    if not missing_keys:
        print("[PASS] All cited keys exist in references.bib.")
    else:
        print("[FAIL] Cited keys missing from references.bib:")
        for k in missing_keys:
            print(f"  - {k}")
            
    if len(bib_keys) >= 20:
        print(f"[PASS] Total references count: {len(bib_keys)} (Target: >= 20)")
    else:
        print(f"[FAIL] Total references count: {len(bib_keys)} (Target: >= 20)")
        
    print()
    
    # -------------------------------------------------------------
    # CHECK 4: Figure reference audit
    # -------------------------------------------------------------
    print("=== CHECK 4: Figure Reference Audit ===")
    fig_refs = re.findall(r'Figure~\\ref\{([^}]+)\}|Fig\.~\\ref\{([^}]+)\}', text)
    flat_fig_refs = []
    for ref in fig_refs:
        for item in ref:
            if item:
                flat_fig_refs.append(item)
                
    unique_fig_refs = set(flat_fig_refs)
    print(f"Found figure references: {list(unique_fig_refs)}")
    
    # Check corresponding files
    all_figs_exist = True
    fig_mapping = {
        "fig:framework": "Fig1_framework_overview.png",
        "fig:distributions": "Fig2_target_distributions.png",
        "fig:comparison": "Fig3_model_comparison.png",
        "fig:actual_vs_predicted": "Fig4_actual_vs_predicted.png",
        "fig:shap_importance": "Fig5_shap_importance.png",
        "fig:shap_beeswarm": "Fig6_shap_beeswarm_roughness.png",
        "fig:shap_dependence": "Fig7_feature_conflict.png",
        "fig:nsga2_convergence": "Fig8_nsga2_convergence.png",
        "fig:pareto_projections": "Fig9_pareto_projections.png",
        "fig:topsis_radar": "Fig10_topsis_radar.png",
        "fig:sensitivity_heatmap": "Fig11_sensitivity_heatmap.png",
        "fig:tool_wear_pareto_shift": "Fig12_tool_wear_pareto_shift.png",
        "fig:proximity_distances": "Fig13_proximity_distances.png"
    }
    
    for ref in unique_fig_refs:
        if ref in fig_mapping:
            filename = fig_mapping[ref]
            file_path = fig_dir / filename
            if file_path.exists():
                print(f"[PASS] {ref} -> {filename} exists.")
            else:
                print(f"[FAIL] {ref} -> {filename} DOES NOT EXIST.")
                all_figs_exist = False
        else:
            print(f"[WARN] Reference {ref} has no direct mapping in fig_mapping.")
            
    print()
    
    # -------------------------------------------------------------
    # CHECK 5: Forbidden phrases check
    # -------------------------------------------------------------
    print("=== CHECK 5: Forbidden Phrases Check ===")
    forbidden = [
        ("optimized the energy", "energy SEC was constant, cannot claim this"),
        ("reduced energy consumption", "not demonstrated"),
        ("significantly outperforms", "requires p-value citation"),
        ("proves that", "too strong for ML results"),
        ("guarantees", "never use in ML papers"),
        ("state of the art", "requires citation or avoid"),
        ("first paper to", "avoid priority claims"),
        ("novel", "avoid self-praise, use 'integrated' instead")
    ]
    
    forbidden_found = False
    for phrase, reason in forbidden:
        matches = list(re.finditer(re.escape(phrase), text, re.IGNORECASE))
        if matches:
            print(f"[FAIL] Found forbidden phrase: '{phrase}' ({len(matches)} occurrences) - Reason: {reason}")
            for m in matches[:3]:
                start = max(0, m.start() - 40)
                end = min(len(text), m.end() + 40)
                snippet = text[start:end].replace('\n', ' ')
                print(f"    Snippet: \"... {snippet} ...\"")
            forbidden_found = True
            
    if not forbidden_found:
        print("[PASS] No forbidden phrases found.")
        
    print()
    
    # -------------------------------------------------------------
    # CHECK 7: Section completeness
    # -------------------------------------------------------------
    print("=== CHECK 7: Section Completeness ===")
    subsections = [
        ("3.1.1 Multi-Dataset Surrogate Integration", r"Multi-Dataset Surrogate Integration"),
        ("3.1.2 Energy Target Engineering narrative", r"Energy Target Engineering"),
        ("3.4.1 Surrogate Validity and Bound Enforcement", r"Surrogate Validity and Bound Enforcement"),
        ("3.6 Reproducibility", r"Reproducibility"),
        ("5.5 Ablation Study", r"Ablation Study"),
        ("6.4 Alignment with Machining Theory", r"Alignment with Machining Theory"),
        ("6.5 Practical Deployment Workflow", r"Practical Deployment Workflow")
    ]
    
    all_subsections_exist = True
    for label, pattern in subsections:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            print(f"[PASS] Subsection found: {label}")
        else:
            print(f"[FAIL] Subsection MISSING: {label}")
            all_subsections_exist = False

if __name__ == "__main__":
    main()
