import os
import shutil
from pathlib import Path

def main():
    src_dir = Path(".")
    dest_dir = Path("paper/submission_package")
    
    # Create target directories
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "figures").mkdir(exist_ok=True)
    (dest_dir / "tables").mkdir(exist_ok=True)
    
    # Copy main documents
    shutil.copy2(src_dir / "paper/manuscript.md", dest_dir / "manuscript.md")
    shutil.copy2(src_dir / "paper/cover_letter.md", dest_dir / "cover_letter.md")
    shutil.copy2(src_dir / "paper/references.bib", dest_dir / "references.bib")
    
    # Copy figures
    src_fig_dir = src_dir / "paper/figures"
    dest_fig_dir = dest_dir / "figures"
    for f in src_fig_dir.glob("Fig*.png"):
        shutil.copy2(f, dest_fig_dir / f.name)
    shutil.copy2(src_fig_dir / "figure_captions.md", dest_fig_dir / "figure_captions.md")
    
    # Copy tables
    src_tab_dir = src_dir / "paper/tables"
    dest_tab_dir = dest_dir / "tables"
    for t in src_tab_dir.glob("table*.tex"):
        shutil.copy2(t, dest_tab_dir / t.name)
        
    print("Copied all source files to submission package.")
    
    # Create README_for_journal.txt
    readme_content = """SUBMISSION PACKAGE CONTENTS
============================
Main manuscript:  manuscript.md (convert to .tex or
                  .docx per journal instructions)
Cover letter:     cover_letter.md
Bibliography:     references.bib
Figures:          Fig1 through Fig13 (PNG, 300 DPI)
Figures captions: figure_captions.md
Tables:           table1 through table8 (LaTeX .tex)

REPRODUCTION:
Full code available at: https://github.com/your-username/EnergyOptAI
Python version: 3.12
Key packages: see pyproject.toml

WORD COUNT: 6,685 words
FIGURES: 13
TABLES: 8
REFERENCES: 20"""
    
    (dest_dir / "README_for_journal.txt").write_text(readme_content, encoding="utf-8")
    print("Created README_for_journal.txt")
    
    # Create highlights.md
    highlights_content = """MANUSCRIPT HIGHLIGHTS
=====================
• EnergyOptAI integrates SHAP, NSGA-II, and TOPSIS into the first unified CNC parameter optimization pipeline with explicit multi-dataset surrogate architecture.
• Feed rate is confirmed as the dominant driver of surface roughness (SHAP |r|=0.916) and tool wear as the key determinant of cycle time (|r|=0.958).
• NSGA-II generates 100 Pareto-optimal solutions under discrete scenarios, quantifying the quality-throughput trade-off.
• Ablation study confirms each framework component contributes measurably: SEC engineering alone improved energy R² from −13.3 to 0.50.
• Full open-source implementation ensures complete reproducibility of all reported results."""
    
    (dest_dir / "highlights.md").write_text(highlights_content, encoding="utf-8")
    print("Created highlights.md")
    
    # Create final_human_checklist.md in paper/ directory
    checklist_content = """# Items Requiring Human Action Before Submission

## 1. Author Information (CRITICAL)
- [ ] Add all author names to title page
- [ ] Add all author affiliations
- [ ] Designate corresponding author + email
- [ ] Add ORCID IDs if available

## 2. Ethics and Declarations
- [ ] Confirm no conflicts of interest
- [ ] Confirm no human subjects / ethics approval needed
- [ ] Confirm datasets are licensed for academic use:
      Mendeley: CC BY 4.0 ✅
      Kaggle: check adorigueto terms
      UCI: check Bosch dataset terms
- [ ] Write data availability statement:
      "All code and trained models are available at
      [GitHub URL]. Datasets used are publicly available
      at [URLs for each dataset]."

## 3. GitHub Repository
- [ ] Create public repository at github.com
- [ ] Push all project code
- [ ] Verify README renders correctly
- [ ] Get permanent DOI via Zenodo (optional but strong)
- [ ] Update [GitHub URL] placeholder in manuscript

## 4. Proofreading
- [ ] Read manuscript aloud once (catches flow issues)
- [ ] Have a colleague read Section 5 specifically
- [ ] Verify all equation numbering is sequential
- [ ] Verify all table numbering is sequential (1–8)
- [ ] Verify all figure numbering is sequential (1–13)
- [ ] Check that abstract matches conclusions (same numbers)

## 5. Journal Selection and Submission
- [ ] Confirm target journal (original CFP or alternative)
- [ ] Read journal's "Guide for Authors" fully
- [ ] Check word limit (this paper: 6,685 words)
- [ ] Check figure format requirements (PNG 300 DPI ✅)
- [ ] Check reference style (update .bib if needed)
- [ ] Register on journal's submission system (e.g., EM)
- [ ] Upload manuscript in required format
- [ ] Upload figures as separate files if required
- [ ] Upload cover letter
- [ ] Select appropriate editor / subject area
- [ ] Suggest 3–5 potential reviewers (optional but helpful)

## 6. Suggested Reviewers (prepare names)
Identify 3–5 researchers who have published in:
  - CNC energy optimization
  - Explainable AI in manufacturing
  - Multi-objective machining optimization
Do NOT suggest anyone from your institution.
Check that they have no conflicts of interest.

## 7. After Submission
- [ ] Save submission confirmation email
- [ ] Note manuscript number for tracking
- [ ] Expect first decision in 4–12 weeks (varies by journal)
- [ ] Prepare for revision — have all raw data accessible"""
    
    (src_dir / "paper/final_human_checklist.md").write_text(checklist_content, encoding="utf-8")
    shutil.copy2(src_dir / "paper/final_human_checklist.md", dest_dir / "final_human_checklist.md")
    print("Created final_human_checklist.md")

if __name__ == "__main__":
    main()
