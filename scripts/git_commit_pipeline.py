import subprocess
import os

def run_cmd(args):
    result = subprocess.run(args, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"Error running: {' '.join(args)}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
    else:
        print(f"Success: {' '.join(args)}")
    return result

def main():
    # 1. Configure local git credentials just in case they are not set globally
    run_cmd(["git", "config", "user.name", "Researcher"])
    run_cmd(["git", "config", "user.email", "researcher@energyoptai.org"])

    # Define the 20 commits sequence
    commits = [
        {
            "files": [".gitignore", ".python-version", "pyproject.toml", "uv.lock"],
            "msg": "chore: initialize repository structure and dependency files"
        },
        {
            "files": ["src/utils/logger.py", "src/utils/config.py"],
            "msg": "feat(utils): implement logger and baseline configurations"
        },
        {
            "files": ["src/data/loaders.py"],
            "msg": "feat(data): implement multi-dataset loaders for Mendeley and Kaggle"
        },
        {
            "files": ["src/data/preprocessors.py"],
            "msg": "feat(data): implement preprocessing logic with IQR outlier removal and median imputation"
        },
        {
            "files": ["src/data/feature_engineering.py"],
            "msg": "feat(data): implement active G-code filtering and Specific Energy Consumption engineering"
        },
        {
            "files": ["src/models/base_model.py"],
            "msg": "feat(models): implement BaseModel wrapper with prediction interval estimation"
        },
        {
            "files": ["src/models/baseline_models.py"],
            "msg": "feat(models): add Linear Regression and Support Vector Regression baselines"
        },
        {
            "files": [
                "src/models/xgboost_model.py",
                "src/models/catboost_model.py",
                "src/models/random_forest_model.py",
                "src/models/model_registry.py"
            ],
            "msg": "feat(models): implement XGBoost, CatBoost, and Random Forest ensemble models"
        },
        {
            "files": ["src/evaluation/metrics.py"],
            "msg": "feat(eval): add performance evaluation metrics (RMSE, MAE, R2, MAPE)"
        },
        {
            "files": ["src/evaluation/statistical_tests.py"],
            "msg": "feat(eval): add Wilcoxon signed-rank test for model performance significance"
        },
        {
            "files": ["src/explainability/shap_analysis.py"],
            "msg": "feat(xai): implement SHAP TreeExplainer for global and local explainability"
        },
        {
            "files": ["src/optimization/decision_space.py", "src/optimization/surrogate_adapter.py"],
            "msg": "feat(opt): add decision space adapters and surrogate model interfaces for pymoo"
        },
        {
            "files": ["src/optimization/nsga2_optimizer.py"],
            "msg": "feat(opt): implement vectorized NSGA-II multi-objective genetic algorithm"
        },
        {
            "files": ["src/optimization/topsis.py"],
            "msg": "feat(opt): implement TOPSIS multi-criteria decision making ranker"
        },
        {
            "files": ["src/optimization/sensitivity.py"],
            "msg": "feat(opt): implement TOPSIS weight scenario sensitivity analysis"
        },
        {
            "files": ["src/optimization/proximity_validator.py"],
            "msg": "feat(opt): implement nearest-neighbor out-of-distribution (OOD) validator"
        },
        {
            "files": ["src/utils/runtime_tracker.py"],
            "msg": "feat(utils): implement execution stage runtime tracker"
        },
        {
            "files": ["tests/"],
            "msg": "test: add comprehensive automated unit tests for all pipeline modules"
        },
        {
            "files": ["scripts/"],
            "msg": "feat(scripts): add pipeline execution scripts for training, explainability, optimization, and reporting"
        },
        {
            "files": ["README.md", "LICENSE", "architecture_image.png", "docs/architecture.md", "outputs/", "main.py"],
            "msg": "docs: add license, README setup instructions, architecture diagram, and output results"
        }
    ]

    # Run the staging and committing loop
    for i, commit in enumerate(commits, 1):
        print(f"\n--- Commit {i}/20 ---")
        for f in commit["files"]:
            if os.path.exists(f):
                run_cmd(["git", "add", f])
            else:
                # Try directory contents if file doesn't exist directly or wildcards
                run_cmd(["git", "add", f])
        
        # Verify if anything is staged before committing
        status = run_cmd(["git", "status", "--porcelain"])
        if status.stdout.strip():
            run_cmd(["git", "commit", "-m", commit["msg"]])
        else:
            print(f"Skipping commit {i} - nothing staged.")

    # 3. Push to origin master
    print("\n--- Pushing to GitHub ---")
    run_cmd(["git", "push", "-u", "origin", "master", "--force"])

if __name__ == "__main__":
    main()
