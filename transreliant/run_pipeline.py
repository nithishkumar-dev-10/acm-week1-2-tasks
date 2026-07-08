# run_pipeline.py
"""
Step 18 — single end-to-end runnable script.
One command, empty artifacts/ folder -> trained artifacts + figures + metrics:

    python run_pipeline.py

Matches your ACTUAL src/ files rather than the roadmap's illustrative
pseudocode:
- train_stage1.main() already runs the full Step 10+11 sequence internally
  (baseline compare -> tune on CV AUC -> evaluate on test -> save
  preprocessor_stage1.pkl + stage1_classifier.pkl separately).
- train_stage2.main() does the same for Step 14+15 (compare -> tune on
  CV RMSE -> evaluate on test -> save preprocessor_stage2.pkl +
  stage2_regressor.pkl).
- evaluate.evaluate_stage1(cfg) / evaluate_stage2(cfg) then run Step 12
  (threshold sweep, writes chosen threshold back into config.yaml) and
  Step 13/16 (all confusion matrix / ROC / PR / feature-importance /
  pred-vs-actual / residuals figures + metrics CSVs).

NOTE on data_cleaning.py / feature_engineering.py: those two aren't
wired in below because they weren't part of what you shared. This script
currently assumes data/processed/featured.csv already exists (it must,
since your Stage 1/2 training already ran off it through Step 16). If you
have those two scripts with clean(df, cfg) / add_features(df) functions,
uncomment the block marked TODO to make this a true empty-folder ->
trained-artifacts run.
"""
import sys
from pathlib import Path

# so `from config_loader import load_config` etc. (used inside src/*.py)
# resolve the same way they do when those files are run directly from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_loader import load_config
import train_stage1
import train_stage2
import evaluate


def main():
    cfg = load_config()

    # ---- TODO: uncomment if you have data_cleaning.py + feature_engineering.py ----
    # import pandas as pd
    # from data_cleaning import clean
    # from feature_engineering import add_features
    #
    # raw_df = pd.read_csv(cfg["data"]["raw"])
    # cleaned_df = clean(raw_df, cfg)
    # Path(cfg["data"]["cleaned"]).parent.mkdir(parents=True, exist_ok=True)
    # cleaned_df.to_csv(cfg["data"]["cleaned"], index=False)
    #
    # featured_df = add_features(cleaned_df)
    # featured_df.to_csv(cfg["data"]["featured"], index=False)
    # ---------------------------------------------------------------------------

    featured_path = Path(cfg["data"]["featured"])
    if not featured_path.exists():
        raise FileNotFoundError(
            f"{featured_path} not found. Run your data_cleaning.py / "
            f"feature_engineering.py steps first, or wire them into the "
            f"TODO block above, before running the full pipeline."
        )
    print(f"Using existing featured data at {featured_path}\n")

    print("=== Stage 1: baseline compare -> tune -> evaluate -> save ===")
    train_stage1.main()

    print("\n=== Stage 2: baseline compare -> tune -> evaluate -> save ===")
    train_stage2.main()

    print("\n=== Evaluation: threshold optimization + figures + metrics ===")
    evaluate.evaluate_stage1(cfg)   # Steps 12+13
    evaluate.evaluate_stage2(cfg)   # Step 16

    print("\n=== System-level (cascade) evaluation ===")
    evaluate.evaluate_system(cfg)   # Step 20

    print(
        "\nPipeline complete.\n"
        "  Artifacts -> artifacts/models/\n"
        "  Figures   -> reports/figures/\n"
        "  Metrics   -> reports/metrics/ (incl. system_metrics.csv)\n"
        "  Logs      -> logs/experiment_log.csv"
    )


if __name__ == "__main__":
    main()