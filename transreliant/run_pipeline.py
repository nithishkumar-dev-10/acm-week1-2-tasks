
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

    print(
        "\nPipeline complete.\n"
        "  Artifacts -> artifacts/models/\n"
        "  Figures   -> reports/figures/\n"
        "  Metrics   -> reports/metrics/\n"
        "  Logs      -> logs/experiment_log.csv"
    )


if __name__ == "__main__":
    main()