
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent / "src"))

from config_loader import load_config
import train_stage1
import train_stage2
import evaluate


def main():
    cfg = load_config()



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