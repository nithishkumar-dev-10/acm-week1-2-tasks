"""
main.py
TalentIQ — Pipeline Entry Point

Phase 1 : Setup only (this file is a stub)
Phase 2+ : Uncomment the relevant stages below

Usage:
    python main.py --stage all
    python main.py --stage preprocess
    python main.py --stage train
    python main.py --stage evaluate
"""
import argparse
from src.utils import get_logger, set_seed

log = get_logger()


def run(stage: str):
    set_seed(42)
    log.info(f"TalentIQ pipeline started — stage: {stage}")

    if stage in ("all", "preprocess"):
        log.info("[ Stage 1 ] Preprocessing — (Phase 2)")
        # from src.preprocessing import run_preprocessing
        # run_preprocessing()

    if stage in ("all", "features"):
        log.info("[ Stage 2 ] Feature Engineering — (Phase 2)")
        # from src.feature_engineering import run_feature_engineering
        # run_feature_engineering()

    if stage in ("all", "train"):
        log.info("[ Stage 3 ] Model Training — (Phase 3)")
        # from src.train import run_training
        # run_training()

    if stage in ("all", "evaluate"):
        log.info("[ Stage 4 ] Evaluation — (Phase 4)")
        # from src.metrics import run_evaluation
        # run_evaluation()

    log.info("Pipeline finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TalentIQ ML Pipeline")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "preprocess", "features", "train", "evaluate"],
        help="Which pipeline stage to run"
    )
    args = parser.parse_args()
    run(args.stage)
