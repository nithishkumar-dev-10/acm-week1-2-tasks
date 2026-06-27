import argparse
import logging

from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.train import run_training


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main(stage: str) -> None:
    """Run the requested pipeline stage."""

    if stage == "preprocess":
        preprocess_data()
        logger.info("Preprocessing completed.")

    elif stage == "features":
        df = preprocess_data()
        engineer_features(df)
        logger.info("Feature engineering completed.")

    elif stage == "train":
        run_training()
        logger.info("Training completed.")

    elif stage == "pipeline":
        run_training()
        logger.info("Full pipeline completed.")

    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        type=str,
        default="pipeline",
        choices=[
            "preprocess",
            "features",
            "train",
            "pipeline",
        ],
    )

    args = parser.parse_args()

    main(args.stage)
