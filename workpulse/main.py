import argparse
import logging

from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.train import run_training
from src.predict import run_prediction


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main(stage: str, input_path: str = None, model: str = "random_forest", output_path: str = None) -> None:
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

    elif stage == "predict":
        run_prediction(input_path, model_name=model, output_path=output_path)
        logger.info("Prediction completed.")

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
            "predict",
            "pipeline",
        ],
    )
    parser.add_argument("--input", type=str, default=None,
                         help="Path to CSV for batch prediction, or 'interactive' / omit for single-employee CLI prompt")
    parser.add_argument("--model", type=str, default="random_forest",
                         choices=["logistic", "random_forest", "xgboost"],
                         help="Which trained model to use for prediction")
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions CSV")

    args = parser.parse_args()

    main(args.stage, input_path=args.input, model=args.model, output_path=args.output)