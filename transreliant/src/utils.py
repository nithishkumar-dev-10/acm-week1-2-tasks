# src/utils.py
import csv
from pathlib import Path
from datetime import datetime


def log_experiment(
    stage: str,
    model_name: str,
    params: dict = None,
    cv_metric_name: str = "",
    cv_metric_mean: float = None,
    cv_metric_std: float = None,
    test_metric_name: str = "",
    test_metric_value: float = None,
    log_path: str = "logs/experiment_log.csv",
) -> None:

    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_file.exists()

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "model_name": model_name,
        "params": str(params) if params else "",
        "cv_metric_name": cv_metric_name,
        "cv_metric_mean": round(cv_metric_mean, 5) if cv_metric_mean is not None else "",
        "cv_metric_std": round(cv_metric_std, 5) if cv_metric_std is not None else "",
        "test_metric_name": test_metric_name,
        "test_metric_value": round(test_metric_value, 5) if test_metric_value is not None else "",
    }

    with open(log_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # BUG FIX: the old print always referenced cv_metric_name/cv_metric_mean, even when a
    # call only logged a test metric (cv_metric_mean=None). That made the console output
    # disagree with what was actually written to the CSV. This now prints whichever metric(s)
    # were actually passed in, so console output always matches the logged row.
    logged_parts = [f"stage={stage}", f"model={model_name}"]
    if cv_metric_name and row["cv_metric_mean"] != "":
        logged_parts.append(f"{cv_metric_name}={row['cv_metric_mean']}")
    if test_metric_name and row["test_metric_value"] != "":
        logged_parts.append(f"{test_metric_name}={row['test_metric_value']}")

    print("Logged: " + ", ".join(logged_parts))