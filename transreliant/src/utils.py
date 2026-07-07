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
    """
    Append-only experiment log. Every cross_val_score call and every final
    test evaluation calls this — one row per (stage, model, metric).
    Never overwrites; this is the record that proves you compared
    alternatives instead of just picking XGBoost by default.
    """
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

    print(f"Logged: stage={stage}, model={model_name}, {cv_metric_name}={row['cv_metric_mean']}")